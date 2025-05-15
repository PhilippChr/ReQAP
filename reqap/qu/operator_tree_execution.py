import traceback
import stopit
import time as timing
from loguru import logger
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, List, Dict, Tuple, Any, Optional

# required for `eval` inference
import json
from datetime import date, time, datetime, timedelta 
from dateutil.relativedelta import relativedelta

from reqap.library.library import store_jsonl
from reqap.classes.qu_execution_res import QUExecutionResult
from reqap.retrieval.retrieval import Retrieval
from reqap.extract.extract_module import ExtractModule
from reqap.qu.operator_tree import OperatorTree
from reqap.qu.qu_operators import (
    RETRIEVE,
    SELECT,
    GROUP_BY,
    JOIN,
    FILTER,
    APPLY,
    MAP,
    UNNEST,
    SUM,
    AVG,
    MAX,
    MIN,
    ARGMAX,
    ARGMIN
)


class OperatorTreeExecution:
    RESTRICTED_GLOBALS = {"__builtins__": {}} 
    RESTRICTED_LOCALS = {}

    EXECUTION_DICTS = {
        "RETRIEVE(": "RETRIEVE(self.retrieval, self.extract, ",
        "SELECT(": "SELECT(self.extract, self.persona_dict, ",
        "EXTRACT(": "SELECT(self.extract, self.persona_dict, ",
    }

    def __init__(self, qu_config: DictConfig, retrieval: "Retrieval", extract: ExtractModule, persona_dict: Optional[Dict]=None, dev_mode: bool=False):
        self.retrieval = retrieval
        self.extract = extract
        self.dev_mode = dev_mode  # in this mode, any failure in execution will raise an Exception, which will not be catched
        self.cache = QUExecutionCache(qu_config.qu_execution_cache_size)
        if persona_dict is None:
            logger.warning("Initiated OperatorTreeExecution with persona_dict=None")    
        self.persona_dict = persona_dict

    def derive_result(self, operator_trees: List[OperatorTree], run_all: bool=False, reference_date: date=date.today(), error_file: Optional[str]=None, time_budget: int=600) -> Tuple[Dict, Any | None, bool]:
        result = self.run(operator_trees=operator_trees, run_all=run_all, reference_date=reference_date, error_file=error_file, timeout=time_budget)
        failed = False
        if result is None:
            result_dict = dict()   
            derived_answer = None
            failed = True
        else:
            try:
                result_dict = result.to_dict()
                derived_answer = str(result.execution_result.result)
            except Exception as e:
                logger.error(f"Error when processing result: {e}. result={result}")
                result_dict = dict()
                derived_answer = None
        return result_dict, derived_answer, failed

    @stopit.threading_timeoutable(default=None)
    def run(self, operator_trees: List[OperatorTree], run_all: bool=False, reference_date: date=date.today(), error_file: Optional[str]=None) -> OperatorTree | None | List[OperatorTree | None]:
        """
        Execute the provided Operator Trees.
        When `run_all` is set to True, all runs are executed,
        even if the first one succeeds. 
        """
        results = list()
        for operator_tree in operator_trees:
            try:
                res = self.run_operator_tree(operator_tree, reference_date)
                if run_all:
                    results.append(res)
                else:
                    return res
            except (self.OperatorTreeExecutionError, TimeoutError) as e:
                if not error_file is None:
                    failure_case = {
                        "error": str(e),
                        "operator_tree": str(operator_tree.to_dict())
                    }
                    store_jsonl(error_file, [failure_case], "a")
                logger.warning(e)
                results.append(None)
                if self.dev_mode:
                    raise e
                continue
        if run_all:
            return results
        logger.error("All Operator trees failed.")
        return None
    
    def run_operator_tree(self, operator_tree: OperatorTree, reference_date: date) -> OperatorTree:
        node = operator_tree.get_next_unprocessed_node()
        while node:
            # check if in cache
            exec_result = self.cache.get(node)
            if exec_result is None:
                # prepare params
                child_exec_results = node.get_child_execution_results()
                globals()['child_exec_results'] = child_exec_results
                call = node.qu_input
                
                # execute call
                start_time = timing.time()
                exec_result = self.execute_qu_call(call, child_exec_results, reference_date)
                time_taken = (timing.time() - start_time)
                exec_result.set_timing(time_taken)
                self.cache.store(node, exec_result)
            # set result
            node.set_execution_result(exec_result)
            
            # get next node to execute (prepare for next iteration)
            node = operator_tree.get_next_unprocessed_node()
        return operator_tree

    def execute_qu_call(self, qu_call: str, child_exec_results: Dict[str, QUExecutionResult], reference_date: date) -> QUExecutionResult:
        """
        Run the given QU call.
        TODO: Could be risky, as input comes from LLM. Sandbox recommended when running without Docker.
        """
        def _prepare_qu_call(qu_call: str, child_exec_results: Dict[str, QUExecutionResult]) -> str:
            for key, value in self.EXECUTION_DICTS.items():
                qu_call = qu_call.replace(key, value)
            for qu_input in child_exec_results:
                qu_input_escaped = qu_input.replace('"', '\\"')
                qu_call = qu_call.replace("{{ " + qu_input + " }}", f"child_exec_results[\"{qu_input_escaped}\"]")
            return qu_call
        def _incorporate_reference_date(qu_call: str, reference_date: date) -> str:
            # dates
            qu_call = qu_call.replace("date.today()", f"date.fromisoformat(\"{reference_date.isoformat()}\")")
            # datetimes
            reference_datetime = datetime(reference_date.year, reference_date.month, reference_date.day)
            qu_call = qu_call.replace("datetime.now()", f"datetime.fromisoformat(\"{reference_datetime.isoformat()}\")")
            return qu_call

        prepared_qu_call = _prepare_qu_call(qu_call, child_exec_results)
        prepared_qu_call = _incorporate_reference_date(prepared_qu_call, reference_date)
        try:
            logger.debug(f"Running QU call {prepared_qu_call}")
            exec_result = eval(prepared_qu_call)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            raise self.OperatorTreeExecutionError(
                f"Exception {type(e)} ({e}) catched in `OperatorTreeExecution.execute_qu_call` when executing `{prepared_qu_call}` (was `{qu_call}` before preparation)."
            )
        return exec_result
    
    def clear_cache(self):
        self.cache.clear()

    class OperatorTreeExecutionError(Exception):
        pass


class QUExecutionCache:
    """
    Implement LRU cache for Operator Tree execution results.
    """
    def __init__(self, max_cache_size: int):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

    def get(self, node: OperatorTree) -> QUExecutionResult | None:
        cache_key = node.get_cache_key()
        if cache_key in self.cache:
            logger.debug(f"Using cache for {cache_key}")
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        return None

    def store(self, node: OperatorTree, qu_result: QUExecutionResult) -> None:
        cache_key = node.get_cache_key()
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
        elif len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)
        self.cache[cache_key] = qu_result

    def clear(self):
        self.cache = OrderedDict()
