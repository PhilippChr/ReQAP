import os
import math
from omegaconf import DictConfig
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
from datetime import date

from reqap.library.metrics import hit_at_1
from reqap.library.library import get_persona_names, load_persona_dict, load_json, load_jsonl, store_jsonl, clear_file, handle_output_file, batchify, avg
from reqap.qu.qu_module import QuestionUnderstandingModule
from reqap.qu.operator_tree_execution import QUPlanExecution
from reqap.qu.operator_tree import OperatorTree
from reqap.retrieval.retrieval import Retrieval
from reqap.extract.extract_module import ExtractModule


class ReQAP:
    QU_PROCESSOR_BATCH_SIZE = 10
    MAX_NUM_OPERATOR_TREES = 1000

    def __init__(self, config: DictConfig):
        self.config = config
        self.qu_loaded = False

    def run_qud_on_split(self, split: str="test", override: bool=False):
        """
        Run the QUD stage of ReQAP on a full split.
        Derives a list of distinct questions in the split first,
        and then derives a single list of Operator Trees for each unique 
        question.
        """
        # init
        self.load_qu()
        benchmark_dir = self.config.benchmark.benchmark_dir
        result_dir = self.config.benchmark.result_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)

        # derive all unique questions
        questions = set()
        for persona in personas:
            questions_path = f"{persona_dir}/{persona}/questions.json"
            data = load_json(questions_path)
            questions.update(it["question"] for it in data)
        questions = list(questions)
        logger.debug(f"Identified {len(questions)} different questions.")

        # load existing result or start from scratch if override=True
        qu_result_path = self.config.qu.qu_result_paths[split]
        data = handle_output_file(qu_result_path, data, override)
        question_to_operator_tree = QuestionUnderstandingModule.initialize_question_to_operator_tree(qu_result_path, override)

        # drop questions which are covered already:
        for question in questions.copy():
            if question in question_to_operator_tree:
                questions.remove(question)

        # derive Operator Trees
        logger.debug(f"Running inference for {len(questions)} questions.")
        batch_size = self.QU_PROCESSOR_BATCH_SIZE
        num_batches = math.ceil(len(questions)/batch_size)
        for batch in tqdm(batchify(questions, batch_size=batch_size), total=num_batches):
            # run QU in batch-wise manner
            operator_trees_list = self.qu.run_batch(
                questions=batch,
                sampling_params=self.config.qu.sampling_params
            )
            logger.debug(f"Done with next {len(operator_trees_list)} questions")

            # store result in dict
            batch_question_to_operator_tree = dict()
            for question, operator_trees in zip(batch, operator_trees_list):
                operator_trees_list = list()
                if len(operator_trees) > self.MAX_NUM_OPERATOR_TREES:
                    logger.warning(f"Derived too many Operator Trees ({len(operator_trees)})...")
                for operator_tree in operator_trees[:self.MAX_NUM_OPERATOR_TREES]:
                    if type(operator_tree) == dict:  # already converted into dict
                        operator_trees_list.append(operator_tree)
                    else:
                        operator_tree_dict = operator_tree.to_dict()
                        operator_trees_list.append(operator_tree_dict)
                batch_question_to_operator_tree[question] = operator_trees_list
            
            # store Operator Trees
            for question, value in batch_question_to_operator_tree.items():
                store_jsonl(qu_result_path, [{question: value}], file_mode="a")
            question_to_operator_tree.update(batch_question_to_operator_tree)
        logger.debug(f"Done with creating Operator Trees.")

        # store data
        logger.debug(f"Storing data...")
        for persona in tqdm(personas):
            questions_path = f"{persona_dir}/{persona}/questions.json"
            output_path = f"{result_dir}/{persona}/qu_result.jsonl"
            data = load_json(questions_path)
            clear_file(output_path)
            for instance in data:
                question = instance["question"]
                operator_trees_list = question_to_operator_tree[question]
                instance["operator_trees"] = operator_trees_list
                question_to_operator_tree[question] = operator_trees_list
                store_jsonl(output_path, [instance], file_mode="a")
        logger.debug(f"Done with storing QU result.")
    
    def run_otx_on_split(self, split: str="test", override: bool=False) -> None:
        """
        Run the Operator Trees derived via the QU stage on a full split.
        Accesses the result of the function `run_qu_module_on_split`.
        """
        logger.info(f"Starting to run Operator Trees with config={self.config}...")
        # init
        benchmark_dir = self.config.benchmark.benchmark_dir
        result_dir = self.config.benchmark.result_dir
        splade_indices_dir = self.config.splade.splade_indices_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)

        # iterate through personas
        for i, persona in enumerate(personas):
            # init paths for persona
            hit_at_1_list = list()
            relaxed_hit_at_1_10_list = list()
            relaxed_hit_at_1_20_list = list()
            num_failures = 0
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            splade_index_path = f"{splade_indices_dir}/{persona}.splade_index"
            persona_path = f"{persona_dir}/{persona}/{persona}.json"
            input_path = f"{result_dir}/{persona}/qu_result.jsonl"
            output_path = f"{result_dir}/{persona}/result.jsonl"
            data = load_jsonl(input_path)

            # construct engine
            qu_engine = self.load_engine(obs_events_csv_path, splade_index_path, persona_path)
            
            # prepare output file
            data = handle_output_file(output_path, data, override)

            # run all Operator Trees for persona
            for j, instance in tqdm(enumerate(data), total=len(data)):
                logger.debug(f"Starting with inference for question {j} / {len(data)}, persona {i} / {len(personas)}")
                reference_date = date.fromisoformat(instance["reference_date"])
                operator_tree_dicts = instance["operator_trees"]
                operator_trees = OperatorTree.from_operator_tree_dicts(operator_tree_dicts)
                result_dict, derived_answer, failed = qu_engine.derive_result(
                    operator_trees=operator_trees,
                    reference_date=reference_date,
                    error_file=output_path.replace(".jsonl", ".errors.jsonl")
                )
                instance["failed"] = failed   
                if failed:
                    num_failures += 1
                instance["result"] = result_dict
                instance["derived_answer"] = derived_answer
                try:
                    hit_at_1_score = hit_at_1(derived_answer, instance["answers"]) if not failed else 0.0
                    instance["hit_at_1"] = hit_at_1_score
                    hit_at_1_list.append(hit_at_1_score)
                    relaxed_hit_at_1_10_score = hit_at_1(derived_answer, instance["answers"], relax_factor=0.1) if not failed else 0.0
                    instance["relaxed_hit_at_1_10"] = relaxed_hit_at_1_10_score
                    relaxed_hit_at_1_10_list.append(relaxed_hit_at_1_10_score)
                    relaxed_hit_at_1_20_score = hit_at_1(derived_answer, instance["answers"], relax_factor=0.2) if not failed else 0.0
                    instance["relaxed_hit_at_1_20"] = relaxed_hit_at_1_20_score
                    relaxed_hit_at_1_20_list.append(relaxed_hit_at_1_20_score)
                except Exception as e:
                    logger.error(f"Catched exception: {e} when evaluating.")
                store_jsonl(output_path, [instance], file_mode="a")

                # log result
                logger.info(f"Persona {i}: avg(hit_at_1_list)={avg(hit_at_1_list)} ({sum(hit_at_1_list)}/{len(hit_at_1_list)})")
                logger.info(f"Persona {i}: avg(relaxed_hit_at_1_10_list)={avg(relaxed_hit_at_1_10_list)} ({sum(relaxed_hit_at_1_10_list)}/{len(relaxed_hit_at_1_10_list)})")
                logger.info(f"Persona {i}: avg(relaxed_hit_at_1_20_list)={avg(relaxed_hit_at_1_20_list)} ({sum(relaxed_hit_at_1_20_list)}/{len(relaxed_hit_at_1_20_list)})")
                logger.info(f"Persona {i}: num failures={num_failures} ({num_failures}/{len(hit_at_1_list)})")

    def run_on_question(self, question: str, qu_engine: QUPlanExecution, reference_date: date=date.today()):
        """
        Run the full ReQAP pipeline on a single question.
        """
        operator_trees = self.qu.run(
            question=question,
            sampling_params=self.config.qu.sampling_params
        )
        logger.debug(f"Operator Trees: {[operator_tree.to_dict() for operator_tree in operator_trees]}")
        result_dict, derived_answer, failed = qu_engine.derive_result(operator_trees, run_all=False, reference_date=reference_date)
        if failed:
            logger.error(f"Failure for question={question}")
        return result_dict, derived_answer, failed
    
    def example(self, question: str):
        # init paths
        benchmark_dir = self.config.benchmark.benchmark_dir
        persona_dir = os.path.join(benchmark_dir, "dev")
        personas = get_persona_names(persona_dir)
        persona = personas[0]
        obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
        splade_indices_dir = self.config.splade.splade_indices_dir
        splade_index_path = f"{splade_indices_dir}/{persona}.splade_index"
        persona_path = f"{persona_dir}/{persona}/{persona}.json"
        
        # load modules
        self.load_qu()
        qu_engine = self.load_engine(obs_events_csv_path, splade_index_path, persona_path)

        """"""
        import torch

        # Memory allocated by tensors
        allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
        # Total reserved by the caching allocator
        reserved = torch.cuda.memory_reserved() / 1024**2  # in MB

        print(f"Allocated memory: {allocated:.2f} MB")
        print(f"Reserved memory: {reserved:.2f} MB")

        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        print(f"RSS (Resident Set Size): {mem_info.rss / 1024 ** 2:.2f} MB")  # Physical RAM
        print(f"VMS (Virtual Memory Size): {mem_info.vms / 1024 ** 2:.2f} MB")  # Virtual Memory
        """"""
        
        # inference
        self.run_on_question(question=question, qu_engine=qu_engine)


    def training_loop(self, split: str="test", override: bool=False, loop_persona=None) -> None:
        # init
        benchmark_dir = self.config.benchmark.benchmark_dir
        splade_indices_dir = self.config.splade.splade_indices_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)

        # iterate through personas
        question_to_count = defaultdict(lambda: 0)
        for i, persona in enumerate(personas):
            # enables per-persona inference
            if not loop_persona is None and persona != loop_persona:
                continue

            # construct engine
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            splade_index_path = f"{splade_indices_dir}/{persona}.splade_index"
            persona_path = f"{persona_dir}/{persona}/{persona}.json"
            qu_engine = self.load_engine(obs_events_csv_path, splade_index_path, persona_path)

            # load QU result
            qu_result_path = self.config.qu.qu_result_paths[split]
            question_to_operator_tree = QuestionUnderstandingModule.initialize_question_to_operator_tree(qu_result_path, override=False)
            
            # prepare input + output file
            questions_path = f"{persona_dir}/{persona}/questions.json"
            output_path = self.config.qu.qu_training.qu_result_data[split]
            output_path = output_path.replace(".jsonl", f"_{persona}.jsonl")
            data = load_json(questions_path)
            data = handle_output_file(output_path, data, override)

            # run all Operator Trees for persona
            for i, instance in tqdm(enumerate(data), total=len(data)):
                logger.debug(f"Starting with inference for question {i} / {len(data)}")
                reference_date = date.fromisoformat(instance["reference_date"])
                question = instance["question"]
                operator_tree_dicts = question_to_operator_tree[question]
                operator_trees = OperatorTree.from_operator_tree_dicts(operator_tree_dicts)
                operator_tree_results = list()
                for j, operator_tree in enumerate(operator_trees):
                    if question_to_count[question] > 3:  # not more than three plans for single instance
                        logger.info(f"Skipping question={question}")
                        break
                    _, derived_answer, failed = qu_engine.derive_result(
                        operator_trees=[operator_tree],
                        reference_date=reference_date,
                        error_file=output_path.replace(".jsonl", ".errors.jsonl")
                    )
                    hit_at_1_score = hit_at_1(derived_answer, instance["answers"])
                    operator_tree_result = {
                        "failed": failed,
                        "hit_at_1": hit_at_1(derived_answer, instance["answers"]),
                        "relaxed_hit_at_1_10": hit_at_1(derived_answer, instance["answers"], relax_factor=0.1),
                        "relaxed_hit_at_1_20": hit_at_1(derived_answer, instance["answers"], relax_factor=0.2),
                        "derived_answer": derived_answer,
                        "operator_tree": operator_tree_dicts[j]
                    }
                    if hit_at_1_score:
                        question_to_count[question] += 1

                    operator_tree_results.append(operator_tree_result)
                instance["results"] = operator_tree_results
                store_jsonl(output_path, [instance], file_mode="a")

    def run_icl_examples(self) -> None:
        """
        Run the ICL Operator Trees for the first persona.
        """
        # init
        benchmark_dir = self.config.benchmark.benchmark_dir
        result_dir = "./data/dev"
        splade_indices_dir = self.config.splade.splade_indices_dir
        persona_dir = os.path.join(benchmark_dir, "train")
        persona = get_persona_names(persona_dir)[0]

        # construct engine
        obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
        splade_index_path = f"{splade_indices_dir}/{persona}.splade_index"
        persona_path = f"{persona_dir}/{persona}/{persona}.json"
        qu_engine = self.load_engine(obs_events_csv_path, splade_index_path, persona_path)
        
        ### Option 1: load ICL examples
        icl_examples_path = self.config.qu.qu_supervisor_icl
        icl_examples = load_json(icl_examples_path)
        operator_trees = [OperatorTree.from_list(operator_tree_list) for operator_tree_list in icl_examples]
        
        ### Option 2: load output of Operator Tree creation
        # icl_examples_path = "./data/dev/operator_trees_debug.jsonl"
        # icl_examples = load_jsonl(icl_examples_path)
        # operator_trees = [operator_tree for instance in icl_examples for operator_tree in QUPlan.from_operator_tree_dicts(instance["operator_trees"])]

        output_path = f"{result_dir}/icl_result.jsonl"
        clear_file(output_path)

        # run all Operator Trees for persona
        for j, (operator_tree, icl_example) in enumerate(zip(operator_trees, icl_examples)):
            logger.debug(f"Starting with inference for Operator Tree {j} / {len(operator_trees)}")
            reference_date = date.fromisoformat("2024-11-25")  # reference date for PerQA
            result_dict, derived_answer, failed = qu_engine.derive_result(
                operator_trees=[operator_tree],
                reference_date=reference_date,
                error_file=output_path.replace(".jsonl", ".errors.jsonl")
            )
            instance = {
                "question": operator_tree.qu_input,
                "result": result_dict,
                "derived_answer": derived_answer,
                "failed": failed
            }
            store_jsonl(output_path, [instance], file_mode="a")

    def load_qu(self) -> None:
        """
        Initiate the QU module.
        """
        if self.qu_loaded:
            return
        qu_mode = self.config.qu.qu_mode
        if qu_mode == "openai":
            from reqap.llm.openai import OpenAIModel
            icl_model = OpenAIModel(openai_config=self.config.openai, use_cache=self.config.openai.use_cache)
        elif qu_mode == "instruct_model":
            from reqap.llm.instruct_model import InstructModel
            icl_model = InstructModel(instruct_config=self.config.instruct_model, use_cache=False)
        elif qu_mode == "seq2seq":
            icl_model = None  # use seq2seq instead
        else:
            logger.warning("Not loading any QU Module/Model!")
            return
        self.qu = QuestionUnderstandingModule(self.config.qu, train=False, icl_model=icl_model)
        self.qu_loaded = True
    
    def load_engine(self, obs_events_csv_path: str, splade_index_path: str, persona_dict_path: str = None) -> QUPlanExecution:
        retrieval = Retrieval(self.config, obs_events_csv_path, splade_index_path)
        retrieval.load()
        extract = ExtractModule(self.config.extract)
        extract.load()
        persona_dict = load_persona_dict(persona_dict_path) if self.config.extract.extract_add_persona and persona_dict_path else None
        qu_engine = QUPlanExecution(qu_config=self.config.qu, retrieval=retrieval, extract=extract, persona_dict=persona_dict)
        return qu_engine
