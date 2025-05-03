
import re
import itertools
from typing import List, Callable, Dict, Any, Optional
from copy import deepcopy
from loguru import logger
from collections import defaultdict
from tqdm import tqdm

from reqap.retrieval.retrieval import Retrieval
from reqap.extract.extract_module import ExtractModule
from reqap.library.library import argmax, argmin, avg, sum_skip_nones
from reqap.classes.qu_execution_res import QUExecutionResult, GroupByResult, Group
from reqap.classes.computed_event import ComputedEvent


INFINITY = float('inf')
NEG_INFINITY = float('-inf')


def RETRIEVE(retrieval: Retrieval, extract: ExtractModule, query: str) -> QUExecutionResult:
    observable_events = retrieval.retrieve(query)
    computed_items = extract.create_computed_events(observable_events, interpretation=query)
    exec_res = QUExecutionResult(computed_items)
    return exec_res


def SELECT(extract: ExtractModule, persona_dict: Dict, l: QUExecutionResult, attr_names: List[str], attr_types: List) -> QUExecutionResult:
    computed_items = [deepcopy(ce) for ce in l.computed_items]
    computed_items = extract.run(computed_items, attr_names, attr_types, persona_dict)
    result = [
        [
            i.attributes[attr_name]
            for i in computed_items
        ]
        for attr_name in attr_names
    ]
    exec_res = QUExecutionResult(computed_items, result)
    return exec_res


def GROUP_BY(l: QUExecutionResult, attr_names: List[str]) -> GroupByResult:
    # create groups
    groups_dict: Dict[Any: ComputedEvent] = defaultdict(lambda: [])
    for ce in l.computed_items:
        #extend to tuple
        values = tuple([str(ce.attributes[attr_name]) for attr_name in attr_names])
        groups_dict[values].append(ce)

    groups = list()
    for values, group in groups_dict.items():
        attributes = {attr_name: val for attr_name, val in zip(attr_names, values)}
        group = Group(
            computed_items=group,
            attributes=attributes,
        )
        groups.append(group)
    return GroupByResult(groups)


SMALLER = "<"
SMALLER_EQUAL = "<="
GREATER = ">"
GREATER_EQUAL = ">="
INEQUALITY_OPERATORS = [SMALLER_EQUAL, SMALLER, GREATER_EQUAL, GREATER]
def JOIN(l1: QUExecutionResult, l2: QUExecutionResult, condition: str) -> QUExecutionResult:
    def greater(i1: ComputedEvent, i2: ComputedEvent, l1_attributes: List[str], l2_attributes: List[str]) -> bool:
        if i1 is None:
            return True
        elif i1 is not None and i2 is None:
            return False
        l1_vals = [i1[attr] for attr in l1_attributes]
        l2_vals = [i2[attr] for attr in l2_attributes]
        return l1_vals > l2_vals

    def equal(i1: ComputedEvent, i2: ComputedEvent, l1_attributes: List[str], l2_attributes: List[str]) -> bool:
        if i1 is None and i2 is None:
            return True
        elif i1 is None or i2 is None:
            return False
        l1_vals = [i1[attr] for attr in l1_attributes]
        l2_vals = [i2[attr] for attr in l2_attributes]
        return l1_vals == l2_vals 
    
    def merge(i1: ComputedEvent, i2: ComputedEvent, condition: str) -> ComputedEvent:
        """ Merge the items i1 and i2 and create a new ComputedEvent. """
        i1_obs_events = i1.get_observable_events() if type(i1) is ComputedEvent else i1.observable_events
        i2_obs_events = i2.get_observable_events() if type(i2) is ComputedEvent else i2.observable_events
        obs_events = i1_obs_events + i2_obs_events
        attributes = i1.attributes
        attributes.update(i2.attributes)
        interpretation = f"join(i1={i1.interpretation}, i2={i2.interpretation}, condition={condition})"
        return ComputedEvent(attributes=attributes, interpretation=interpretation, observable_events=obs_events)
    
    # split and separate conditions
    if "or" in condition or "OR" in condition:
        raise NotImplementedError(f"`or` condition in {condition} currently not supported.")
    condition = condition.replace("l1", "i1").replace("l2", "i2")  # easy fix for cases of using l1 instead of i1
    conditions = condition.split("and")
    equality_conditions = [c for c in conditions if "==" in c]
    other_conditions = [c for c in conditions if not "==" in c]
    l1_attributes = [re.findall(r'\bi1\.(\w+)\b', c)[0] for c in equality_conditions]
    l2_attributes = [re.findall(r'\bi2\.(\w+)\b', c)[0] for c in equality_conditions]
    if len(l1_attributes) != len(l2_attributes):
        raise NotImplementedError(f"Currently not supported to not mention same number of attributes: len(l1_attributes)={len(l1_attributes)} != len(l2_attributes)={len(l2_attributes)} (in {condition}).")

    l1_items = _get_items(l1)
    l2_items = _get_items(l2)
    # sort items in case equality is required for >= 1 attribute
    if l1_attributes:
        l1_items = sorted(l1_items, key=lambda i: [i[attr] for attr in l1_attributes])
        l2_items = sorted(l2_items, key=lambda i: [i[attr] for attr in l2_attributes])
    
        # join
        remaining_condition = " and ".join(other_conditions)
        l1_idx = 0
        l2_idx = 0

        items = list()
        while l1_idx < len(l1_items) and l2_idx < len(l2_items):
            i1 = l1_items[l1_idx]
            i2 = l2_items[l2_idx]
            # equal -> merge items (if other conditions hold) and advance
            if equal(i1, i2, l1_attributes, l2_attributes):
                if not remaining_condition or eval(remaining_condition):
                    merged_event = merge(i1, i2, condition)
                    items.append(merged_event)
                next_i1 = l1_items[l1_idx+1] if l1_idx+1 < len(l1_items) else None
                next_i2 = l2_items[l2_idx+1] if l2_idx+1 < len(l2_items) else None
                if greater(next_i1, next_i2, l1_attributes, l2_attributes):
                    l2_idx += 1
                else:
                    l1_idx += 1
            elif greater(i1, i2, l1_attributes, l2_attributes):
                l2_idx += 1
            else:
                l1_idx += 1
    elif any(op in condition for op in INEQUALITY_OPERATORS):
        items = _join_inequalities(
            condition=condition,
            l1_items=l1_items,
            l2_items=l2_items,
            merge=merge
        )
    # otherwise, go for cross-product
    else:
        items = list()
        num_candidates = len(l1_items) * len(l2_items)
        logger.debug(f"Going for cross-product with condition {condition} and {num_candidates} candidates")
        join_candidates = itertools.product(l1_items, l2_items)
        for i1, i2 in tqdm(join_candidates):
            condition_satisfied = eval(condition)
            if not condition_satisfied:
                continue
            merged_event = merge(i1, i2, condition)
            items.append(merged_event)
    
    # construct final result  
    exec_res = QUExecutionResult(computed_items=items)
    return exec_res

def APPLY(l: QUExecutionResult | GroupByResult, fct: Callable, res_name: str="res_name") -> QUExecutionResult:
    items = _get_items(l)
    result = fct(items)
    # check if output of fct is a list of items!
    if isinstance(result, list) and type(result[0]) in [Group, ComputedEvent]:
        items = result
    exec_res = QUExecutionResult(computed_items=items, result=result)
    return exec_res


def MAP(l: QUExecutionResult | GroupByResult, fct: Callable, res_name: Optional[str]="map_result") -> QUExecutionResult:
    items = [deepcopy(i) for i in _get_items(l)]
    mapped_items = list()
    results = list()
    for i in items:
        try:
            if type(i) == Group:
                res = fct(i.computed_items)
            else:
                res = fct(i)
            i.attributes[res_name] = res
            results.append(res)
            mapped_items.append(i)
        except Exception as e:
            logger.error(f"Exception catched when applying MAP {fct} to item=`{i}` with attributes={i.attributes}: {e}.")
    exec_res = QUExecutionResult(computed_items=mapped_items, result=results)
    return exec_res


def FILTER(l: QUExecutionResult, filter: Callable) -> QUExecutionResult:
    error_thrown = False
    items = list()
    for item in _get_items(l):
        try:
            if filter(item.attributes):
                items.append(item)
        except Exception as e:  # catch cases in which accessed attribute is empty
            if not error_thrown:
                logger.error(f"Exception catched when applying FILTER {filter} to item=`{item}` with attributes={item.attributes}: {e}.")
                error_thrown = True
            continue
    exec_res = QUExecutionResult(computed_items=items)
    return exec_res


def UNNEST(l: QUExecutionResult, nested_attr_name: str, unnested_attr_name: str) -> QUExecutionResult:
    error_thrown = False
    unnested_items = list()
    for item in _get_items(l):
        if not type(item.attributes[nested_attr_name]) == list:
            if not error_thrown:
                logger.error(f"Called UNNEST with nested_attr_name={nested_attr_name}, but attribute is not a list: type(item.attributes[nested_attr_name])=`{type(item.attributes[nested_attr_name])}`, item.attributes[nested_attr_name]=`{item.attributes[nested_attr_name]}`.")
                error_thrown = True
        for attr in item.attributes[nested_attr_name]:
            new_item = deepcopy(item)
            new_item.attributes[unnested_attr_name] = attr
            unnested_items.append(new_item)
    exec_res = QUExecutionResult(computed_items=unnested_items)
    return exec_res


def SUM(l: QUExecutionResult, attr_name: str) -> QUExecutionResult:
    attributes = list(_get_attributes(
        l=l, attr_name=attr_name
    ))
    result = sum_skip_nones(attributes)
    items = _get_items(l)
    sum_items = [item for item, attr in zip(items, attributes) if attr is not None]
    exec_res = QUExecutionResult(computed_items=sum_items, result=result)
    return exec_res


def AVG(l: QUExecutionResult, attr_name: str) -> QUExecutionResult:
    attributes = list(_get_attributes(
        l=l, attr_name=attr_name
    ))
    result = avg(attributes, skip_none=True)
    items = _get_items(l)
    avg_items = [item for item, attr in zip(items, attributes) if attr is not None]
    exec_res = QUExecutionResult(computed_items=avg_items, result=result)
    return exec_res


def MAX(l: QUExecutionResult, attr_name: str) -> QUExecutionResult:
    attributes = list(_get_attributes(
        l=l, attr_name=attr_name
    ))
    indices = argmax(attributes)
    items = _get_items(l)
    argmax_items = [items[idx] for idx in indices]
    result = attributes[indices[0]] if indices else 0
    exec_res = QUExecutionResult(computed_items=argmax_items, result=result)
    return exec_res


def MIN(l: QUExecutionResult, attr_name: str) -> QUExecutionResult:
    attributes = list(_get_attributes(
        l=l, attr_name=attr_name
    ))
    indices = argmin(attributes)
    items = _get_items(l)
    argmin_items = [items[idx] for idx in indices]
    result = attributes[indices[0]] if indices else 0
    exec_res = QUExecutionResult(computed_items=argmin_items, result=result)
    return exec_res


def ARGMAX(l: QUExecutionResult | GroupByResult, arg_attr_name: str, val_attr_name: Optional[str] = "") -> QUExecutionResult:
    arg_attributes = list(_get_attributes(
        l=l, attr_name=arg_attr_name
    ))
    indices = argmax(arg_attributes)
    items = _get_items(l)
    argmax_items = [items[idx] for idx in indices]
    if val_attr_name:
        val_attributes = list(_get_attributes(
            l=l, attr_name=val_attr_name
        ))
        result = [val_attributes[idx] for idx in indices]
    else:
        result = [i.attributes for i in argmax_items]
    exec_res = QUExecutionResult(computed_items=argmax_items, result=result)
    return exec_res


def ARGMIN(l: QUExecutionResult, arg_attr_name: str, val_attr_name: Optional[str] = "") -> QUExecutionResult:
    arg_attributes = list(_get_attributes(
        l=l, attr_name=arg_attr_name
    ))
    indices = argmin(arg_attributes)
    items = _get_items(l)
    argmin_items = [items[idx] for idx in indices]
    if val_attr_name:
        val_attributes = list(_get_attributes(
            l=l, attr_name=val_attr_name
        ))
        result = [val_attributes[idx] for idx in indices]
    else:
        result = [i.attributes for i in argmin_items]
    exec_res = QUExecutionResult(computed_items=argmin_items, result=result)
    return exec_res


def _get_attributes(l: QUExecutionResult | GroupByResult, attr_name: str, attr_type: type=None) -> List:
    """
    Get the attributes for the list.
    NoneType values are omitted.
    """
    attributes = list()
    items = _get_items(l)
    for item in items:
        attr = item.attributes[attr_name]
        if attr_type:
            attr = attr_type(attr)
        attributes.append(attr)
    return attributes


def _get_items(l: QUExecutionResult | GroupByResult) -> List[ComputedEvent] | List[Group]:
    items = l.computed_items if type(l) == QUExecutionResult else l.groups
    return items


def _join_inequalities(condition: str, l1_items: List[ComputedEvent], l2_items: List[ComputedEvent], merge: Callable):
    MAX_NUM_PAIRS = 10000000
    conditions = condition.split("and")
    inequality_conditions = [c for c in conditions if any(op in c for op in INEQUALITY_OPERATORS)]
    other_conditions = [c for c in conditions if c not in inequality_conditions]
    other_conditions_joined = " and ".join(other_conditions)

    # derive set of candidate pairs for all conditions
    condition_to_candidate_pairs = dict()
    for c in inequality_conditions:
        # sort both lists by relevant attribute
        l1_attr = re.findall(r'\bi1\.(\w+)\b', c)[0]
        l2_attr = re.findall(r'\bi2\.(\w+)\b', c)[0]
        l1_items = sorted(l1_items, key=lambda i: i[l1_attr])
        l2_items = sorted(l2_items, key=lambda i: i[l2_attr])

        # identify condition operator
        c = c.strip()
        operator = [op for op in INEQUALITY_OPERATORS if op in c]
        assert len(operator), f"Faiure: operator expected to occur in c=`{c}`, but not present"
        operator = operator[0]  # order in OPERATORS is important! e.g., <= should come before <

        # understand which part is supposed to be larger
        c1, _ = c.split(operator, 1)

        # i1.attr1 <= i2.attr2 or i2.attr2 >= i1.attr1 
        l2_larger= ("i1" in c1 and operator in [SMALLER_EQUAL, SMALLER]) or ("i2" in c1 and operator == [GREATER_EQUAL, GREATER])
        
        # derive a set of candidate pairs for next iteration; pairs are the indices in the lists
        candidate_pairs = set()
        pbar = tqdm(total=len(l1_items) * len(l2_items))
        idx_l1 = 0
        idx_l2 = 0
        while idx_l1 < len(l1_items) and idx_l2 < len(l2_items):  # stopping criterion
            # access items at index
            i1 = l1_items[idx_l1]
            i2 = l2_items[idx_l2]

            # check condition
            if eval(c):
                # 1. if condition holds, then for all of the larger items in the list the condition holds as well
                # 2. advance pointer of list which is supposed to be smaller
                if l2_larger:
                    new_candidate_pairs = [(idx_l1, idx) for idx in range(len(l2_items))[idx_l2:]]
                    idx_l1 += 1
                    pbar.update(len(l2_items)-idx_l2)
                else:
                    new_candidate_pairs = [(idx, idx_l2) for idx in range(len(l1_items))[idx_l1:]]
                    idx_l2 += 1
                    pbar.update(len(l1_items)-idx_l1)
                candidate_pairs.update(new_candidate_pairs)
                # Avoid too large computations
                if len(candidate_pairs) >= MAX_NUM_PAIRS:
                    break
                
            else:
                # otherwise, advance relevant pointer, and try again
                if l2_larger:
                    idx_l2 += 1
                    pbar.update(len(l1_items)-idx_l1)
                else:
                    idx_l1 += 1
                    pbar.update(len(l2_items)-idx_l2)
            
        # store candidates for condition
        condition_to_candidate_pairs[c] = candidate_pairs
    logger.debug("Done with identifying candidates")

    # derive intersection of all candidate pairs
    if not len(condition_to_candidate_pairs):
        logger.warning(f"Derived empty dictionary condition_to_candidate_pairs={condition_to_candidate_pairs} for conditions={conditions}")
    result_iterator = iter(condition_to_candidate_pairs.values())
    candidate_pairs = next(result_iterator)  # init with first result
    candidate_set = candidate_pairs
    while candidate_set:
        candidate_pairs = candidate_pairs & candidate_set  # intersect with ongoing result
        try:
            candidate_set = next(result_iterator)
        except StopIteration:
            break
    logger.debug("Done with intersection")

    # derive result
    items = list()
    final_pairs = set()
    for (idx_l1, idx_l2) in candidate_pairs:
        i1 = l1_items[idx_l1]
        i2 = l2_items[idx_l2]
        # check if all conditions hold (others than equality too)
        if other_conditions_joined.strip():
            condition_satisfied = eval(other_conditions_joined)
            if not condition_satisfied:
                continue
        final_pairs.add((idx_l1, idx_l2))
        merged_event = merge(i1, i2, condition)
        items.append(merged_event)
    return items
