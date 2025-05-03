import datetime
from typing import Any, List, Set
from loguru import logger

from reqap.library.temporal import datetime_to_timestamp, date_to_timestamp, temporal_equivalent, is_temporal_expression, numbers_equivalent
from reqap.library.library import normalize_str, flatten


"""QA"""
def hit_at_1(derived_answers: Any, gold_answers: Any, relax_factor: float=0.0) -> float:
    """
    Function that computes whether there is a hit or not.
    Compares the sets in an element-wise manner.
    """
    def _hit_at_1(derived_answers: Any, gold_answer: Any, relax_factor: float=0.0) -> bool:
        hit_fct = _relaxed_hit if relax_factor > 0.0 else _hit
        if derived_answers == gold_answers:
            return True

        if type(derived_answers) == list:
            return any(
                _hit_at_1(d_ans, gold_answer, relax_factor)
                if type(d_ans) == list
                else hit_fct(d_ans, gold_answer, relax_factor)
                for d_ans in derived_answers
            )
        else:
            return hit_fct(derived_answers, gold_answer, relax_factor)

    def _hit(derived_answer: Any, gold_answer: Any, relax_factor: float=0.0) -> bool:
        # process based on type
        if type(derived_answer) == datetime.datetime:
            return datetime_to_timestamp(derived_answer) == gold_answer
        elif type(derived_answer) == datetime.date:
            return date_to_timestamp(derived_answer) == gold_answer
        elif type(derived_answer) == datetime.time:
            return derived_answer.isoformat() == gold_answer
        elif type(derived_answer) in (float, int):
            return numbers_equivalent(derived_answer, gold_answer)
        elif type(derived_answer) == str:
            # check for matching temporal expressions
            if temporal_equivalent(derived_answer, gold_answer):
                return True
            
            # normalize strings
            if type(gold_answer) == str:
                return normalize_str(derived_answer) == normalize_str(gold_answer)
            
            # ensure that answer is not a nested string (e.g., str-tuple like '(10, 2020)')
            try:
                derived_answer = eval(derived_answer)
                if _hit(derived_answer, gold_answer, relax_factor):
                    return True
            except:
                pass
            
            # try to convert result into same type as answer
            t = type(gold_answer)
            try:
                derived_answer = t(derived_answer)
                return _hit(derived_answer, gold_answer, relax_factor)
            except:
                return False
        elif type(derived_answer) == dict:
            gold_answer = normalize_str(gold_answer) if isinstance(gold_answer, str) else gold_answer
            return gold_answer in [normalize_str(a) if isinstance(a, str) else a for a in derived_answer.values()]
        elif type(derived_answer) in [tuple, list]:
            if type(gold_answer) in [int, float]:
                # logger.warning(f"Failure with gold_answer={gold_answer}, derived_answer={derived_answer}")
                return False
            return sorted(tuple(derived_answer)) == sorted(tuple(gold_answer))
        # logger.warning(f"Type {type(derived_answer)} not covered: derived_answer={derived_answer}, gold_answer={gold_answer} ...")
        return derived_answer == gold_answer
    
    def _relaxed_hit(derived_answer: Any, gold_answer: Any, relax_factor: float) -> bool:
        if type(derived_answer) in (float, int):
            return numbers_equivalent(derived_answer, gold_answer, relax_factor=relax_factor)
        return _hit(derived_answer, gold_answer, relax_factor)

    # gold answer none, derived answer failed
    if gold_answers == None and derived_answers in [0, "0", 0.0, [0]]:
        return 1.0
    # both are empty => correct
    if _is_empty(gold_answers) and _is_empty(derived_answers):
        return 1.0
    # one is empty => incorrect
    elif _is_empty(gold_answers) or _is_empty(derived_answers):
        return 0.0
    
    # flatten
    derived_answers = flatten(derived_answers)
    gold_answers = flatten(gold_answers)

    # eval (useful for datetime objects)
    if type(derived_answers) == str and not is_temporal_expression(derived_answers):
        try:
            derived_answers = eval(derived_answers)
        except Exception as e:
            # logger.debug(f"Could not run eval({derived_answers}): {e}")
            pass

    if any(_hit_at_1(derived_answers, ans, relax_factor) for ans in gold_answers):
        return 1.0
    return 0.0


def _is_empty(answer):
    if answer is None:
        return True
    elif isinstance(answer, list):
        if not answer:
            return True
        else:
            return _is_empty(answer[0])  # resolve nested empty results
    elif isinstance(answer, str):
        try:
            answer = eval(answer)
            return _is_empty(answer)
        except:
            return False
    else:
        return False


"""Retrieval"""
def recall(gold_obs_event_ids: Set[int] | List[int], pred_obs_event_ids: Set[int] | List[int]) -> float:
    gold_obs_event_ids = set(gold_obs_event_ids)
    pred_obs_event_ids = set(pred_obs_event_ids)
    if not len(gold_obs_event_ids):
        return 1.0
    rec = len(gold_obs_event_ids & pred_obs_event_ids) / len(gold_obs_event_ids)
    return round(float(rec), 3)


def missed_event_ids(gold_obs_event_ids: Set[int] | List[int], pred_obs_event_ids: Set[int] | List[int]) -> Set[int]:
    gold_obs_event_ids = set(gold_obs_event_ids)
    pred_obs_event_ids = set(pred_obs_event_ids)
    return gold_obs_event_ids - pred_obs_event_ids
