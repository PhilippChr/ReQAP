import re
import ast
from omegaconf import DictConfig
from typing import Dict, List
from loguru import logger
import numpy as np

from reqap.library.library import batchify
from reqap.qu.qu_dataset import represent_input_seq2seq
from reqap.llm.seq2seq_model import Seq2SeqModel, compute_metrics
from reqap.llm.causal_model import CausalModel


FUNCTION_PATTERNS = [
    re.compile(r"^RETRIEVE\s*\(\s*query\s*=\s*.*\s*\)$"),
    re.compile(r"^SELECT\s*\(\s*l\s*=\s*.*\s*,\s*attr_names\s*=\s*.*\s*,\s*attr_types\s*=\s*.*\s*\)$"),
    re.compile(r"^GROUP_BY\s*\(\s*l\s*=\s*.*\s*,\s*attr_names\s*=\s*.*\s*\)$"),
    re.compile(r"^JOIN\s*\(\s*l1\s*=\s*.*\s*,\s*l2\s*=\s*.*\s*,\s*condition\s*=\s*.*\s*\)$"),
    re.compile(r"^APPLY\s*\(\s*l\s*=\s*.*\s*,\s*fct\s*=\s*[^,]*\s*\)$"),
    re.compile(r"^APPLY\s*\(\s*l\s*=\s*.*\s*,\s*fct\s*=\s*.*\s*,\s*res_name\s*=\s*.*\s*\)$"),
    re.compile(r"^MAP\s*\(\s*l\s*=\s*.*\s*,\s*fct\s*=\s*[^,]*\s*\)$"),
    re.compile(r"^MAP\s*\(\s*l\s*=\s*.*\s*,\s*fct\s*=\s*.*\s*,\s*res_name\s*=\s*.*\s*\)$"),
    re.compile(r"^FILTER\s*\(\s*l\s*=\s*.*\s*,\s*filter\s*=\s*.*\s*\)$"),
    re.compile(r"^UNNEST\s*\(\s*l\s*=\s*.*\s*,\s*nested_attr_name\s*=\s*.*\s*,\s*unnested_attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^SUM\s*\(\s*l\s*=\s*.*\s*,\s*attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^AVG\s*\(\s*l\s*=\s*.*\s*,\s*attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^MAX\s*\(\s*l\s*=\s*.*\s*,\s*attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^MIN\s*\(\s*l\s*=\s*.*\s*,\s*attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^ARGMAX\s*\(\s*l\s*=\s*.*\s*,\s*arg_attr_name\s*=\s*.*\s*,\s*val_attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^ARGMAX\s*\(\s*l\s*=\s*.*\s*,\s*arg_attr_name\s*=\s*[^,]*\s*\)$"),
    re.compile(r"^ARGMIN\s*\(\s*l\s*=\s*.*\s*,\s*arg_attr_name\s*=\s*.*\s*,\s*val_attr_name\s*=\s*.*\s*\)$"),
    re.compile(r"^ARGMIN\s*\(\s*l\s*=\s*.*\s*,\s*arg_attr_name\s*=\s*[^,]*\s*\)$")
]


def compute_metrics_qu(eval_preds, tokenizer):
    """ Add exact match metric. """
    # run basic generation metrics
    metrics = compute_metrics(eval_preds, tokenizer)

    # derive predictions
    preds, _ = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # measure errors
    metrics["syntax_errors"] = sum(1 for p in decoded_preds if has_syntax_error(p)) / max(1, len(decoded_preds))
    metrics["valid_calls"] = sum(1 for p in decoded_preds if is_valid(p)) / max(1, len(decoded_preds))
    return metrics


def clean_calls(qu_calls: List[str]):
    """ Derive clean calls from the list."""
    clean_calls = list()
    for call in qu_calls:
        if not has_syntax_error(call) and is_valid(call):
            clean_calls.append(call)
    
    # ensure that there is a call as output to avoid errors
    if not clean_calls:
        logger.warning(f"No clean call found. Candidates: {qu_calls}")
        return qu_calls[0]
    return clean_calls[0]


def has_syntax_error(qu_call: str):
    call = re.sub("{{ QU(.*?) }}", "variable", qu_call)
    try:
        ast.parse(call)
    except SyntaxError:
        return True
    return False


def is_valid(function_call: str) -> bool:
    if has_syntax_error(function_call):
        return False
    return any(p.match(function_call.strip()) for p in FUNCTION_PATTERNS)


def adapt_params(generation_params: Dict) -> Dict:
    """ Adapt params for QU model for seq2seq. """
    # adapt params for sampling
    if "n" in generation_params:
        n = generation_params["n"]
        sampling_factor = generation_params["sampling_factor"]
        if generation_params["sampling_factor"] > 1:
            generation_params["do_sample"] = True
            generation_params["num_return_sequences"] = n * sampling_factor
            generation_params["num_beams"] = n * sampling_factor
            generation_params["early_stopping"] = True
        else:
            generation_params["early_stopping"] = False
            generation_params["do_sample"] = False
    
    # drop max_tokens
    if "max_tokens" in generation_params:
        del(generation_params["max_tokens"])
    
    # drop params that are not used for Seq2seq inference
    del(generation_params["n"])
    del(generation_params["sampling_factor"])
    return generation_params


class QUModel(Seq2SeqModel):
    def __init__(self, qu_config: DictConfig, train: bool=True, use_cache: bool=False):
        super(QUModel, self).__init__(qu_config, train, use_cache)

    def inference(self, input_text: List[str], sampling_params: Dict) -> str | List[str]:
        return self.batch_inference(input_texts=[input_text], sampling_params=sampling_params)[0]
    
    def batch_inference(self, input_texts: List[str], sampling_params: Dict) -> List[str] | List[List[str]]:
        # adjust params
        generation_params = adapt_params(sampling_params.copy())
        logger.debug(f"QUModel inference with len(input_texts)={len(input_texts)}, original_sampling_params={sampling_params}, used_sampling_params={generation_params}")
        
        # always sample multiple and then prune
        batch_size = self.seq2seq_config.qu_inference_batch_size
        result = list()
        for batch in batchify(input_texts, batch_size):
            result += super(QUModel, self).batch_inference(batch, generation_params)
        
        # if sampling, clean output output
        result = [clean_calls(calls) for calls in result]
        return result


class QUModelCausal(CausalModel):
    def __init__(self, qu_config: DictConfig, train: bool=True, use_cache: bool=False):
        super(QUModelCausal, self).__init__(qu_config, train, use_cache)

    def inference(self, input_text: List[str], sampling_params: Dict) -> str | List[str]:
        return self.batch_inference(input_texts=[input_text], sampling_params=sampling_params)[0]
    
    def batch_inference(self, input_texts: List[str], sampling_params: Dict) -> List[str] | List[List[str]]:
        # adjust params
        generation_params = self.causal_config.sampling_params.copy()
        generation_params["num_return_sequences"] = generation_params["n"]
        del(generation_params["n"])
        logger.debug(f"QUModel inference with len(input_texts)={len(input_texts)}, original_sampling_params={sampling_params}, used_sampling_params={generation_params}")
        
        # always sample multiple and then prune
        batch_size = self.causal_config.qu_inference_batch_size
        result = list()
        for batch in batchify(input_texts, batch_size):
            result += super(QUModelCausal, self).batch_inference(batch, generation_params)
        return result
