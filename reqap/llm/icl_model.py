from typing import List, Dict
from omegaconf import DictConfig


class ICLModel:
    def __init__(self, icl_config: DictConfig, use_cache: bool=False):
        self.icl_config = icl_config
        self.use_cache = use_cache
        self.cache = dict()

    def batch_inference(self, dialogs: List, sampling_params: Dict=dict(), use_tqdm: bool=True) -> List[str] | List[List[str]]:
        raise NotImplementedError("Trying to run `batch_inference` from abstract class `ICLModel`.")
    
    def inference(self, dialog: List[Dict], sampling_params: Dict=dict()) -> str | List[str]:
        raise NotImplementedError("Trying to run `inference` from abstract class `ICLModel`.")

    def load_from_cache(self, dialogs: List[Dict], s_params: Dict={}) -> List[str] | List[List[str]]:
        if self.use_cache:
            outputs = list()
            for dialog in dialogs:
                cache_key = str(dialog[-1]["content"])
                output = self.cache.get(cache_key)
                outputs.append(output)
            return outputs
        else:
            return [None] * len(dialogs)
        
    def store_in_cache(self, outputs: List[str] | List[List[str]], dialogs: List[Dict], s_params: Dict={}) -> None:
        if self.use_cache:
            for dialog, output in zip(dialogs, outputs):
                cache_key = str(dialog[-1]["content"])
                self.cache[cache_key] = output

    def free_cache(self) -> None:
        self.cache = dict()