from omegaconf import DictConfig
from openai import OpenAI
from typing import List, Dict
from loguru import logger
from tqdm import tqdm

from reqap.llm.icl_model import ICLModel


class OpenAIModel(ICLModel):
    DEFAULT_SAMPLING_PARAMS = {
        "n": 1,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }

    def __init__(self, openai_config: DictConfig, use_cache: bool=True):
        super(OpenAIModel, self).__init__(openai_config, use_cache)
        self.client = OpenAI(
            api_key=openai_config.openai_key,
            organization=openai_config.openai_organization,
            project=openai_config.openai_project,
        )

    def run_system_prompt(self, system_prompt: str, sampling_params: dict=DEFAULT_SAMPLING_PARAMS) -> str:
        """
        Run the OpenAI model with only the provided system prompt as input.
        """
        response = self.client.chat.completions.create(
            model=self.icl_config["openai_model"],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
            ],
            **sampling_params
        )
        res = response.choices[0].message.content
        return res
    
    def batch_inference(self, dialogs: List, sampling_params: Dict=DEFAULT_SAMPLING_PARAMS, use_tqdm: bool=True) -> List[str] | List[List[str]]:
        """
        Generates responses for a batch of dialogs.
        """
        logger.debug(f"Batch inference called with {len(dialogs)} inputs...")

        # try with cache => reduces inputs by all cached entries
        outputs = self.load_from_cache(dialogs, sampling_params)
        actual_llm_dialogs = [dialog for dialog, output in zip(dialogs, outputs) if output is None]
        logger.debug(f"Leveraging cache for {len(dialogs) - len(actual_llm_dialogs)} dialogs (cache size: {len(self.cache)}). {len(actual_llm_dialogs)} dialogs remaining.")
        llm_dialog_indices = [i for i, output in enumerate(outputs) if output is None]

        if actual_llm_dialogs:
            responses = [self.inference(d, sampling_params) for d in tqdm(actual_llm_dialogs)]

            # obtain list of outputs
            for response, input_idx in zip(responses, llm_dialog_indices):
                outputs[input_idx] = response

        # fill cache
        self.store_in_cache(outputs, dialogs, sampling_params)
        return outputs
    
    def inference(self, dialog: List[Dict], sampling_params: Dict=DEFAULT_SAMPLING_PARAMS) -> str | List[str]:
        """
        Prompt the OpenAI model with a list of messages (each with keys "role" and "content").
        """
        params = self.DEFAULT_SAMPLING_PARAMS
        for key in sampling_params:
            if key in params:
                params[key] = sampling_params[key]

        logger.debug(f"Calling OpenAI model with sampling params: {params}")
        response = self.client.chat.completions.create(
            model=self.icl_config["openai_model"],
            messages=dialog,
            **params
        )
        if params["n"] > 1:
            res = [c.message.content.strip() for c in response.choices]
        else: 
            res = response.choices[0].message.content.strip()
        return res
