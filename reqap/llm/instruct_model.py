import os
import torch
if torch.cuda.is_available():
    import vllm

from loguru import logger
from typing import List, Dict
from omegaconf import DictConfig

from reqap.llm.icl_model import ICLModel


class InstructModel(ICLModel):
    DEF_VLLM_TENSOR_PARALLEL_SIZE = 4
    DEF_VLLM_GPU_MEMORY_UTILIZATION = 0.85
    DEF_VLLM_KWARGS = {"dtype": "float16"}
    DEF_VLLM_MAX_MODEL_LEN = 8196

    DEFAULT_SAMPLING_PARAMS = {
        "n": 1,
        "top_p": 0.8,
        "temperature": 0.6,
        "max_tokens": 4096,
        "skip_special_tokens": True
    }

    def __init__(self, instruct_config: DictConfig, use_cache: bool=False):
        super(InstructModel, self).__init__(instruct_config, use_cache)
        
        vllm_tensor_parallel_size = int(os.environ.get("GPU_NUM", self.DEF_VLLM_TENSOR_PARALLEL_SIZE))
        vllm_gpu_memory_utilization = instruct_config.get("vllm_gpu_memory_utilization", self.DEF_VLLM_GPU_MEMORY_UTILIZATION)
        vllm_kwargs = instruct_config.get("vllm_kwargs", self.DEF_VLLM_KWARGS)
        vllm_max_model_len = instruct_config.get("vllm_max_model_len", self.DEF_VLLM_MAX_MODEL_LEN)

        # initialize the model with vllm
        self.max_model_len = vllm_max_model_len
        model_path = self.icl_config.icl_model_path
        logger.info(f"Using {vllm_gpu_memory_utilization} of {vllm_tensor_parallel_size} GPUs for loading the model {model_path}.")
        self.llm = vllm.LLM(
            model_path,
            worker_use_ray=True,
            enforce_eager=True,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_model_len=vllm_max_model_len,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            **vllm_kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()

    def batch_inference(self, dialogs: List, sampling_params: Dict={}, use_tqdm: bool=True) -> List[str] | List[List[str]]:
        """
        Generates responses for a batch of dialogs.
        """
        return self._batch_inference(dialogs, sampling_params=sampling_params, use_tqdm=use_tqdm)
    
    def inference(self, dialog: List[Dict], sampling_params: Dict={}) -> str | List[str]:
        """
        Generates the response for a provided dialog.
        """
        return self._batch_inference([dialog], sampling_params=sampling_params, use_tqdm=False)[0]
    
    def run_system_prompt(self, system_prompt: str, sampling_params: Dict={}) -> str:
        """
        Run the model with the provided system prompt as input.
        """
        dialog = [{"role": "system", "content": system_prompt}]
        output = self.inference(dialog, sampling_params=sampling_params)
        return output
    
    def _prepare_dialog(self, dialog: List[Dict]) -> str:
        llm_input = self.tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True,
        )
        return llm_input
    
    def _batch_inference(self, dialogs: List[Dict], sampling_params: Dict={}, use_tqdm: bool=True) -> List[str] | List[List[str]]:
        """
        Generates responses for a batch of formatted inputs.
        """
        # set sampling params
        s_params = self.DEFAULT_SAMPLING_PARAMS.copy()
        s_params.update(sampling_params)
        sampling_params = vllm.SamplingParams(**s_params)

        # try with cache => reduces inputs by all cached entries
        outputs = self.load_from_cache(dialogs, s_params)
        actual_llm_dialogs = [dialog for dialog, output in zip(dialogs, outputs) if output is None]
        logger.debug(f"Leveraging cache for {len(dialogs) - len(actual_llm_dialogs)} dialogs (cache size: {len(self.cache)}). {len(actual_llm_dialogs)} dialogs remaining.")
        llm_dialog_indices = [i for i, output in enumerate(outputs) if output is None]

        # convert dialogs to strings
        actual_llm_inputs = [self._prepare_dialog(d) for d in actual_llm_dialogs]

        # run generation
        if actual_llm_inputs:
            responses = self.llm.generate(
                actual_llm_inputs,
                sampling_params,
                use_tqdm=use_tqdm
            )

            # obtain list of outputs
            for response, input_idx in zip(responses, llm_dialog_indices):
                output_texts = [output.text.strip() for output in response.outputs]
                # flatten single outputs
                if len(output_texts) == 1:
                    output_texts = output_texts[0]
                outputs[input_idx] = output_texts

        # fill cache
        self.store_in_cache(outputs, dialogs, s_params)
        return outputs
