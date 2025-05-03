from typing import Dict, List

from reqap.llm.instruct_model import InstructModel
from reqap.llm.openai import OpenAIModel


class QUSupervisor:
    def __init__(self, icl_model: InstructModel | OpenAIModel):
        self.icl_model = icl_model
    
    def inference(self, dialog: List[Dict], sampling_params: Dict={}) -> str:
        return self.icl_model.inference(dialog, sampling_params)
    
    def batch_inference(self, dialogs: List[List[Dict]], sampling_params: Dict={}) -> List[str]:
        return self.icl_model.batch_inference(dialogs, sampling_params)
