import os
import copy
import torch
import numpy as np
from typing import List, Dict, Tuple, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BatchEncoding
from loguru import logger
from torch.utils.data import Dataset
from omegaconf import DictConfig


class CausalModel(torch.nn.Module):
    """
    This class is only used for training.
    For inference (both via vllm and Pytorch), use InstructModel.
    """
    def __init__(self, causal_config: DictConfig, train: bool=False, use_cache: bool=False):
        super(CausalModel, self).__init__()
        self.causal_config = causal_config
        self.use_cache = use_cache
        # load model
        if train:
            logger.info(f"Loading model from pre-trained `{causal_config.model}`")
            self.model = AutoModelForCausalLM.from_pretrained(causal_config.model, device_map="auto")
        else:
            logger.info(f"Loading model from fine-tuned checkpoint at `{causal_config.trained_model_path}`")
            self.model = AutoModelForCausalLM.from_pretrained(causal_config.trained_model_path, device_map="cuda:0")  # fixed issues with multi-GPU setup
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Loaded model has {num_params} trainable parameters.")
        # load tokenizer
        if causal_config.get("tokenizer_path") and os.path.exists(causal_config.tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(causal_config.tokenizer_path, clean_up_tokenization_spaces=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(causal_config.model, clean_up_tokenization_spaces=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.pad_token_id


    def train(self, train_dataset: Dataset, dev_dataset: Dataset, compute_metrics_fct: Callable=None):
        # arguments for training
        logger.debug(f"Training arguments in use: {self.causal_config.training_params}")
        training_args = TrainingArguments(
            **self.causal_config.training_params
        )

        # create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset
        )

        # train
        trainer.train()

    def save(self):
        # save model
        model_path = self.causal_config.trained_model_path
        self.model.save_pretrained(model_path)

        # save tokenizer
        if self.causal_config.get("tokenizer_path"):
            tokenizer_path = self.causal_config.tokenizer_path
            self.tokenizer.save_pretrained(tokenizer_path)

    def inference(self, input_text: str, sampling_params: Dict):
        return self.batch_inference(input_texts=[input_text], sampling_params=sampling_params)[0]

    def batch_inference(self, input_texts: List[str], sampling_params: Dict):
        output_texts = list()
        self.model.eval()
        with torch.no_grad():
            for input_text in input_texts:
                input_encodings = self.tokenizer(
                    input_text,
                    truncation=True,
                    padding=False,
                    max_length=self.causal_config.max_input_length,
                    return_tensors="pt"
                )
                if torch.cuda.is_available():
                    input_encodings = {k: v.to("cuda:0") for k, v in input_encodings.items()}
                outputs = self.model.generate(
                    input_ids=input_encodings["input_ids"],
                    attention_mask=input_encodings["attention_mask"],
                    eos_token_id=self.tokenizer.pad_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **sampling_params
                )

                input_len = input_encodings["input_ids"].shape[1]
                outputs = outputs[:, input_len:]

                output_txt = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                # derive single str (instead of List[str])
                if sampling_params["num_return_sequences"] == 1:
                    output_txt = output_txt[0]
                logger.debug(f"output_txt={output_txt}")
                
                output_texts.append(output_txt)
        return output_texts


class DatasetCausalModel(torch.utils.data.Dataset):
    IGNORE_INDEX = -100
    
    def __init__(self, input_encodings: BatchEncoding, labels: BatchEncoding, dataset_length: int):
        self.input_encodings = input_encodings
        self.labels = labels
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        if self.labels is None:
            return {
                "input_ids": self.input_encodings["input_ids"][idx],
                "attention_mask": self.input_encodings["attention_mask"][idx]
            }
        label = self.labels[idx]
        item = {
            "input_ids": self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels": label,
        }
        return item

    def __len__(self):
        return self.dataset_length

    @staticmethod
    def format_dialog(tokenizer: AutoTokenizer, dialog: List[Dict]):
        return tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def prepare_encodings_train(tokenizer: AutoTokenizer, inputs: List[str], outputs: List[str], max_length: int) -> Tuple[BatchEncoding, torch.Tensor]:
        # concatenate inputs and outputs
        targets = [i + o for i, o in zip(inputs, outputs)]
        
        # tokenize targets (inputs + outputs)
        target_encodings = tokenizer(
            targets, 
            max_length=max_length, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        labels = copy.deepcopy(target_encodings["input_ids"])

        for label, input_ in zip(labels, inputs):
            prompt_tokens = tokenizer(input_, max_length=max_length, truncation=True, padding=False, return_tensors="pt")["input_ids"]
            prompt_length = prompt_tokens.shape[1]
            label_length = label.shape[0]
            if prompt_length >= max_length:
                logger.warning(f"Truncating target as label_length={label_length} (prompt_length={prompt_length})")
            else:
                logger.debug(f"No truncation with label_length={label_length} (prompt_length={prompt_length})")
            label[:prompt_length] = torch.full((prompt_length,), DatasetCausalModel.IGNORE_INDEX, dtype=torch.long)
        return target_encodings, labels
    
    @staticmethod
    def prepare_encodings_inference(tokenizer: AutoTokenizer, inputs: List[str], max_input_length: int) -> BatchEncoding:
        encodings = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length", return_tensors="pt")
        return encodings
    
    @staticmethod
    def trim_text_to_max_tokens(tokenizer: AutoTokenizer, text: str, max_num_tokens: int):
        """Trims the given text to the given maximum number of tokens for the tokenizer."""
        tokenized_prediction = tokenizer.encode(text)
        if len(tokenized_prediction) > max_num_tokens:
            logger.debug(f"Trimming input with {len(tokenized_prediction)} here.")
        trimmed_tokenized_prediction = tokenized_prediction[1: max_num_tokens + 1]
        trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
        return trimmed_prediction
