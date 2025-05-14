import os
import torch
import numpy as np
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, BatchEncoding
from torch.utils.data import Dataset
from loguru import logger
from typing import List, Dict, Tuple, Callable


def compute_metrics(eval_preds, tokenizer):
    """ Add exact match metric. """
    preds, labels = eval_preds

    # handle -100 in preds and labels (convert to pad token ID for decoding)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # compute exact match metric
    correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    for p, l in zip(decoded_preds, decoded_labels):
        is_correct = p ==  l
        logger.debug(f"Correct: {is_correct}, p={p}, l={l}")
    exact_match = correct / len(decoded_preds)
    return {
        "exact_match": round(exact_match, 3),
        "total_instances": len(decoded_preds)
    }


class Seq2SeqModel(torch.nn.Module):
    cache = dict()
    
    def __init__(self, seq2seq_config: DictConfig, train: bool=True, use_cache: bool=False):
        super(Seq2SeqModel, self).__init__()
        self.seq2seq_config = seq2seq_config
        self.use_cache = use_cache
        # load model
        if train:
            logger.info(f"Loading model from pre-trained `{seq2seq_config.model}`")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_config.model)
        else:
            logger.info(f"Loading model from fine-tuned checkpoint at `{seq2seq_config.trained_model_path}`")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_config.trained_model_path)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Loaded model has {num_params} trainable parameters.")
        
        # load tokenizer
        if seq2seq_config.get("tokenizer_path") and os.path.exists(seq2seq_config.tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(seq2seq_config.tokenizer_path, clean_up_tokenization_spaces=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(seq2seq_config.model, clean_up_tokenization_spaces=True)

        # enable inference on multiple GPUs
        if torch.cuda.device_count() > 1:
            logger.debug(f"Using {torch.cuda.device_count()} GPUs for seq2seq inference")
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(torch.device("cuda")) 

    def train(self, train_dataset: Dataset, dev_dataset: Dataset, compute_metrics_fct: Callable=None) -> Seq2SeqTrainer:
        def _compute_metrics(eval_preds):
            if not compute_metrics_fct is None:
                return compute_metrics_fct(eval_preds, self.tokenizer)
            return compute_metrics(eval_preds, self.tokenizer)
        
        # arguments for training
        logger.debug(f"Training arguments in use: {self.seq2seq_config.training_params}")
        training_args = Seq2SeqTrainingArguments(
            **self.seq2seq_config.training_params
        )

        # create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics,
        )

        # train
        trainer.train()
        return trainer
  
    def batch_inference(self, input_texts: List[str], generation_params: Dict=dict()) -> List[str]:
        # try with cache => reduces inputs by all cached entries
        batch_outputs = self.load_from_cache(input_texts)
        actual_input_texts = [input_text for input_text, output in zip(input_texts, batch_outputs) if output is None]
        input_text_indices = [i for i, output in enumerate(batch_outputs) if output is None]

        if len(actual_input_texts):
            # encode
            input_encodings = self.tokenizer(
                actual_input_texts,
                padding=True,
                truncation=True,
                max_length=self.seq2seq_config.max_input_length,
                return_tensors="pt",
            )
            
            # generate
            if torch.cuda.is_available():
                input_encodings = input_encodings.to(torch.device("cuda"))
            self.model.eval()
            with torch.no_grad():
                model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                outputs = model.generate(
                    input_ids=input_encodings["input_ids"],
                    attention_mask=input_encodings["attention_mask"],
                    max_length=self.seq2seq_config.max_output_length,
                    **generation_params
                )

            # decoding
            results = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # process and "join" with cached results
            if generation_params.get("num_return_sequences", 1) > 1:
                n = generation_params["num_return_sequences"]
                assert len(results) == n * len(actual_input_texts), f"Assertion failed: len(results)={len(results)}, n={n} and len(actual_input_texts)={len(actual_input_texts)}"
                for i, input_idx in zip(range(len(actual_input_texts)), input_text_indices):
                    start_idx = i*n
                    end_idx = i*n + n
                    batch_outputs[input_idx] = results[start_idx:end_idx]
            else:
                for result, input_idx in zip(results, input_text_indices):
                    batch_outputs[input_idx] = result

        # store in cache
        self.store_in_cache(batch_outputs, input_texts)

        return batch_outputs

    def single_inference(self, input_text: str, generation_params: Dict=dict()) -> str:
        return self.batch_inference([input_text], generation_params)[0]

    def save(self):
        # save model
        model_path = self.seq2seq_config.trained_model_path
        self.model.save_pretrained(model_path)

        # save tokenizer
        if self.seq2seq_config.get("tokenizer_path"):
            tokenizer_path = self.seq2seq_config.tokenizer_path
            self.tokenizer.save_pretrained(tokenizer_path)
    
    def load_from_cache(self, input_texts: List[str]) -> List[str] | List[List[str]]:
        if self.use_cache:
            batch_outputs = list()
            for input_text in input_texts:
                cache_key = str(input_text)
                output = self.cache.get(cache_key)
                batch_outputs.append(output)
            return batch_outputs
        else:
            return [None] * len(input_texts)
        
    def store_in_cache(self, batch_outputs: List[str] | List[List[str]], input_texts: List[str]) -> None:
        if self.use_cache:
            for input_text, output in zip(input_texts, batch_outputs):
                cache_key = str(input_text)
                self.cache[cache_key] = output


class DatasetSeq2Seq(torch.utils.data.Dataset):
    def __init__(self, input_encodings: BatchEncoding, output_encodings: BatchEncoding, dataset_length: int):
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        labels = self.output_encodings["input_ids"][idx]
        item = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }
        return item

    def __len__(self):
        return self.dataset_length

    @staticmethod
    def tokenize(tokenizer: AutoTokenizer, inputs: List[str], outputs: List[str], max_input_length: int, max_output_length: int) -> Tuple[BatchEncoding, BatchEncoding]:
        # encode inputs
        input_encodings = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        # check for overflowing inputs
        exceeding_count = 0
        for text in inputs:
            if len(tokenizer.encode(text)) > max_input_length:
                exceeding_count += 1
        logger.debug(f"{exceeding_count} inputs have been truncated.")

        # encode outputs
        output_encodings = tokenizer(
            outputs,
            padding="max_length",
            truncation=True,
            max_length=max_output_length,
            return_tensors="pt",
        )   
        # check for overflowing outputs
        exceeding_count = 0
        for text in outputs:
            if len(tokenizer.encode(text)) > max_output_length:
                exceeding_count += 1
        logger.debug(f"{exceeding_count} outputs have been truncated.")

        return input_encodings, output_encodings
