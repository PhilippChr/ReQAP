import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, BatchEncoding
from collections import defaultdict
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm

from reqap.library.library import batchify


def compute_metrics(eval_preds):
    """ Add accuracy metrics. """
    logits, labels = eval_preds

    # compute probabilities and predicted classes
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    preds = torch.argmax(probs, dim=-1).numpy()

    if labels.ndim >= 1:
        labels = np.argmax(labels, axis=1)
    correct = (preds == labels).sum()
    total = len(labels)
    accuracy = correct / total
    num_classes = logits.shape[-1]

    # compute accuracy when rejecting predictions with low confidence (<0.6)
    preds_at_06 = list()
    for pred, prob in zip(preds, probs):
        if prob[pred] < 0.6:
            pred = num_classes - 2
        preds_at_06.append(pred)
    preds_at_06 = np.array(preds_at_06)
    correct_at_06 = (preds_at_06 == labels).sum()
    accuracy_at_06 = correct_at_06 / total

    # compute fractions of predictions
    fractions = defaultdict(lambda: 0)
    correct = defaultdict(lambda: 0)
    for pred, label in zip(preds, labels):
        key = f"fraction_{pred}"
        fractions[key] += 1
        if pred == label:
            key = f"correct_{pred}"
            correct[key] += 1

    # derive fractions for each absolute number
    for key in correct:
        fraction_key = key.replace("correct", "fraction")
        correct[key] = round(correct[key] / fractions[fraction_key], 3)
    for key in fractions:
        fractions[key] = round(fractions[key] / total, 3)

    # return metrics
    return {
        "accuracy": accuracy,
        "accuracy@0.60": accuracy_at_06,
        "total_instances": total,
        **fractions,
        **correct
    }


class CrossEncoderModel(torch.nn.Module):
    def __init__(self, ce_config: DictConfig, train: bool=True, num_outputs: int=2):
        super(CrossEncoderModel, self).__init__()
        self.ce_config = ce_config
        
        # load model
        if train:
            logger.info(f"Loading model from pre-trained `{ce_config.crossencoder_model}`")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ce_config.crossencoder_model
            )
            self.model.classifier = nn.Linear(self.model.config.hidden_size, num_outputs)  # adjust classifier to num_outputs
        else:
            logger.info(f"Loading model from fine-tuned checkpoint at `{ce_config.crossencoder_trained_model_path}`")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ce_config.crossencoder_trained_model_path, num_labels=num_outputs
            )
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Loaded model has {num_params} trainable parameters.")

        # load tokenizer
        if ce_config.get("crossencoder_tokenizer_path") and os.path.exists(ce_config.crossencoder_tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(ce_config.crossencoder_tokenizer_path, clean_up_tokenization_spaces=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(ce_config.crossencoder_model, clean_up_tokenization_spaces=True)

        # enable inference on multiple GPUs
        if torch.cuda.device_count() > 1:
            logger.debug(f"Using {torch.cuda.device_count()} GPUs for crossencoder")
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(torch.device("cuda"))

    def train(self, train_dataset: Dataset, dev_dataset: Dataset):
        # arguments for training
        logger.debug(f"Training arguments in use: {self.ce_config.training_params}")
        training_args = TrainingArguments(**self.ce_config.training_params)

        # create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        # train
        trainer.train()

    def inference(self, input_tuples: Tuple[str, str], max_length: int, batch_size: int):
        # prepare inputs
        tokenized_inputs = self.tokenizer(
            [i[0] for i in input_tuples],
            [i[1] for i in input_tuples],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )

        # run model
        logits = self._inference(tokenized_inputs, batch_size)

        # obtain classes
        probs = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        return probs, predicted_classes
    
    def _inference(self, tokenized_inputs: BatchEncoding, batch_size: int):
        self.model.eval()
        with torch.no_grad(): 
            if isinstance(tokenized_inputs, dict):
                batch = {key: value.to(torch.device("cuda")) if torch.cuda.is_available() else value for key, value in tokenized_inputs.items() if key != "label"}
                outputs = self.model(**batch)
                logits = outputs.logits
                # drop from CUDA
                del(batch)
                del(outputs)
                return logits
            else:
                if torch.cuda.is_available():
                    tokenized_inputs = tokenized_inputs.to(torch.device("cuda"))
                # run model
                logits = None
                num_batches = math.ceil(len(tokenized_inputs["input_ids"]) / batch_size)
                for batch in tqdm(batchify(tokenized_inputs, batch_size), total=num_batches):
                    outputs = self.model(**batch)
                    logits = outputs.logits if logits is None else torch.cat((logits, outputs.logits), dim=0)
                return logits

    def save(self):
        # save model
        model_path = self.ce_config.crossencoder_trained_model_path
        self.model.save_pretrained(model_path)

        # save tokenizer
        if self.ce_config.get("crossencoder_tokenizer_path"):
            tokenizer_path = self.ce_config.crossencoder_tokenizer_path
            self.tokenizer.save_pretrained(tokenizer_path)


class DatasetCrossEncoder(torch.utils.data.Dataset):
    def __init__(self, input_encodings: BatchEncoding, labels: List[List[int]], dataset_length: int):
        self.input_encodings = input_encodings
        self.labels = labels
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        label = self.labels[idx]
        item = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "label": label,
        }
        return item

    def __len__(self):
        return self.dataset_length