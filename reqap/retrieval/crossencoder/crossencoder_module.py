import torch
from loguru import logger
from omegaconf import DictConfig
from typing import List, Dict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from reqap.classes.observable_event import ObservableEvent
from reqap.llm.crossencoder_model import CrossEncoderModel, compute_metrics
from reqap.retrieval.crossencoder.crossencoder_dataset import DatasetRetrieval, DatasetCrossEncoderFactory
from reqap.retrieval.splade.models import Splade


class CrossEncoder:
    def __init__(self, config: DictConfig, ce_config: DictConfig):
        self.config = config
        self.ce_config = ce_config
        self.model_events_loaded = False
        self.model_patterns_loaded = False

    def derive_retrieve_calls(self):
        # initialize modules
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)

        # run QU
        dataset_fac.derive_retrieve_calls(
            split="train",
            data_output_path=self.ce_config.crossencoder_retrieve_calls_train_set
        )
        dataset_fac.derive_retrieve_calls(
            split="dev",
            data_output_path=self.ce_config.crossencoder_retrieve_calls_dev_set
        )
        logger.debug("Done with deriving retrieve calls!")

    def derive_data(self, persona: str=None) -> None:
        # initialize modules
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)
        splade_model = Splade(self.config.splade.splade_model_type_or_path, agg="max")

        # derive data
        dataset_fac.derive_data(
            splade_model=splade_model,
            split="train",
            data_input_path=self.ce_config.crossencoder_retrieve_calls_train_set,
            data_output_path=self.ce_config.crossencoder_train_data,
            single_persona=persona
        )
        dataset_fac.derive_data(
            splade_model=splade_model,
            split="dev",
            data_input_path=self.ce_config.crossencoder_retrieve_calls_dev_set,
            data_output_path=self.ce_config.crossencoder_dev_data,
            single_persona=persona
        )
        logger.debug("Done with deriving data!")

    def derive_equivalent_retrieve_queries(self) -> None:
        # initialize modules
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)

        # derive data
        dataset_fac.derive_equivalent_retrieve_queries(
            input_path=self.ce_config.crossencoder_train_data,
            output_path=self.ce_config.crossencoder_train_equivalent_queries
        )
        dataset_fac.derive_equivalent_retrieve_queries(
            input_path=self.ce_config.crossencoder_dev_data,
            output_path=self.ce_config.crossencoder_dev_equivalent_queries
        )
        logger.debug("Done with deriving equivalent RETRIEVE queries!")

    def train_ce_events(self) -> None:
        # init
        self.model_events = CrossEncoderModel(ce_config=self.ce_config, train=True)
        tokenizer = self.model_events.tokenizer
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)

        # create datasets
        train_set = dataset_fac.create(tokenizer, self.ce_config.crossencoder_train_data, input_type="event")
        logger.info(f"Derived train set with {len(train_set)} instances.")
        dev_set = dataset_fac.create(tokenizer, self.ce_config.crossencoder_dev_data, input_type="event")
        logger.info(f"Derived dev set with {len(dev_set)} instances.")

        # train
        self.model_events.train(train_set, dev_set)
        self.model_events.save()
        self.model_events_loaded = True
        logger.info(f"Done with training.")
        self.evaluate_ce_events()
    
    def train_ce_patterns(self) -> None:
        # init
        self.model_patterns = CrossEncoderModel(ce_config=self.ce_config, train=True, num_outputs=3)
        tokenizer = self.model_patterns.tokenizer
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)

        # create datasets
        train_set = dataset_fac.create(tokenizer, self.ce_config.crossencoder_train_data, input_type="pattern")
        logger.info(f"Derived train set with {len(train_set)} instances.")
        dev_set = dataset_fac.create(tokenizer, self.ce_config.crossencoder_dev_data, input_type="pattern")
        logger.info(f"Derived dev set with {len(dev_set)} instances.")

        # train
        self.model_patterns.train(train_set, dev_set)
        self.model_patterns.save()
        self.model_patterns_loaded = True
        logger.info(f"Done with training.")
        self.evaluate_ce_patterns()

    def evaluate_ce_events(self) -> None:
        # load model
        self.load_models(events_only=True)            
        
        # init
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)
        tokenizer = self.model_events.tokenizer

        # load dev set
        input_path = self.ce_config.crossencoder_dev_data
        dataset = dataset_fac.create(tokenizer, input_path, input_type="event")
        logger.info(f"Loaded dev set with {len(dataset)} instances.")

        # eval
        metrics = self._evaluate(
            self.model_events,
            dataset,
            batch_size=self.ce_config.crossencoder_inference_batch_size
        )
        logger.info(f"Metrics: {metrics}")

    def evaluate_ce_patterns(self) -> None:
        # load model
        self.load_models(patterns_only=True)
        
        # init
        dataset_fac = DatasetCrossEncoderFactory(config=self.config, ce_config=self.ce_config)
        tokenizer = self.model_patterns.tokenizer

        # load dev set
        input_path = self.ce_config.crossencoder_dev_data
        dataset = dataset_fac.create(tokenizer, input_path, input_type="pattern")
        logger.info(f"Loaded dev set with {len(dataset)} instances.")

        # eval
        metrics = self._evaluate(
            self.model_patterns,
            dataset,
            batch_size=self.ce_config.crossencoder_inference_batch_size
        )
        logger.info(f"Metrics: {metrics}")
    
    @staticmethod
    def _evaluate(model: CrossEncoderModel, dataset: Dataset, batch_size: int):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds = list()
        all_labels = list()
        all_inputs = list()
        for batch in tqdm(dataloader):
            labels = batch["label"]
            inputs = model.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            preds = model._inference(batch, batch_size)
            
            all_preds += preds.tolist()
            if len(labels) == 3:
                labels = [(i1, i2, i3) for i1, i2, i3 in zip(labels[0].tolist(), labels[1].tolist(), labels[2].tolist())]
            else:
                labels = [(i1, i2) for i1, i2 in zip(labels[0].tolist(), labels[1].tolist())]
            for l in labels:
                assert sum(l) == 1.0
                
            all_labels += labels
            all_inputs += inputs

        # convert into correct format
        all_preds = np.array(all_preds, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.float32)

        # DEV: print incorrect ones
        probs = torch.nn.functional.softmax(torch.tensor(all_preds), dim=-1)
        pred_classes = torch.argmax(probs, dim=-1).numpy()
        for label, pred, input_str in zip(all_labels, pred_classes, all_inputs):
            label = np.argmax(label)
            if label != pred:
                logger.error(f"Input: `{input_str}` => Predicted class {pred}, correct is {label}")
        
        # compute metrics
        metrics = compute_metrics(eval_preds=(all_preds, all_labels))
        return metrics

    def run_for_events(self, input_query: str, input_events: List[ObservableEvent], ordered: bool=False) -> List[Dict]:
        self.load_models()
        
        # run model
        input_tuples = [DatasetRetrieval.prepare_event_input(input_query, e) for e in input_events]
        max_length = self.ce_config.model_events.max_length
        batch_size = self.ce_config.model_events.inference_batch_size
        probs, classes = self.model_events.inference(input_tuples, max_length, batch_size)
        
        # process result
        result = list()
        for e, p, c in zip(input_events, probs, classes):
            p = [float(prob) for prob in p]
            result.append({
                "obs_event": e,
                "class": int(c),
                "score": p[int(c)],
                "probabilities": p,
            })
        if ordered:
            result = sorted(result, key=lambda res: res["score"], reverse=True)
        return result
    
    def run_for_patterns(self, input_query: str, patterns: List[str]) -> List[Dict]:
        self.load_models()

        # run model
        input_tuples = [DatasetRetrieval.prepare_pattern_input(input_query, p) for p in patterns]
        max_length = self.ce_config.model_patterns.max_length
        batch_size = self.ce_config.model_patterns.inference_batch_size
        probs, classes = self.model_patterns.inference(input_tuples, max_length, batch_size)
        
        # process result
        result = list()
        for p, prob, c in zip(patterns, probs, classes):
            prob = [float(pr) for pr in prob]
            result.append({
                "pattern": p,
                "class": int(c),
                "score": prob[int(c)],
                "probabilities": prob,
            })
        return result

    def load_models(self, events_only: bool=False, patterns_only: bool=False) -> None:
        # load model for scoring events
        if not self.model_events_loaded and not patterns_only and self.ce_config.get("model_events", True):
            model_cfg = self.ce_config.model_events if "model_events" in self.ce_config else self.ce_config
            self.model_events = CrossEncoderModel(model_cfg, train=False, num_outputs=2)
            self.model_events_loaded = True
            if torch.cuda.is_available():
                self.model_events.model = self.model_events.model.cuda()
        
        # load model for scoring patterns
        if not self.model_patterns_loaded and not events_only and self.ce_config.get("model_patterns", True):
            model_cfg = self.ce_config.model_patterns if "model_patterns" in self.ce_config else self.ce_config
            self.model_patterns = CrossEncoderModel(model_cfg, train=False, num_outputs=3)
            self.model_patterns_loaded = True
            if torch.cuda.is_available():
                self.model_patterns.model = self.model_patterns.model.cuda()
