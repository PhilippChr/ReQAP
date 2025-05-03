import re
import math
import torch
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Any, Callable, Optional
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from datetime import datetime, time, timedelta
import datetime as datetime_lib
DEFAULT_TZ = datetime_lib.timezone.utc
from tqdm import tqdm

from reqap.library.library import batchify, store_json
from reqap.classes.computed_event import ComputedEvent
from reqap.classes.observable_event import ObservableEvent, ObservableEventType
from reqap.llm.seq2seq_model import Seq2SeqModel, compute_metrics
from reqap.extract.extract_dataset import DatasetExtract, DatasetExtractFactory, is_simple_lookup, obtain_derival_function
from reqap.llm.openai import OpenAIModel



class ExtractModule:
    QUANTITY_WITH_UNIT_PATTERN = re.compile(r"^(\d+\.?\d*)\s*([a-zA-Z%]+)$")

    def __init__(self, extract_config: DictConfig):
        self.extract_config = extract_config
        self.model_loaded = False

        # dict to store (event_type, attribute)-pairs for which a simple lookup is sufficient
        # stores the corresponding key-path to access the value
        self.simple_lookup_dict = dict()  

    def derive_attributes(self, config: DictConfig) -> None:
        # initialize modules
        dataset_fac = DatasetExtractFactory(self.extract_config)

        # derive attributes
        logger.debug("Deriving attributes from train set...")
        attribute_triples_dict = dataset_fac.derive_relevant_attributes(
            config=config,
            split="train",
        )
        # store
        attributes = list()
        for (attribute, event_type), event in attribute_triples_dict.items():
            attributes.append([attribute, event_type, event])
        data_output_path = self.extract_config.extract_training.extract_attributes_train_path
        store_json(data_output_path, attributes)
        logger.debug("Done with deriving attributes for train set!")
        
        # derive attributes
        logger.debug("Deriving attributes from dev set...")
        attribute_triples_dict = dataset_fac.derive_relevant_attributes(
            config=config,
            split="dev",
        )
        # store
        attributes = list()
        for (attribute, event_type), event in attribute_triples_dict.items():
            attributes.append([attribute, event_type, event])
        data_output_path = self.extract_config.extract_training.extract_attributes_dev_path
        store_json(data_output_path, attributes)
        logger.debug("Done with deriving attributes for dev set!")

    def derive_extract_call_attributes(self, config) -> None:
        dataset_fac = DatasetExtractFactory(self.extract_config)
        
        logger.debug("Deriving attribute calls for train set...")
        input_path = config.qu.qu_result_paths["train"]
        output_path = self.extract_config.extract_training.extract_attribute_calls["train"]
        dataset_fac.derive_extract_call_attributes(input_path, output_path)

        logger.debug("Deriving attribute calls for dev set...")
        input_path = config.qu.qu_result_paths["dev"]
        output_path = self.extract_config.extract_training.extract_attribute_calls["dev"]
        dataset_fac.derive_extract_call_attributes(input_path, output_path)
        logger.debug("Done with deriving attribute calls!")

    def derive_attribute_mappings(self, config) -> None:
        dataset_fac = DatasetExtractFactory(self.extract_config)
        icl_model = OpenAIModel(openai_config=config.openai, use_cache=False)
        
        logger.debug("Deriving attribute mapping for train set...")
        input_path = self.extract_config.extract_training.extract_attributes_train_path
        output_path = self.extract_config.extract_training.extract_attribute_mapping["train"]
        dataset_fac.derive_attribute_mapping(icl_model, input_path, output_path)

        logger.debug("Deriving attribute mapping for dev set...")
        input_path = self.extract_config.extract_training.extract_attributes_dev_path
        output_path = self.extract_config.extract_training.extract_attribute_mapping["dev"]
        dataset_fac.derive_attribute_mapping(icl_model, input_path, output_path)
        logger.debug("Done with deriving attribute mappings!")
        
    def derive_data(self, config: DictConfig) -> None:
        dataset_fac = DatasetExtractFactory(self.extract_config)
        dataset_fac.derive_data(config, "train")
        dataset_fac.derive_data(config, "dev")

    def train(self) -> None:
        dataset_fac = DatasetExtractFactory(self.extract_config)
        self.extract_model = Seq2SeqModel(seq2seq_config=self.extract_config, train=True)
        tokenizer = self.extract_model.tokenizer
        dataset_cfg = self.extract_config.extract_training.extract_data

        for i, complexity in enumerate(dataset_cfg["train"]):
            complexity_levels = len(dataset_cfg["train"])
            logger.info(f"Starting with {complexity} training...")
            
            # load train set
            input_path = dataset_cfg["train"][complexity]
            train_set = dataset_fac.create(tokenizer, input_path)
            logger.info(f"Loaded train set with {len(train_set)} instances.")
            
            # load dev set
            input_path = dataset_cfg["dev"][complexity]
            dev_set = dataset_fac.create(tokenizer, input_path)
            logger.info(f"Loaded dev set with {len(dev_set)} instances.")
            
            # train
            self.extract_model.train(train_set, dev_set)
            logger.info(f"Done with training {i+1}/{complexity_levels}.")

        self.model_loaded = True
        self.extract_model.save()
        self.evaluate()

    def evaluate(self) -> None:
        # load model
        self.load()            
        
        # init
        dataset_fac = DatasetExtractFactory(self.extract_config)
        tokenizer = self.extract_model.tokenizer
        evaluation_cfg = self.extract_config.extract_training.evaluation_data

        metrics = dict()
        for _, complexity in enumerate(evaluation_cfg):
            logger.info(f"Evaluation for {complexity} questions...")
            
            # load dev set
            input_path = evaluation_cfg[complexity]
            dataset = dataset_fac.create(tokenizer, input_path)
            logger.info(f"Loaded dev set with {len(dataset)} instances.")

            # eval
            met = self._evaluate(
                self.extract_model,
                dataset,
                batch_size=self.extract_config.training_params.per_device_eval_batch_size,
                max_length=self.extract_config.training_params.generation_max_length,
                generation_params=self.extract_config.generation_params
            )
            metrics[complexity] = met
        logger.info(f"Metrics: {metrics}")
    
    @staticmethod
    def _evaluate(model: Seq2SeqModel, dataset: Dataset, batch_size: int, max_length: int, generation_params: Dict=dict()):
        logger.debug(f"generation_params={generation_params}")
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds = list()
        all_labels = list()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(model.model.device)
            attention_mask = batch["attention_mask"].to(model.model.device)
            labels = batch["labels"].to(model.model.device)

            with torch.no_grad():
                preds = model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    **generation_params
                )
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        # convert lists to np arrays
        all_preds = np.array(all_preds, dtype=object)
        all_labels = np.array(all_labels, dtype=object)
        
        # compute metrics
        metrics = compute_metrics(
            eval_preds=(all_preds, all_labels),
            tokenizer=model.tokenizer
        )
        return metrics

    def run(self, computed_events: List[ComputedEvent], attributes: List[str], types: List[str], persona_dict: Optional[Dict]=None) -> List[ComputedEvent]:
        def _attribute_is_set(ce: ComputedEvent, attribute: str) -> bool:
            if ce.attributes.get(attribute):
                return True
            return False
        def _get_value(ce: ComputedEvent, attribute: str) -> Any:
            return ce.attributes.get(attribute)
        
        # derive all computation inputs
        computation_inputs = list()
        for ce in computed_events:
            for attr, t in zip(attributes, types):
                if _attribute_is_set(ce, attr):
                    ce.attributes[attr] = _get_value(ce, attr)
                else:
                    for oe in ce.get_observable_events():
                        computation_inputs.append({
                            "observable_event": oe,
                            "attribute": attr,
                            "type": t
                        })

        # do computations: derives a mapping from (oe.id, attr) to value
        computation_results: Dict = self._run(computation_inputs, persona_dict)

        # process results
        filtered_computed_events = list()  
        for ce in computed_events:
            all_attributes_set = [False] * len(attributes)
            for attr_idx, attr in enumerate(attributes):
                if _attribute_is_set(ce, attr):
                    all_attributes_set[attr_idx] = True
                    continue
                attr_results = [computation_results[(oe.id, attr)] for oe in ce.get_observable_events()]
                attr_results = [res for res in attr_results if res is not None]
                if attr_results:
                    ce.attributes[attr] = attr_results[0]  # consider first result (there should not be multiple in theory)
                    all_attributes_set[attr_idx] = True
            # only add if all attributes are set
            if all(attribute_set for attribute_set in all_attributes_set):
                filtered_computed_events.append(ce)
        return filtered_computed_events

    def _run(self, computation_inputs: List[Dict], persona_dict: Optional[Dict]=None) -> Dict:
        """
        Run the EXTRACT inference on input dicts.
        Each input dict has the keys `observable_event`, `attribute` and `type`.
        - TODO: Incorporating expected value type into seq2seq input might help.
        - TODO: Incorporating the corresponding retrieval query would provide additional context:
            Results for a mail event and key="date" can be different for "I sent a mail" vs. "I was working out".
        """
        self.load()

        # for ablation study: simple look-up of keys
        if self.extract_config.get("keys_only", False):
            computation_results = dict()
            for computation_input in computation_inputs:
                oe = computation_input["observable_event"]
                attr = computation_input["attribute"]
                attr_type = computation_input["type"]
                value = oe.event_data.get(attr, None)
                if value is None:
                    value = getattr(oe, attr, None)
                if not value is None:
                    value = self.apply_type_loading(value, attr_type, attr, oe)
                computation_results[(oe.id, attr)] = value
            return computation_results

        # group by attribute
        inputs_grouped_by_attr = defaultdict(lambda: list())
        for computation_input in computation_inputs:
            attr = computation_input["attribute"]
            inputs_grouped_by_attr[attr].append(computation_input)

        # inference (per attribute)
        computation_results = dict()
        for attr in inputs_grouped_by_attr:
            # run model
            events_for_attr = [input_["observable_event"] for input_ in inputs_grouped_by_attr[attr]]
            attr_type = inputs_grouped_by_attr[attr][0]["type"] if inputs_grouped_by_attr[attr] else None
            inference_result = self.inference(events_for_attr, attr, attr_type, persona_dict)
            
            # process result
            for oe in events_for_attr:
                value = inference_result[(oe.id, attr)]
                
                # check if NONE_VALUE
                if value == DatasetExtract.NONE_VALUE:
                    computation_results[(oe.id, attr)] = None
                    continue
                
                # apply type loading function
                computation_results[(oe.id, attr)] = self.apply_type_loading(value, attr_type, attr, oe)
        return computation_results
    
    @staticmethod
    def apply_type_loading(value: Any, attr_type: Callable, attr: str, oe: ObservableEvent) -> Any:
        ATTR_TYPE_MAPPING = {
            list: lambda fct: list(eval(fct))  # for list loading, first eval string, then convert to list
        }
        
        # make use of specific function conversion
        attr_type = ATTR_TYPE_MAPPING.get(attr_type, attr_type)

        # drop units in case an int/float is requested
        if attr_type in (int, float) and isinstance(value, str) and ExtractModule.QUANTITY_WITH_UNIT_PATTERN.match(value.strip()):
            new_value = value.split()[0].strip()
            logger.debug(f'Transforming number with unit to plain number: value={value} -> new_value={new_value}')
            value = new_value

        # try applying loading function directly
        exceptions = list()
        try:
            value = attr_type(value)
            return value
        except (TypeError, ValueError, SyntaxError) as e:
            if value == "":
                return None
            exception_msg = f"Encountered {e.__class__} when trying to run conversion for attr=`{attr}`, value=`{value}`, oe=`{oe.to_dict()}`"
            exceptions.append(exception_msg)
        
        # try loading into Python first, then applying loading function
        try:
            value_eval = eval(value)
            value = attr_type(value_eval)
            return value
        except Exception as e:
            exception_msg = f"Encountered {e.__class__} for type(value)=`{type(value)}`."
            exceptions.append(exception_msg)

        # log that there was a problem
        logger.debug(f"Cannot convert {value}: {exceptions}")

        # fall-back when conversion failed: return None
        return None

    def inference(self, events: List[ObservableEvent], attribute: str, attribute_type: str, persona_dict: Optional[Dict]=None) -> Dict[Tuple[int, str], str]:
        """
        Run inference.
        The output is a dictionary mapping from (event ID, attribute) pairs to values (which are strings at this point).
        """
        ### EFFICIENT WITH SAMPLING BASED LOOKUP INFERENCE 
        sample_size = self.extract_config.extract_sample_size_simple_lookup
        inference_result: Dict[Tuple[int, str], str] = dict()

        # derive computations for stage 1 inference
        # => this means that max. `sample_size` inputs are collected for a (event, attribute) combination.
        # If for all of them, the value is equivalent to the lookup, then the combination is remembered as
        # simple lookup (within self.simple_lookup_dict).
        stage1_inference = defaultdict(lambda: list())  # list of computations for pruning
        waiting_list = defaultdict(lambda: list())  # maps from group to events
        for ev in events:
            event_type = ev.event_type
            group = (event_type, attribute)
            # group has been seen before, and stage1 outcome is clear
            if group in self.simple_lookup_dict:
                waiting_list[group].append(ev)
            # add computation to stage1 inference
            elif len(stage1_inference[group]) < sample_size:
                stage1_inference[group].append(ev)
            # add to waiting list, as sample for stage1 already full for this group
            else:
                waiting_list[group].append(ev)

        # stage 1: inference, process results, identify simple lookups
        def save_simple_lookup(config: DictConfig, is_simple_lookup_list: List[bool]) -> bool:
            simple_lookup_threshold = config.get("simple_lookup_threshold", 1.0)
            simple_lookup_ratio = sum(1 if b else 0 for b in is_simple_lookup_list)/len(is_simple_lookup_list)
            if simple_lookup_ratio >= simple_lookup_threshold:
                logger.debug(f"Detected new simple lookup: simple_lookup_ratio={simple_lookup_ratio}, simple_lookup_threshold={simple_lookup_threshold}")
                return True
            else:
                logger.debug(f"No simple lookup: simple_lookup_ratio={simple_lookup_ratio}, simple_lookup_threshold={simple_lookup_threshold}")
            return False
        stage1_results = self._inference(stage1_inference, attribute_type, persona_dict)
        for group in stage1_results:
            event_type, attribute = group
            
            # store result
            for oe, value in stage1_results[group]:
                inference_result[(oe.id, attribute)] = value

            # obtain derival functions
            if len(stage1_results[group]) < sample_size:
                logger.debug(f"Continuing for group={group}")
                continue
            elif save_simple_lookup(self.extract_config, [is_simple_lookup(event_dict=oe.to_dict(), key=attribute, target=value) for oe, value in stage1_results[group]]):
                oe, value = stage1_results[group][0]
                derival_function = obtain_derival_function(event_dict=oe.to_dict(), target=value, attribute=attribute)
                logger.debug(f"Derival function derived for group={group} via event={oe}, value={value}.")
                self.simple_lookup_dict[group] = derival_function
            else:
                no_simple_lookup = [(oe.to_dict(), attribute, value) for oe, value in stage1_results[group] if not is_simple_lookup(event_dict=oe.to_dict(), key=attribute, target=value)]
                logger.debug(f"No simple lookup for group={group}: no_simple_lookup[0]={no_simple_lookup[0]}, len(no_simple_lookup)={len(no_simple_lookup)}")
                # Adding the following means that candidate groups are never re-considered...as there might be errors made by the EXTRACT module, this is too restrictive
                # self.simple_lookup_dict[group] = None

        # apply lookups
        stage2_inference = defaultdict(lambda: list())  # list of computations 
        for group, ev_list in waiting_list.items():
            if not self.simple_lookup_dict.get(group) is None:
                event_type, attribute = group
                derival_function = self.simple_lookup_dict[group]
                for oe in ev_list:
                    try:
                        inference_result[(oe.id, attribute)] = derival_function(oe.to_dict())
                    except Exception as e:
                        logger.error(f"Derival function {derival_function} for group={group} failed for oe.to_dict()={oe.to_dict()}. Exception: {e}.")
                        inference_result[(oe.id, attribute)] = None
            else:
                stage2_inference[group] = ev_list

        # stage 2: process remaining inputs
        stage2_results = self._inference(stage2_inference, attribute_type, persona_dict)
        for event_type, attribute in stage2_results:
            for oe, value in stage2_results[(event_type, attribute)]:
                inference_result[(oe.id, attribute)] = value
        return inference_result

    def _inference(self, inference_data: Dict[Tuple[ObservableEventType, str], List[ObservableEvent]], attribute_type: str, persona_dict: Optional[Dict]=None) -> Dict[Tuple[ObservableEventType, str], List[Tuple[ObservableEvent, str]]]:
        """
        Perform computations for the given set of inputs.
        `inference_data` is a dictionary that maps from a group, i.e. a (event_type, attribute) pair, to a list of observable events.
        
        The function output is of a similar structure:
        a dictionary that maps from a group, i.e. a (event_type, attribute) pair, to a list of (observable event, value) pairs.
        Values, as part of the output, are ALL strings at this point.
        """
        result_data = defaultdict(lambda: list())  # maps from group to a list of (event, str) tuples
        if not inference_data:
            return result_data

        outputs = list()
        attributes = set(attr for (_, attr), _ in inference_data.items())
        input_texts = [DatasetExtract.prepare_input_event(attribute=attr, attribute_type=attribute_type, event=ev, persona_dict=persona_dict) for (_, attr), ev_list in inference_data.items() for ev in ev_list]
        batch_size = self.extract_config.inference_batch_size
        num_batches = math.ceil(len(input_texts) / batch_size)
        logger.debug(f"Starting with EXTRACT inference for attributes={attributes}. Number of inputs is {len(input_texts)}, with a batch size of {batch_size}.")
        for batch in tqdm(batchify(input_texts, batch_size=batch_size), total=num_batches):
            outputs += self.extract_model.batch_inference(batch, self.extract_config.generation_params)

        # construct output_dict
        idx = 0
        for group, ev_list in inference_data.items():
            for ev in ev_list:
                result_data[group].append((ev, outputs[idx]))
                idx += 1
        return result_data
    
    def create_computed_events(self, observable_events: List[ObservableEvent], interpretation=str) -> List[ComputedEvent]:
        # prepare inference of EXTRACT module
        computation_inputs = [
            {
                "observable_event": oe,
                "attribute": attr,
                "type": t
            }
            for oe in observable_events
            for attr, t in zip(ComputedEvent.EXTRACT_ATTRIBUTES, ComputedEvent.EXTRACT_TYPES)
        ]
        
        # run inference of EXTRACT module
        computation_results: Dict = self._run(computation_inputs)
        
        # process result and create dicts
        computed_event_dicts = list()
        for oe in observable_events:
            attributes = dict()
            for attr in ComputedEvent.EXTRACT_ATTRIBUTES:
                res = computation_results[(oe.id, attr)]
                attributes[attr] = res
            
            # detect failure
            failure = not attributes["start_date"]
            if failure:
                logger.debug(f"Failure in EXTRACT module for oe={oe.to_dict()}: derived attributes={attributes}.")
                continue
                
            # set missing temporal information (e.g., calendar entries might not have time, purchase might not have end date, etc.)
            original_attributes = attributes.copy()
            if not attributes["start_time"] and not attributes["end_time"]:
                attributes["start_time"] = time(0)
                attributes["end_time"] = time(23, 59)
            if not attributes["start_time"]:
                attributes["start_time"] = time(0)
            if not attributes["end_time"]:
                attributes["end_time"] = attributes["start_time"]
            if not attributes["end_date"]:
                attributes["end_date"] = attributes["start_date"]
            
            # derive start and end datetime for overlap checks
            start_datetime = datetime.combine(attributes["start_date"], attributes["start_time"]).replace(tzinfo=DEFAULT_TZ)
            end_datetime = datetime.combine(attributes["end_date"], attributes["end_time"]).replace(tzinfo=DEFAULT_TZ)
            attributes["start_datetime"] = start_datetime
            attributes["end_datetime"] = end_datetime

            if start_datetime > end_datetime:  # fix for incorrect end_date pred to avoid problems
                new_end_datetime = start_datetime + timedelta(seconds=1)
                logger.warning(f"Detected that start_datetime > end_datetime for start_datetime={start_datetime}, end_datetime={end_datetime}, original_attributes={original_attributes} => new_end_datetime={new_end_datetime}, oe={oe.to_dict()}.")
                end_datetime = new_end_datetime
            
            # remember result
            computed_event_dicts.append({
                "observable_event": oe,
                "attributes": attributes,
                "interpretation": interpretation,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
            })

        # resolve overlaps
        if self.extract_config.get("merge_overlapping_events", False):
            computed_event_dicts = self.resolve_overlapping_events(computed_event_dicts)

        # derive computed events
        computed_events = list()
        for ce_dict in computed_event_dicts:
            if "start_datetime" in ce_dict:
                del(ce_dict["start_datetime"])
            if "end_datetime" in ce_dict:
                del(ce_dict["end_datetime"])
            ce = ComputedEvent(**ce_dict)
            computed_events.append(ce)
        return computed_events

    def load(self) -> None:
        if self.model_loaded:
            return
        self.extract_model = Seq2SeqModel(seq2seq_config=self.extract_config, train=False)
        self.model_loaded = True
        self.extract_model.model.eval()
        if torch.cuda.is_available():
            self.extract_model.model = self.extract_model.model.cuda()

    def resolve_overlapping_events(self, computed_event_dicts: List[Dict]):
        """
        From the list of computed events, remove all overlapping events.
        This ensures that there are not multiple events on the same real life event,
        which could otherwise lead to double-counting.
        """
        # derive list with all datetimes
        events = list()
        for idx, e_dict in enumerate(computed_event_dicts):
            events.append((e_dict["start_datetime"], "start", e_dict["end_datetime"], idx))
            events.append((e_dict["end_datetime"], "vend", e_dict["start_datetime"], idx))

       # sort event dates by datetime, then by type ("start" before "end" for same time)
        events = sorted(events, key=lambda x: (x[0], x[1]))

        # process: idea is to keep track of ongoing events
        ongoing_events = dict()  # track active events: event_idx -> end_dt
        overlaps = set()  # remember overlapping events as tuples
        for dt, dt_type, other_dt, event_idx in events:
            if dt_type == "start":
                for ongoing_event_idx, end_dt in ongoing_events.items():
                    if end_dt == dt: # drop cases in which event1.start_dt == event2.end_dt
                        continue
                    overlaps.add((ongoing_event_idx, event_idx))
                ongoing_events[event_idx] = other_dt  # add current event with end_dt
            else:  # dt_type == "end" 
                del(ongoing_events[event_idx])

        # process overlaps
        processed_indices = set()
        new_computed_event_dicts = list()
        for group in overlaps:
            # merge events
            event_dicts = [computed_event_dicts[e_idx] for e_idx in group]
            
            # log result
            events = [e_dict["observable_event"] for e_dict in event_dicts]
            intervals = [(e_dict["start_datetime"], e_dict["end_datetime"]) for e_dict in event_dicts]
            logger.debug(f"Detected overlapping events: intervals={intervals}, events={events}")
    
            # prefer events with longer duration (e.g., a longer trip to prague, which includes notable events)
            event_dicts = sorted(event_dicts, key=lambda e_dict: e_dict["end_datetime"]-e_dict["start_datetime"], reverse=True)
            
            # prefer structured events for attributes (=temporal information for now)
            str_event_dicts = [e_dict for e_dict, e in zip(event_dicts, events) if e.is_structured()]
            if str_event_dicts:
                attributes = str_event_dicts[0]["attributes"]
            else:
                attributes = event_dicts[0]["attributes"]

            # derive new dict
            new_dict = {
                "observable_events": events,
                "attributes": attributes,
                "interpretation": event_dicts[0]["interpretation"]  # same for all
            }
            new_computed_event_dicts.append(new_dict)
            processed_indices.update(group)

        # process remaining events
        for idx, e_dict in enumerate(computed_event_dicts):
            if idx in processed_indices:
                continue
            new_computed_event_dicts.append(e_dict)
        return new_computed_event_dicts
