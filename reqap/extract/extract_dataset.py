import re
import os
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from omegaconf import DictConfig
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict
from transformers import PreTrainedTokenizer, BatchEncoding

from reqap.library.library import get_persona_names, prepare_persona_dict, load_json, store_json, load_jsonl, store_jsonl, duration_to_s, load_prompt_template
from reqap.qu.operator_tree import OperatorTree
from reqap.llm.icl_model import ICLModel
from reqap.llm.seq2seq_model import DatasetSeq2Seq
from reqap.retrieval.query_execution import QueryExecution
from reqap.classes.observable_event import ObservableEvent, ObservableEventType, METADATA_KEYS


def is_simple_lookup(event_dict: Dict, target: Any, key: str=None, **kwargs):
    # actual computation
    if not key is None: 
        if str(event_dict.get(key)) == target:
            return True
        elif str(event_dict["event_data"].get(key)) == target:
            return True
        else:
            return False
    else:
        if any(str(event_dict[key]) == target for key in event_dict):
            return True
        elif any(str(event_dict["event_data"][key]) == target for key in event_dict["event_data"]):
            return True
        else:
            return False


def obtain_derival_function(event_dict: Dict, target: Any, attribute: str=None, **kwargs) -> Callable | None:
    # prefer direct look-up of attribute
    if attribute:
        if attribute in event_dict and str(event_dict[attribute]) == target:
            logger.debug(f"Obtained derival function for exact attribute: `lambda ev: ev[{attribute}]`.")
            return lambda ev: ev[attribute]
        elif attribute in event_dict["event_data"] and str(event_dict["event_data"][attribute]) == target:
            logger.debug(f"Obtained derival function for exact attribute: `lambda ev: ev['event_data'][{attribute}]`.")
            return lambda ev: ev["event_data"][attribute]

    # otherwise, take first match
    for key in event_dict:
        if str(event_dict[key]) == target:
            logger.debug(f"Obtained derival function `lambda ev: ev[{key}]`.")
            return lambda ev: ev[key]
    for key in event_dict["event_data"]:
        if str(event_dict["event_data"][key]) == target:
            logger.debug(f"Obtained derival function `lambda ev: ev['event_data'][{key}]`.")
            return lambda ev: ev["event_data"][key]
    return None


class DatasetExtract(DatasetSeq2Seq):
    PROMPT = "Attribute: {attribute}, Event:\n{event}"
    PROMPT_WITH_PERSONA = "Attribute: {attribute}\n, Event: {event}\n, Relevant information:\n{persona_data}"
    NONE_VALUE = "null"

    def __init__(self, input_encodings: BatchEncoding, output_encodings: BatchEncoding, dataset_length: int):
        super(DatasetExtract, self).__init__(input_encodings, output_encodings, dataset_length)

    @staticmethod
    def prepare_input_event(attribute: str, attribute_type: str, event: ObservableEvent, persona_dict: Optional[Dict]=None) -> str:
        return DatasetExtract.prepare_input_dict(attribute, attribute_type, event.to_dict(), persona_dict)
    
    @staticmethod
    def prepare_input_dict(attribute: str, attribute_type: str, event: Dict, persona_dict: Optional[Dict]=None) -> str:
        #TODO: integrate attribute type into input
        if "properties_mentioned" in event["event_data"]:
            del(event["event_data"]["properties_mentioned"])

        # copy and remove id
        input_dict = event.copy()
        del(input_dict["id"])

        # verify assumption that data is same
        for key, value in input_dict.items():
            if key in input_dict["event_data"]:
                event_data_value = input_dict["event_data"][key]
                assert value == event_data_value or (value is None and event_data_value.strip() == ""),\
                    f'Assertion failed for {input_dict["event_data"]}, with key={key}, value={value} and value in event_data {input_dict["event_data"][key]}'
        
        # derive one dict with all key, value pairs
        event_data = input_dict["event_data"]
        del(input_dict["event_data"])
        event_dict = {
            **input_dict,
            **event_data
        }
        
        # transform to string
        event_str = ",\n".join(f"{k}: {json.dumps(v)}" for k, v in event_dict.items() if not k in METADATA_KEYS)

        # format prompt
        if not persona_dict is None:
            return DatasetExtract.PROMPT_WITH_PERSONA.format(
                attribute=attribute,
                event=event_str,
                persona_data=prepare_persona_dict(persona_dict)
            )

        return DatasetExtract.PROMPT.format(
            attribute=attribute,
            event=event_str,
        )
    

class DatasetExtractFactory:
    NONE_VALUE = "null"
    NUM_RANDOM_NEGATIVES = 1  # number of randomly selected negatives per instance
    TEMPORAL_ATTRIBUTES = ["start_date", "start_time", "end_date", "end_time"]
    EXTRACT_ATTRIBUTES_PATTERN = re.compile(r'attr_names=\[([^\]]+)\]')
    EXTRACT_ATTRIBUTE_TYPES_PATTERN = re.compile(r'attr_types=\[([^\]]+)\]')

    def __init__(self, extract_config: DictConfig):
        self.extract_config = extract_config

    def create(self, tokenizer: PreTrainedTokenizer, data_input_path: str) -> DatasetExtract:
        # load data
        data = load_jsonl(data_input_path)
        inputs = [it["input"] for it in data]
        outputs = [it["output"] for it in data]

        # encode inputs and outputs
        input_encodings, output_encodings = DatasetSeq2Seq.tokenize(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_input_length=self.extract_config.max_input_length,
            max_output_length=self.extract_config.max_output_length,
        )

        # return result
        dataset_length = len(inputs)
        dataset = DatasetExtract(input_encodings, output_encodings, dataset_length)
        return dataset
    
    def derive_relevant_attributes(self, config: DictConfig, split: str) -> None:
        """
        Derive a set of all relevant attributes from the event data.
        """
        # set paths
        benchmark_dir = config.benchmark.benchmark_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)

        # derive triples: (attribute, event_type, example event)
        attribute_triples_dict = dict()  # nested dict mapping from (attribute, source) to example event
        for persona in tqdm(personas):
            # derive attributes for observable events
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            observable_events_df = pd.read_csv(obs_events_csv_path, converters={"event_data": json.loads, "properties_mentioned": json.loads})
            for _, row in observable_events_df.iterrows():
                event = row.replace(np.nan, None).to_dict().copy()
                event_type = event["event_type"]
                # event_data keys
                for attr in event["event_data"]:
                    if not (attr, event_type) in attribute_triples_dict:
                        attribute_triples_dict[(attr, event_type)] = event
                # temporal keys
                for attr in self.TEMPORAL_ATTRIBUTES:
                    if attr in event and not (attr, event_type) in attribute_triples_dict:
                        attribute_triples_dict[(attr, event_type)] = event

            # derive attributes for structured events
            str_events_csv_path = f"{persona_dir}/{persona}/{persona}_str.csv"
            structured_events_df = pd.read_csv(str_events_csv_path, converters={"event_data": json.loads})
            for _, row in structured_events_df.iterrows():
                event = row.replace(np.nan, None).to_dict().copy()
                event_type = event["event_type"]
                # event_data keys
                for attr in event["event_data"]:
                    if not (attr, event_type) in attribute_triples_dict:
                        attribute_triples_dict[(attr, event_type)] = event
                # temporal keys
                for attr in self.TEMPORAL_ATTRIBUTES:
                    if attr in event and not (attr, event_type) in attribute_triples_dict:
                        attribute_triples_dict[(attr, event_type)] = event

        return attribute_triples_dict
    
    def derive_extract_call_attributes(self, data_input_path: str, data_output_path: str) -> None:
        """
        Derive EXTRACT attributes from actual EXTRACT calls in the QU model.
        """
        # load all Operator Trees
        data = load_jsonl(data_input_path)
        all_operator_trees = [
            operator_tree
            for d in data  # iterate through lines in .jsonl
            for operator_tree_dicts in d.values()  # single dict per row
            for operator_tree in OperatorTree.from_operator_tree_dicts(operator_tree_dicts)  # multiple Operator Trees per instance
        ]

        # derive EXTRACT calls
        #TODO: derive EXTRACT attribute types here
        attributes = set()
        for operator_tree in tqdm(all_operator_trees):
            extract_calls = operator_tree.get_extract_calls()
            for extract_call in extract_calls:
                extract_attribute_clause = self.EXTRACT_ATTRIBUTES_PATTERN.findall(extract_call)
                if not extract_attribute_clause:
                    logger.warning(f'No extract attributes for extract_call=`{extract_call}`.')
                    continue
                extract_attributes_str = extract_attribute_clause[0]
                extract_attributes = extract_attributes_str.split(",")
                extract_attributes = [attr.strip().replace("\"", "") for attr in extract_attributes]
                attributes.update(extract_attributes)
        
        # store data
        store_json(data_output_path, list(attributes))
    
    def derive_attribute_mapping(self, icl_model: ICLModel, data_input_path: str, data_output_path: str) -> None:
        """
        Derive aliases for the different attributes via an LLM.
        """
        # prepare prompts
        attribute_triples = load_json(data_input_path)
        mapping_prompt = load_prompt_template(self.extract_config.extract_training.extract_attribute_mapping_prompt)
        dialogs = list()
        for attr, _, event in attribute_triples:
            prompt = mapping_prompt.render(key=attr, event_str=str(event))
            dialog = [{
                "role": "user",
                "content": prompt
            }]
            dialogs.append(dialog)

        # LLM inference
        attribute_mapping_tuples = list()  # store as list to be able to save as JSON
        results = icl_model.batch_inference(dialogs, sampling_params={"n": 1})
        for (attr, event_source, event), result in zip(attribute_triples, results):
            try:
                result = result.replace("`json", "").replace("`","")
                attribute_aliases = json.loads(result)
                attribute_mapping_tuples.append([attr, event_source, attribute_aliases])
            except Exception as e:
                logger.error(f"Catched exception {e} for result {result}")
        store_json(data_output_path, attribute_mapping_tuples)
    
    def derive_data(self, config: DictConfig, split: str) -> None:
        # load attribute mapping
        attribute_mapping_path = self.extract_config.extract_training.extract_attribute_mapping[split]
        attribute_mapping_tuples = load_json(attribute_mapping_path)
        attribute_mapping = dict()
        for (attr, event_type, attr_aliases) in attribute_mapping_tuples:
            # drop cases in which the type is dropped from the name 
            for t in ["date", "time"]:
                if attr.endswith(t):
                    attr_aliases = [alias for alias in attr_aliases if alias.endswith(t)]
            
            # derive mapping
            attribute_mapping[(attr, event_type)] = attr_aliases

        # derive parallel data
        benchmark_dir = config.benchmark.benchmark_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)
        logger.info(f"Starting with split {split} ({len(personas)} personas)")
        data = list()
        for persona in personas:
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            str_events_csv_path = f"{persona_dir}/{persona}/{persona}_str.csv"
            persona_dict_path = f"{persona_dir}/{persona}/{persona}.json"
            data += self._derive_parallel_data(obs_events_csv_path, str_events_csv_path, persona_dict_path)
            logger.info(f"...done with persona `{persona}`")

        # derive input source distribution
        source_distribution = defaultdict(lambda: 0)  # derive source distribution
        for instance in data:
            event_type = instance["event_dict"]["event_type"]
            source_distribution[event_type] += 1
        logger.debug(f"Input source distribution: {dict(source_distribution)}")

        """
        Derive sets of different complexity for curriculum learning
        """
        max_instances = self.extract_config.extract_training.extract_max_instances[split]
        self.derive_data_copy(data, split, max_instances=max_instances)  # simplest dataset
        self.derive_data_mixed(data, split, max_instances=max_instances)
        
        # sampling based on freq distribution
        logger.info(f"Derived a total of {len(data)} instances. Sampling...")
        logger.debug(f"Input source distribution: {dict(source_distribution)}")
        sample_space = range(len(data))
        sample_space_weights = list()
        for instance in data:
            event_type = instance["event_dict"]["event_type"]
            sample_space_weights.append(1/(source_distribution[event_type] + 1))
        # sample
        sample_indices = random.choices(
            sample_space,
            k=min(len(sample_space), max_instances),
            weights=sample_space_weights
        )
        # dedup
        sample_indices = list(set(sample_indices)) 
        data = [data[idx] for idx in sample_indices]

        # Remaining datasets of increasing complexity
        self.derive_data_simple(data, split)
        self.derive_data_negatives(data, split)
        self.derive_data_aliases(data, attribute_mapping, split)

    def derive_data_copy(self, data: List[Dict], split: str, max_instances: int):
        """ Most simple data, which only requires copying values. """
        filtered_data = list()
        for instance in data:
            if instance["type"] == "positive":
                #  and is_simple_lookup(**instance)
                filtered_data.append(instance)
        data = random.sample(filtered_data, k=min(len(filtered_data), max_instances))
        output_path = self.extract_config.extract_training.extract_data[split].very_simple
        self.store_data(data, output_path, split)

    def derive_data_mixed(self, data: List[Dict], split: str, max_instances: int):
        """ Mixed with positive and negatives, but from easy distribution. """
        data = random.sample(data, k=min(len(data), max_instances))
        output_path = self.extract_config.extract_training.extract_data[split].mixed
        self.store_data(data, output_path, split)

    def derive_data_simple(self, data: List[Dict], split: str):
        """ Data which requires only looking up positive values. """
        filtered_data = list()
        for instance in data:
            if instance["type"] == "positive":
                filtered_data.append(instance)
        output_path = self.extract_config.extract_training.extract_data[split].simple
        self.store_data(filtered_data, output_path, split)

    def derive_data_negatives(self, data: List[Dict], split: str):
        """ Data which requires looking up positive and negative values (no aliases of attributes). """
        output_path = self.extract_config.extract_training.extract_data[split].negative
        self.store_data(data, output_path, split)

    def derive_data_aliases(self, data: List[Dict], attribute_mapping: Dict, split: str):
        """ Full data, including positives and negatives, with aliases. """
        alias_data = list()
        for instance in data:
            attr = instance["key"]
            event_type = instance["event_dict"]["event_type"]
            options = attribute_mapping.get((attr, event_type), []).copy()  # derive aliases
            weights = [1.0] + [0.2] * len(options)  # higher weight for original key
            options = [attr] + options  # prepend original key
            new_attr_name = random.choices(options, weights=weights, k=1)[0]
            instance["key"] = new_attr_name
            alias_data.append(instance)
        output_path = self.extract_config.extract_training.extract_data[split].aliases
        self.store_data(data, output_path, split)

    @classmethod
    def store_data(cls, data: List[Dict], output_path: str, split: str):
        cls.data_analysis(data)
        #TODO: add attribute type here!
        inputs = [DatasetExtract.prepare_input_dict(attribute=it["key"], attribute_type=None, event=it["event_dict"], persona_dict=it.get("persona_dict", None)) for it in data]
        outputs = [str(it["target"]) for it in data]
        data = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        logger.info(f"Derived {split} data with {len(data)} instances")
        store_jsonl(output_path, data)

    @staticmethod
    def data_analysis(data: List[Dict]):
        # analyze data: source distribution
        data_src_dist = defaultdict(lambda: 0)
        for instance in data:
            event_type = instance["event_dict"]["event_type"]
            data_src_dist[event_type] += 1
        logger.debug(f"Data source distribution: {dict(data_src_dist)}")
        
        # analyze data: number of simple look-ups
        simple_lookup_count = 0
        for instance in data:
            if instance["type"] == "positive":
                if is_simple_lookup(**instance):
                    simple_lookup_count += 1
        logger.info(f"Fraction of simple look-ups is {simple_lookup_count} / {len(data)}")

        # analyze data: number of negative cases
        null_count = sum(1 for instance in data if instance["type"] == "negative")
        logger.info(f"Fraction of negative samples is {null_count} / {len(data)}")

    def _derive_parallel_data(self, obs_events_csv_path: str, str_events_csv_path: str, persona_dict_path: str) -> List[Dict]:
        # load data
        observable_events_df = pd.read_csv(obs_events_csv_path, converters={"event_data": json.loads, "properties_mentioned": json.loads})
        structured_events_df = pd.read_csv(str_events_csv_path, converters={"event_data": json.loads})

        # prepare structured events
        se_dict = dict()
        for _, row in structured_events_df.iterrows():
            se_id = row["id"]
            se_dict[se_id] = row

        # identify all relevant properties (RANDOM from any source)
        all_properties = set(
            key
            for _, row in observable_events_df.iterrows()
            for key in se_dict.get(row["structured_event_id"], {}).get("event_data", [])
        ) # keys from structured events
        all_properties.update((
            key
            for _, row in observable_events_df.iterrows()
            for key in row["event_data"]
        )) # keys from observable events

        # identify all relevant properties (from same source)
        properties_per_source = dict()
        for _, row in observable_events_df.iterrows():
            source = row["event_type"]
            keys = set(key for key in se_dict.get(row["structured_event_id"], {}).get("event_data", []))
            keys.update(row["event_data"].keys())
            if not source in properties_per_source:
                properties_per_source[source] = set()
            properties_per_source[source].update(keys)

        # process ALL observable events
        data = list()
        for _, row in observable_events_df.iterrows():
            se_id = row["structured_event_id"]
            if not se_id in se_dict:
                continue
            se = se_dict[se_id]

            # drop properties_mentioned key: only used for deriving mapping
            if "properties_mentioned" in row["event_data"]:
                del(row["event_data"]["properties_mentioned"])

            # clean dict for input
            event = row.replace(np.nan, None).to_dict().copy()
            del(event["properties_mentioned"])
            del(event["structured_event_id"])

            # collect training data for dates
            for key in self.TEMPORAL_ATTRIBUTES:
                if not se[key]:
                    continue
                data.append({
                    "event_dict": event,
                    "key": key,
                    "target": se[key] if se[key] else self.NONE_VALUE,
                    "type": "positive" if se[key] else "negative"
                })

            # sample positives / negatives through properties_mentioned
            negative_keys = list()
            for key in se["event_data"]:
                # event key not relevant
                if self.attribute_irrelevant(key):
                    continue
                if key in row["properties_mentioned"]:
                    target = se["event_data"][key]
                    if key == "duration":
                        target = duration_to_s(target)  # convert duration strings to seconds (could be done in more general manner)
                    elif key == "participants":
                        target = [p["name"] for p in target]
                    data.append({
                        "event_dict": event,
                        "key": key,
                        "target": target,
                        "type": "positive"
                    })
                else:
                    negative_keys.append(key)

            # add keys from unstructured OEs
            t = ObservableEventType(row["event_type"])
            if not t.is_structured():
                for key in row["event_data"]:
                    if self.attribute_irrelevant(key):
                        continue
                    target = row["event_data"][key]
                    data.append({
                        "event_dict": event,
                        "key": key,
                        "target": target,
                        "type": "positive"
                    })

            # enhance negatives with random negative keys (from a different source)
            all_negative_keys = all_properties - set(se["event_data"].keys())
            random_negative_keys = random.sample(
                list(all_negative_keys),
                min(self.extract_config.extract_training.num_random_negatives, len(all_negative_keys))
            )
            negative_keys += random_negative_keys

            # enhance negatives with negative keys from same source)
            all_negative_keys = properties_per_source[row["event_type"]] - set(se["event_data"].keys())
            if all_negative_keys:
                random_negative_keys = random.sample(
                    list(all_negative_keys),
                    min(self.extract_config.extract_training.num_negatives_same_source, len(all_negative_keys))
                )
                negative_keys += random_negative_keys

            # add negative data
            for key in negative_keys:
                data.append({
                    "event_dict": event,
                    "key": key,
                    "target": self.NONE_VALUE,
                    "type": "negative"
                })

        # potentially add persona dict to all dicts
        if self.extract_config.extract_training.extract_add_persona:
            persona_dict = load_json(persona_dict_path)
            for instance in data: 
                instance["persona_dict"] = persona_dict
        else:
            logger.warning("Not using persona dicts in input!")

        return data

    def derive_extract_queries(self, question_path: str) -> List[Dict]:
        #NOT IN USE
        retrieval_queries = list()
        data = load_json(question_path)
        for instance in data:
            query = instance["sql_query"]
            if QueryExecution.is_simple_query(query):
                retrieval_query = QueryExecution.derive_extract_query(query)
                retrieval_queries.append(retrieval_query)
        return retrieval_queries
    
    @staticmethod
    def attribute_irrelevant(attribute_key: str):
        BLACKLIST = {"event", "notable_events"}
        if attribute_key in BLACKLIST:
            return True
        # elif attribute_key.strip().endswith("unit"):
            # return True
        return False
    
