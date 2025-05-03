import re
import os
import sys
import json
import torch
import random
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoTokenizer
from typing import List, Dict
from tqdm import tqdm

from reqap.library.library import load_config, load_json, load_jsonl, store_jsonl, get_persona_names, load_prompt_template, handle_output_file, load_txt, clear_file, set_random_seed
from reqap.llm.crossencoder_model import CrossEncoderModel
from reqap.llm.causal_model import DatasetCausalModel, CausalModel
from reqap.retrieval.retrieval import Retrieval
set_random_seed()


class Rag:
    ICL_PROMPT_TEMPLATE = (
        "Question: {question}"
    )

    OUTPUT_TEMPLATE = "{answers}"

    def __init__(self, config: DictConfig):
        self.config = config
        self.answering_instruction = load_txt(self.config.answering.instruction)
        self.answering_prompt = load_prompt_template(config.answering.prompt)
        self.icl_examples = []
        self.answering_loaded = False

    def run(self, split="test", override: bool=False):
        # init
        self.load_answering()
        benchmark_dir = self.config.benchmark.benchmark_dir
        result_dir = self.config.benchmark.result_dir
        retrieval_result_dir = self.config.crossencoder.retrieval_result_dir
        splade_indices_dir = self.config.splade.splade_indices_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)
        max_num_events = self.config.answering.max_num_events

        # iterate through personas
        for i, persona in enumerate(personas):
            # init paths for persona
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            splade_index_path = f"{splade_indices_dir}/{persona}.splade_index"
            if os.path.exists(f"{retrieval_result_dir}/{persona}/retrieval_result.jsonl"):
                questions_path = f"{retrieval_result_dir}/{persona}/retrieval_result.jsonl"
                data = load_jsonl(questions_path)
            else:
                questions_path = f"{persona_dir}/{persona}/questions.json"
                data = load_json(questions_path)
            output_path = f"{result_dir}/{persona}/result.jsonl"
            data = handle_output_file(output_path, data, override)

            # init retrieval
            retrieval = Retrieval(self.config, obs_events_csv_path, splade_index_path)

            # process questions
            for j, instance in enumerate(data):
                logger.debug(f"Starting with inference for question {j} / {len(data)}, persona {i} / {len(personas)}")
                question = instance["question"]
                
                # retrieve events
                if "retrieval_result" in instance:
                    event_dicts = instance["retrieval_result"]
                else:
                    positive_events = retrieval.retrieve(query=question, ordered=True)
                    events = positive_events
                    event_dicts = [event.to_dict() for event in events]
                    instance["num_relevant_events"] = len(event_dicts)
                trimmed_event_dicts = event_dicts[:max_num_events] # restrict LLM input
                trimmed_event_dicts = [self.remove_metadata_from_event_dict(d) for d in trimmed_event_dicts]  # drop metadata

                # construct prompt
                prompt = self.answering_prompt.render(
                    question=question,
                    input_data=json.dumps(trimmed_event_dicts)
                )
                if type(self.answering_model) == CausalModel:
                    max_prompt_length = self.config.answering.max_input_length
                    prompt = self.trim_prompt(prompt, self.answering_model.tokenizer, max_prompt_length)
                dialog = self.construct_dialog(prompt, use_icl_examples=(self.config.answering.answering_mode!="causal"))
                instance["num_events"] = len(trimmed_event_dicts)

                # adjust dialog for CausalModel
                if type(self.answering_model) == CausalModel:
                    dialog = self.sft_adjust_dialog(dialog, tokenizer=self.answering_model.tokenizer)  # returns a str, which is also expected by CausalModel.inference
                    sampling_params = self.config.answering.sampling_params
                else:
                    sampling_params = {"n": 1, "temperature": 0.0, "max_tokens": 20}

                answer = self.answering_model.inference(dialog, sampling_params=sampling_params)
                instance["derived_answer"] = answer
                store_jsonl(output_path, [instance], file_mode="a")

    def construct_dialog(self, prompt: str, use_icl_examples: bool=True) -> List[Dict]:
        dialog = [{
            "role": "system",
            "content": self.answering_instruction,
        }]
        if use_icl_examples:
            for ex in self.icl_examples:
                dialog.append({
                    "role": "user",
                    "content": self.ICL_PROMPT_TEMPLATE.format(**ex),
                })
                dialog.append({
                    "role": "assistant",
                    "content": self.OUTPUT_TEMPLATE.format(**ex),
            })
        dialog += [{
            "role": "user",
            "content": prompt,
        }]
        return dialog

    def load_answering(self) -> None:
        """
        Initiate the answering module.
        """
        if self.answering_loaded:
            return
        answering_mode = self.config.answering.answering_mode
        if answering_mode == "openai":
            from reqap.llm.openai import OpenAIModel
            answering_model = OpenAIModel(openai_config=self.config.openai, use_cache=self.config.openai.use_cache)
            self.icl_examples = load_json(self.config.answering.icl_examples)
        elif answering_mode == "instruct_model":
            from reqap.llm.instruct_model import InstructModel
            answering_model = InstructModel(instruct_config=self.config.instruct_model, use_cache=self.config.instruct_model.use_cache)
            self.icl_examples = load_json(self.config.answering.icl_examples)
        elif answering_mode == "causal":
            from reqap.llm.causal_model import CausalModel
            answering_model = CausalModel(causal_config=self.config.answering, train=False)
            self.icl_examples = load_json(self.config.answering.icl_examples)
        else:
            raise NotImplementedError(f"Answering mode {answering_mode} is not implemented!")
        self.answering_model = answering_model
        self.answering_loaded = True
    
    @staticmethod
    def remove_metadata_from_event_dict(event_dict: dict) -> dict:
        BLACKLIST_KEYS = {"derived_via", "splade_score", "ce_scores"}
        event_dict["event_data"] = {k: v for k, v in event_dict["event_data"].items() if not k in BLACKLIST_KEYS}
        return event_dict

    """
    Retrieval inference.
    """
    # init
    def retrieve(self, split="test", override: bool=False, retrieve_persona: str=None):
        logger.info(f"Starting retrieval on split={split}, retrieve_persona={retrieve_persona}")
        benchmark_dir = self.config.benchmark.benchmark_dir
        retrieval_result_dir = self.config.crossencoder.retrieval_result_dir
        splade_indices_dir = self.config.splade.splade_indices_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)

        # iterate through personas
        for _, persona in enumerate(personas):
            if not retrieve_persona is None and persona != retrieve_persona:
                continue
            # init paths for persona
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            splade_index_path = f"{splade_indices_dir}/{persona}.splade_index"
            questions_path = f"{persona_dir}/{persona}/questions.json"
            output_path = f"{retrieval_result_dir}/{persona}/retrieval_result.jsonl"
            data = load_json(questions_path)
            data = handle_output_file(output_path, data, override)

            # init retrieval
            retrieval = Retrieval(self.config, obs_events_csv_path, splade_index_path)

            # process questions
            for _, instance in tqdm(enumerate(data)):
                question = instance["question"]
                positive_events = retrieval.retrieve(query=question, ordered=True)
                max_num_events = self.config.crossencoder.max_num_events
                instance["num_relevant_events"] = len(positive_events)
                events = positive_events[:max_num_events]
                event_dicts = [event.to_dict() for event in events]
                instance["retrieval_result"] = event_dicts
                store_jsonl(output_path, [instance], file_mode="a")

    """
    LLM training.
    """
    def trim_prompt(self, prompt, tokenizer, max_prompt_length) -> str:
        tokenized_prompt = tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")["input_ids"][0]
        if len(tokenized_prompt) <= max_prompt_length:
            return prompt
        logger.warning(f"Trimming prompt here with len(prompt)={len(prompt)} and max_prompt_length={max_prompt_length}.")
        trimmed_prompt_tokens = tokenized_prompt[:max_prompt_length]
        trimmed_prompt = tokenizer.decode(trimmed_prompt_tokens, skip_special_tokens=True)
        return trimmed_prompt
 
    def derive_data(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.answering.model)
        self.derive_data_split(tokenizer=tokenizer, split="train")
        self.derive_data_split(tokenizer=tokenizer, split="dev")

    def derive_data_split(self, tokenizer: AutoTokenizer, split: str):
        benchmark_dir = self.config.benchmark.benchmark_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)
        max_prompt_length = self.config.answering.max_input_length
        retrieval_result_dir = self.config.crossencoder.retrieval_result_dir
        output_path = self.config.answering.training[split]
        clear_file(output_path)
        max_num_events = self.config.answering.max_num_events

        # iterate through personas
        for i, persona in enumerate(personas):
            # init paths for persona
            questions_path = f"{retrieval_result_dir}/{persona}/retrieval_result.jsonl"
            data = load_jsonl(questions_path)

            # process questions
            for _, instance in tqdm(enumerate(data)):
                # skip "none" answers => led to shortcut learning in always predicting 0
                if instance["answers"] in [None, [0.0], [0]]:
                    continue
                
                # construct prompt
                question = instance["question"]
                event_dicts = instance["retrieval_result"]
                trimmed_event_dicts = event_dicts[:max_num_events] # restrict LLM input
                trimmed_event_dicts = [self.remove_metadata_from_event_dict(d) for d in trimmed_event_dicts]  # drop metadata
                prompt = self.answering_prompt.render(
                    question=question,
                    input_data=json.dumps(trimmed_event_dicts)
                )
                prompt = self.trim_prompt(prompt, tokenizer, max_prompt_length)
                dialog = self.construct_dialog(prompt, use_icl_examples=False)
                instance["num_events"] = len(trimmed_event_dicts)
                input_str = self.sft_adjust_dialog(dialog, tokenizer)

                # store instance
                instance = {
                    "input": input_str,
                    "output": instance["answers"]
                }
                store_jsonl(output_path, [instance], file_mode="a")

    def train(self):
        # derive datasets
        model = CausalModel(causal_config=self.config.answering, train=True)
        dev_set = self.create_dataset(model.tokenizer, split="dev")
        logger.info(f"Loaded dev set with {len(dev_set)} instances.")
        train_set = self.create_dataset(model.tokenizer, split="train")
        logger.info(f"Loaded train set with {len(train_set)} instances.")
        
        # train model
        model.train(train_set, dev_set)
        model.save()

    def create_dataset(self, tokenizer: AutoTokenizer, split: str) -> DatasetCausalModel:
        # load data
        data_path = self.config.answering.training[split]
        inputs = list()
        outputs = list()
        logger.debug(f"Loading data from {data_path}...")
        with open(data_path, "r") as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                inputs.append(instance["input"])
                outputs.append(str(instance["output"]))
        
        # limit num of instances (e.g., for dev)
        if self.config.answering.get("max_training_instances", False):
            max_training_instances = self.config.answering.max_training_instances
            inputs = inputs[:max_training_instances]
            outputs = outputs[:max_training_instances]

        # tokenize
        input_encodings, labels = DatasetCausalModel.prepare_encodings_train(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_length=self.config.answering.max_length
        )

        # construct and return dataset
        dataset = DatasetCausalModel(
            input_encodings,
            labels,
            len(inputs)
        )
        return dataset

    def sft_adjust_dialog(self, dialog: List[Dict], tokenizer: AutoTokenizer, max_chars: int = 10000) -> str:
        input_str = tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=False,
        )
        return input_str

    """
    Cross-encoder training.
    """
    @staticmethod
    def ce_train(config: DictConfig):
        from reqap.retrieval.crossencoder.crossencoder_dataset import DatasetCrossEncoderFactory
        ce_config = config.crossencoder
        dataset_fac = DatasetCrossEncoderFactory(config=config, ce_config=ce_config)
        ce_model = CrossEncoderModel(ce_config=ce_config, train=True)

        # derive train set
        train_set = dataset_fac.create(ce_model.tokenizer, ce_config.crossencoder_train_data, input_type="event")
        logger.info(f"Derived train set with {len(train_set)} instances.")
        torch.save(train_set, ce_config.crossencoder_train_dataset)

        # derive dev set
        dev_set = dataset_fac.create(ce_model.tokenizer, ce_config.crossencoder_dev_data, input_type="event")
        torch.save(dev_set, ce_config.crossencoder_dev_dataset)
        logger.info(f"Derived dev set with {len(dev_set)} instances.")

        # train
        ce_model.train(train_set, dev_set)
        ce_model.save()
        logger.info(f"Done with training.")

    @staticmethod
    def ce_derive_data(config: DictConfig):
        from reqap.retrieval.crossencoder.crossencoder_dataset import DatasetCrossEncoderFactory
        def _transform_retrieve_call(retrieve_call: str) -> List[str]:
            ret_queries = DatasetCrossEncoderFactory.QUERY_PATTERN.findall(retrieve_call)
            return ret_queries
        def _derive_data(config: DictConfig, retrieve_calls_path: str, data_path: str, output_path: str):
            # prepare mapping from questions to queries
            retrieve_calls_list = load_jsonl(retrieve_calls_path)
            questions_to_queries = {d["question"]: _transform_retrieve_call(d["retrieve_call"]) for d in retrieve_calls_list}

            # prepare mapping from queries to training data
            data = load_jsonl(data_path)
            query_to_data = dict()
            for instance in tqdm(data):
                if not instance["input_type"] == "event":
                    continue
                retrieve_query, event = instance["input"]
                instance["event"] = event
                if not retrieve_query in query_to_data:
                    query_to_data[retrieve_query] = list()
                query_to_data[retrieve_query].append(instance)

            # iterate through questions and derive data
            data = list()
            for question, queries in tqdm(questions_to_queries.items()):
                for query in queries:
                    query_data = query_to_data.get(query, [])
                    instances_for_query = list()
                    for instance in query_data:
                        inst = instance.copy()
                        inst["input"] = [question, inst["event"]]
                        del(inst["event"])
                        instances_for_query.append(inst)
                    num_instances = min(len(instances_for_query), config.crossencoder.max_instances_per_query)
                    data += random.sample(instances_for_query, num_instances)
            logger.info(f"Derived set with len(data)={len(data)} instances")
            store_jsonl(output_path, data)

        logger.info("Transforming train data...")
        _derive_data(
            config=config,
            retrieve_calls_path=config.crossencoder.retrieve_calls_train_set,
            data_path=config.crossencoder.input_train_data,
            output_path=config.crossencoder.crossencoder_train_data
        )
        logger.info("Done with transforming train data.")

        logger.info("Transforming dev data...")
        _derive_data(
            config=config,
            retrieve_calls_path=config.crossencoder.retrieve_calls_dev_set,
            data_path=config.crossencoder.input_dev_data,
            output_path=config.crossencoder.crossencoder_dev_data
        )
        logger.info("Done with transforming dev data.")


def main():
    # check if provided options are valid
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python rag.py <FUNCTION> [<CONFIG>]"
        )
    config_path = "config/perqa/rag_openai.yml" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # run
    function = sys.argv[1]
    if function.startswith("--dev"):
        rag = Rag(config)
        rag.run(split="dev")
    elif function.startswith("--test"):
        rag = Rag(config)
        rag.run(split="test")
    elif function.startswith("--ce_derive_data"):
        Rag.ce_derive_data(config)
    elif function.startswith("--ce_train"):
        Rag.ce_train(config)
    elif function.startswith("--derive_data"):
        rag = Rag(config)
        rag.derive_data()
    elif function.startswith("--train"):
        rag = Rag(config)
        rag.train()
    elif function.startswith("--retrieve"):
        rag = Rag(config)
        split = "test" if len(sys.argv) < 4 else sys.argv[3]
        retrieve_persona = None if len(sys.argv) < 5 else sys.argv[4]
        rag.retrieve(
            split=split,
            retrieve_persona=retrieve_persona
        )
    else:
        raise Exception(f"Unknown function {function}.")


if __name__ == "__main__":
    main()