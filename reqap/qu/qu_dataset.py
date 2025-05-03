from tqdm import tqdm
from loguru import logger
from typing import List, Dict
from omegaconf import DictConfig
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from reqap.qu.operator_tree import OperatorTree
from reqap.llm.seq2seq_model import DatasetSeq2Seq
from reqap.llm.causal_model import DatasetCausalModel
from reqap.library.library import load_jsonl, store_jsonl, pairwise, load_txt


def represent_input_seq2seq(dialog: List[Dict] | List[str], tokenizer: AutoTokenizer, max_length: int) -> str:
    def get_content(turn: Dict|str):
        return turn["content"] if isinstance(turn, dict) else turn
    def length_overflow(input_str: str, tokenizer: AutoTokenizer, max_length: int) -> bool:
        return len(tokenizer.encode(input_str)) > max_length
    input_str = ""
    
    # process history
    HISTORY_TEMPLATE = "{qu_input}=> {qu_output}\n"
    for qu_input, qu_output in pairwise(dialog[:-1]):
        qu_input = get_content(qu_input)
        qu_input = qu_input.replace("Input: ", "").replace("Starting with new question.", "").strip()
        qu_output = get_content(qu_output)
        input_str += HISTORY_TEMPLATE.format(
            qu_input=qu_input,
            qu_output=qu_output
        )

    # add current turn
    current_turn = dialog[-1]
    current_turn = get_content(current_turn)
    qu_input = current_turn.replace("Input: ", "").replace("Starting with new question.", "").strip()
    input_str += HISTORY_TEMPLATE.format(
        qu_input=qu_input,
        qu_output=""
    )

    # if overflows length, drop from the beginning and try again
    if length_overflow(input_str, tokenizer, max_length):
        logger.warning(f"Length overflow with max_length={max_length}, len(dialog)={len(dialog)}, dialog={dialog}")
        return represent_input_seq2seq(dialog[2:], tokenizer, max_length)

    return input_str


def represent_input_causal(dialog: List[Dict] | List[str], tokenizer: AutoTokenizer, max_length: int) -> str:
    def length_overflow(input_str: str, tokenizer: AutoTokenizer, max_length: int) -> bool:
        return len(tokenizer.encode(input_str)) > max_length
    # apply chat template
    input_str = tokenizer.apply_chat_template(
        dialog,
        tokenize=False,
        add_generation_prompt=False,
    )

    # if overflows length, drop first turns (but not the instruction!) and try again
    if length_overflow(input_str, tokenizer, max_length):
        logger.warning(f"Length overflow with max_length={max_length}, len(dialog)={len(dialog)}, dialog={dialog}")
        return represent_input_seq2seq([dialog[0]] + dialog[3:], tokenizer, max_length)  # drop initial part
    return input_str


class DatasetQuestionUnderstandingFactory:
    CAUSAL_LM_INPUT_PROMPT = "Input: {qu_input}"
    CAUSAL_LM_OUTPUT_PROMPT = "{operator_tree}"

    def __init__(self, qu_config: DictConfig, causal: bool=False):
        self.qu_config = qu_config
        self.causal = causal
        if causal:
            self.instruction = load_txt(self.qu_config.qu_causal_instr)

    def create(self, tokenizer: AutoTokenizer, split: str) -> Dataset:
        if self.causal:
            return self._create_causal(tokenizer, split)
        else:
            return self._create_seq2seq(tokenizer, split)
    
    def _create_seq2seq(self, tokenizer: AutoTokenizer, split: str) -> DatasetSeq2Seq:
        input_path = self.qu_config.qu_data[split]
        
        # load data
        data = load_jsonl(input_path)
        inputs = list()
        for it in data:
            l = [c for turn in it["input"] for c in turn][:-1]
            inputs.append(represent_input_seq2seq(dialog=l, tokenizer=tokenizer, max_length=self.qu_config.max_input_length))
        outputs = [it["output"] for it in data]

        input_encodings, output_encodings = DatasetSeq2Seq.tokenize(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_input_length=self.qu_config.max_input_length,
            max_output_length=self.qu_config.max_output_length,
        )

        # construct and return dataset
        dataset = DatasetSeq2Seq(
            input_encodings,
            output_encodings,
            len(inputs)
        )
        return dataset

    def _create_causal(self, tokenizer: AutoTokenizer, split: str) -> DatasetCausalModel:
        if self.qu_config.get("one_shot", False):
            return self._create_causal_one_shot(tokenizer=tokenizer, split=split)

        input_path = self.qu_config.qu_data[split]
        
        # load data
        data = load_jsonl(input_path)
        inputs = list()
        for it in data:
            dialog = [{"role": "system", "content": self.instruction}]
            for turn in it["input"]:
                i, o = turn
                dialog.append({
                    "role": "user",
                    "content": self.CAUSAL_LM_INPUT_PROMPT.format(qu_input=i)
                })
                dialog.append({
                    "role": "assistant",
                    "content": self.CAUSAL_LM_OUTPUT_PROMPT.format(operator_tree=o)
                })
            dialog = dialog[:-1]  # drop expected output
            input_txt = represent_input_causal(dialog=dialog, tokenizer=tokenizer, max_length=self.qu_config.max_input_length)
            inputs.append(input_txt)
        outputs = [
            self.CAUSAL_LM_OUTPUT_PROMPT.format(operator_tree=it["output"])
            for it in data
        ]

        input_encodings, labels = DatasetCausalModel.prepare_encodings_train(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_length=self.qu_config.max_length
        )

        # construct and return dataset
        dataset = DatasetCausalModel(
            input_encodings,
            labels,
            len(inputs)
        )
        return dataset
    
    def _create_causal_one_shot(self, tokenizer: AutoTokenizer, split: str) -> DatasetCausalModel:
        input_path = self.qu_config.qu_data[split]
        
        # load data
        data = load_jsonl(input_path)
        inputs = list()
        for it in data:
            dialog = [{"role": "system", "content": self.instruction}]
            dialog.append({
                "role": "user",
                "content": self.CAUSAL_LM_INPUT_PROMPT.format(qu_input=it["input"])
            })
            input_txt = tokenizer.apply_chat_template(
                dialog,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs.append(input_txt)
        outputs = [
            self.CAUSAL_LM_OUTPUT_PROMPT.format(operator_tree=it["output"])
            for it in data
        ]

        input_encodings, labels = DatasetCausalModel.prepare_encodings_train(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_length=self.qu_config.max_length
        )

        # construct and return dataset
        dataset = DatasetCausalModel(
            input_encodings,
            labels,
            len(inputs)
        )
        return dataset

    def derive_data(self, split: str):
        # do NOT make use of training loop, but simply distill ALL data of larger model
        if self.qu_config.qu_training.get("qu_distilled", False):
            return self.derive_data_distill_only(split=split)
        elif self.qu_config.get("one_shot", False):
            return self.derive_data_one_shot(split=split)
        
        # load result from Operator Tree runs
        qu_result_path = self.qu_config.qu_training.qu_result_data[split]
        output_path = self.qu_config.qu_training.qu_data[split]
        
        # run ICL model to derive target
        inputs = list()
        outputs = list()
        data = load_jsonl(qu_result_path)
        logger.debug(f"Processing data...")
        for instance in tqdm(data):
            # derive correct plans (or relaxed correct ones)
            correct_plans = [r["operator_tree"] for r in instance["results"] if r["hit_at_1"]]
            relaxed_correct_plans_10 = [r["operator_tree"] for r in instance["results"] if r["relaxed_hit_at_1_10"]]
            relaxed_correct_plans_20 = [r["operator_tree"] for r in instance["results"] if r["relaxed_hit_at_1_20"]]
            running_plans = [r["operator_tree"] for r in instance["results"] if not r["failed"]]

            # drop none-answers
            if instance["answers"] in [None, [0.0], [0]]:
                continue

            # two options: keep correct plans only, or also leverage running (if none correct)
            if self.qu_config.qu_training.qu_correct_only:
                if correct_plans:
                    operator_tree_dicts = correct_plans
                elif relaxed_correct_plans_10:
                    operator_tree_dicts = relaxed_correct_plans_10
                else:
                    operator_tree_dicts = relaxed_correct_plans_20
            else:
                if correct_plans:
                    operator_tree_dicts = correct_plans
                elif relaxed_correct_plans_10:
                    operator_tree_dicts = relaxed_correct_plans_10
                elif relaxed_correct_plans_20:
                    operator_tree_dicts = relaxed_correct_plans_20
                else:
                    operator_tree_dicts = running_plans

            # derive input/output pairs
            operator_trees = OperatorTree.from_operator_tree_dicts(operator_tree_dicts)
            for operator_tree in operator_trees:
                question_inputs, question_outputs = operator_tree.derive_training_data()
                inputs += question_inputs
                outputs += question_outputs
        logger.debug(f"Derived {len(inputs)} I/O pairs.")

        # dedup
        unique_io_pairs = list()
        for i, o in zip(inputs, outputs):
            io_pair = (i, o)
            if not io_pair in unique_io_pairs:
                unique_io_pairs.append(io_pair)
        inputs = [i for i, _ in unique_io_pairs]
        outputs = [o for _, o in unique_io_pairs]
        logger.debug(f"Derived {len(inputs)} I/O pairs after dedup.")
        
        # store data (tokenization independent)
        data = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        store_jsonl(output_path, data)

    """
    Ablation studies.
    """
    def derive_data_distill_only(self, split: str):
        """
        Ablation: This variant is used for deriving data for all Operator Trees created by the supervisor.
        """
        # load result from Operator Tree runs
        qu_result_path = self.qu_config.qu_training.qu_result_paths[split]
        output_path = self.qu_config.qu_training.qu_data[split]
        operator_tree_mapping = load_jsonl(qu_result_path)
        question_to_operator_tree = {key: value for mapping in operator_tree_mapping for key, value in mapping.items()}

        # run ICL model to derive target
        inputs = list()
        outputs = list()
        logger.debug(f"Processing data...")
        for operator_tree_dicts in question_to_operator_tree.values():
            # derive input/output pairs
            operator_trees = OperatorTree.from_operator_tree_dicts(operator_tree_dicts)
            for operator_tree in operator_trees:
                question_inputs, question_outputs = operator_tree.derive_training_data()
                inputs += question_inputs
                outputs += question_outputs
        logger.debug(f"Derived {len(inputs)} I/O pairs.")

        # dedup
        unique_io_pairs = list()
        for i, o in zip(inputs, outputs):
            io_pair = (i, o)
            if not io_pair in unique_io_pairs:
                unique_io_pairs.append(io_pair)
        inputs = [i for i, _ in unique_io_pairs]
        outputs = [o for _, o in unique_io_pairs]
        logger.debug(f"Derived {len(inputs)} I/O pairs after dedup.")
        
        # store data (tokenization independent)
        data = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        store_jsonl(output_path, data)

    def derive_data_one_shot(self, split: str):
        """
        Ablation: This variant derives one-shot question, full Operator Tree pairs.
        """
        # load result from Operator Tree runs
        qu_result_path = self.qu_config.qu_training.qu_result_data[split]
        output_path = self.qu_config.qu_training.qu_data[split]
        
        # run ICL model to derive target
        inputs = list()
        outputs = list()
        data = load_jsonl(qu_result_path)
        logger.debug(f"Processing data...")
        for instance in tqdm(data):
            question = instance["question"]
            # derive correct plans (or relaxed correct ones)
            correct_plans = [r["operator_tree"] for r in instance["results"] if r["hit_at_1"]]
            relaxed_correct_plans_10 = [r["operator_tree"] for r in instance["results"] if r["relaxed_hit_at_1_10"]]
            relaxed_correct_plans_20 = [r["operator_tree"] for r in instance["results"] if r["relaxed_hit_at_1_20"]]
            running_plans = [r["operator_tree"] for r in instance["results"] if not r["failed"]]

            # drop none-answers
            if instance["answers"] in [None, [0.0], [0]]:
                continue

            if correct_plans:
                operator_tree_dicts = correct_plans
            elif relaxed_correct_plans_10:
                operator_tree_dicts = relaxed_correct_plans_10
            elif relaxed_correct_plans_20:
                operator_tree_dicts = relaxed_correct_plans_20
            else:
                operator_tree_dicts = running_plans
            # derive input/output pairs
            for operator_tree_dict in operator_tree_dicts:
                operator_tree = OperatorTree.derive_full_operator_tree(operator_tree_dict)
                inputs.append(question)
                outputs.append(operator_tree)

        # dedup
        unique_io_pairs = list()
        for i, o in zip(inputs, outputs):
            io_pair = (i, o)
            if not io_pair in unique_io_pairs:
                unique_io_pairs.append(io_pair)
        inputs = [i for i, _ in unique_io_pairs]
        outputs = [o for _, o in unique_io_pairs]
        logger.debug(f"Derived {len(inputs)} I/O pairs after dedup.")
        
        # store data (tokenization independent)
        data = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        store_jsonl(output_path, data)
