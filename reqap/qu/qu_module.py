import re
import os
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from omegaconf import DictConfig
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Dataset

from reqap.library.library import load_txt, load_json, load_jsonl, bm25_scoring, tokenize_code
from reqap.llm.icl_model import ICLModel
from reqap.llm.seq2seq_model import Seq2SeqModel
from reqap.qu.qu_supervisor import QUSupervisor
from reqap.qu.qu_dataset import represent_input_seq2seq, represent_input_causal
from reqap.qu.qu_model import QUModel, QUModelCausal, compute_metrics_qu
from reqap.qu.operator_tree import OperatorTree
from reqap.qu.qu_tree import QUTree, QUTreeBranch


class QuestionUnderstandingModule:
    NEW_QUESTION_PROMPT_TEMPLATE = (
        "Starting with new question.\n\nInput: {qu_input}"
    )
    
    PROMPT_TEMPLATE = (
        "Input: {qu_input}"
    )

    OUTPUT_TEMPLATE = "{operator_tree}"

    QU_INPUT_PATTERN = re.compile("{{ QU(.*?) }}")
    MAX_RECURSION_DEPTH = 8
    MAX_ICL_EXAMPLES = 20

    def __init__(self, qu_config: DictConfig, train: bool=False, icl_model: Optional[ICLModel]=None, use_cache: Optional[bool]=True):
        self.qu_config = qu_config
        self.model_loaded = False
        self.causal_model = self.qu_config.get("causal", False)
        if not train:
            self.load(icl_model, use_cache)

    def derive_training_data(self) -> None:
        from reqap.qu.qu_dataset import DatasetQuestionUnderstandingFactory

        # derive data
        qu_dataset_fac = DatasetQuestionUnderstandingFactory(self.qu_config, causal=self.causal_model)
        qu_dataset_fac.derive_data("train")
        qu_dataset_fac.derive_data("dev")

    def train(self) -> None:
        from reqap.qu.qu_dataset import DatasetQuestionUnderstandingFactory

        qu_dataset_fac = DatasetQuestionUnderstandingFactory(self.qu_config, causal=self.causal_model)

        if self.causal_model:
            self.model = QUModelCausal(self.qu_config, train=True)
        else:
            self.model = QUModel(self.qu_config, train=True)
        train_path = self.qu_config.qu_data.train
        train_set = qu_dataset_fac.create(self.model.tokenizer, "train")
        logger.info(f"Loaded train set with {len(train_set)} instances from {train_path}.")
        logger.debug(f"train_set[0]={train_set[0]}")


        dev_path = self.qu_config.qu_data.dev
        dev_set = qu_dataset_fac.create(self.model.tokenizer, "dev")
        logger.info(f"Loaded dev set with {len(dev_set)} instances from {dev_path}.")
        logger.debug(f"dev_set[0]={dev_set[0]}")

        self.model.train(train_set, dev_set, compute_metrics_fct=compute_metrics_qu)
        self.model.save()
        self.model_loaded = True
        self.model.model.eval()
        if not self.causal_model:  # evaluation does not work for causal model, currently
            self.evaluate()

    def evaluate(self, qu_config: DictConfig) -> None:
        # load model
        self.load()            
        from reqap.qu.qu_dataset import DatasetQuestionUnderstandingFactory
        
        # init
        dataset_fac = DatasetQuestionUnderstandingFactory(qu_config, causal=self.causal_model)
        tokenizer = self.model.tokenizer
        dev_set = dataset_fac.create(tokenizer, "dev")
        logger.info(f"Loaded dev set with {len(dev_set)} instances.")

        # eval
        metrics = self._evaluate(
            self.model,
            dev_set,
            batch_size=qu_config.training_params.per_device_eval_batch_size,
            max_length=qu_config.training_params.generation_max_length,
            generation_params=qu_config.generation_params
        )
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
        metrics = compute_metrics_qu(
            eval_preds=(all_preds, all_labels),
            tokenizer=model.tokenizer
        )
        return metrics

    def load(self, icl_model: Optional[ICLModel]=None, use_cache: Optional[bool]=True) -> None:
        """ Initialize model (trained or ICL model). """
        if not self.model_loaded:
            if icl_model is None:
                # load trained model
                if self.causal_model:
                    self.instruction = load_txt(self.qu_config.qu_causal_instr)
                    self.model = QUModelCausal(self.qu_config, train=False, use_cache=use_cache)
                    self.icl_examples = []
                else:
                    self.instruction = None
                    self.model = QUModel(self.qu_config, train=False, use_cache=use_cache)
                    self.icl_examples = []
                self.model_loaded = True
                self.model.model.eval()
                if torch.cuda.is_available():
                    self.model.model = self.model.model.cuda()
            else:
                # init ICL-based model
                self.model = QUSupervisor(icl_model)
                self.instruction = load_txt(self.qu_config.qu_supervisor_instr)
                icl_examples_grouped = load_json(self.qu_config.qu_supervisor_icl)
                self.icl_examples = [example_group for example_group in icl_examples_grouped]

    def add_to_history(self, history: List[Dict], text: str, role: str):
        """
        Add text to history (which is in dialog format and can be used to init a dialog).
        """
        history.append({
            "role": role,
            "content": text,
        })
        return history
    
    def start_history(self, qu_input: str) -> List[Dict]:
        return [{
            "role": "user",
            "content": self.NEW_QUESTION_PROMPT_TEMPLATE.format(qu_input=qu_input)
        }]
    
    def create_dialog(self, history: List[Dict]) -> List[Dict]:
        """
        For the given interaction history (QU inputs and outputs),
        create a dialog to be used in the LLM generation.
        """
        dialog = []
        # add instruction and ICL examples
        if not self.instruction is None:
            dialog.append({
                    "role": "system",
                    "content": self.instruction
            })
        if self.icl_examples:
            examples = self.select_icl_examples(
                self.icl_examples,
                history,
                self.qu_config.qu_icl_selection_strategy,
                self.qu_config.qu_icl_num_examples
            )
            for ex in examples:
                for i, t in enumerate(ex):
                    user_template = self.PROMPT_TEMPLATE if i > 0 else self.NEW_QUESTION_PROMPT_TEMPLATE
                    dialog.append({
                        "role": "user",
                        "content": user_template.format(**t),
                    })
                    dialog.append({
                        "role": "assistant",
                        "content": self.OUTPUT_TEMPLATE.format(**t),
                    })

        dialog += history
        return dialog
    
    @staticmethod
    def select_icl_examples(icl_examples: List[Dict], history: List[Dict], icl_selection_strategy: str="dynamic", icl_num_examples: int=8) -> List[Dict]:
        if icl_selection_strategy == "dynamic":
            query = " ".join((d["content"] for d in history))
            icl_example_texts = [" ".join(f'{e["qu_input"]} {e["operator_tree"]}' for e in icl_example) for icl_example in icl_examples]
            scored_list = bm25_scoring(query=query, documents=icl_example_texts, tokenize_fct=tokenize_code, n=icl_num_examples)
            icl_examples = [icl_examples[i["index"]] for i in scored_list]
            # logger.debug(f"Selected ICL examples for query {query}: {icl_examples}")
            return icl_examples
        else:
            return icl_examples[:icl_num_examples]

    def run(self, question: str, sampling_params: Dict) -> List[OperatorTree]:
        operator_trees = self.run_batch(questions=[question], sampling_params=sampling_params)[0]
        return operator_trees

    def run_batch(self, questions: List[str], sampling_params: Dict) -> List[List[OperatorTree]]:
        from collections import deque
        """
        Process multiple questions in a batch-wise manner, returning a list of Operator trees for each question.
        """
        # create a QUTree for each question (=root node)
        qu_trees = []
        for question in questions:
            initial_qu_input = "{{ QU(question=\"" + question + "\") }}"
            qu_tree = QUTree(initial_qu_input, None)
            qu_trees.append(qu_tree)

        # nodes that need LLM inference are queued up.
        # each node is a tuple of (qu_tree, qu_branch_input, history, recursion_depth).
        queue = deque()

        # initialize the queue with one entry per qu_tree
        for qu_tree in qu_trees:
            qu_input = qu_tree.qu_input  # or however you store the initial prompt
            next_qu_branches = self.QU_INPUT_PATTERN.findall(qu_input)
            next_qu_branches = [f"QU{qu_params}" for qu_params in next_qu_branches]
            if next_qu_branches:
                # For each discovered QU call, we push a "branch" node into the queue
                for qu_branch in next_qu_branches:
                    queue.append((qu_tree, qu_branch, [], 0))

        # in each batch, we gather all the (dialog, branch_info) that need inference.
        logger.debug(f"Initial queue has {len(queue)} items")
        while queue:
            items_to_process = []
            
            batch_size = self.qu_config.qu_inference_batch_size
            for _ in range(batch_size):
                if not queue:
                    break
                items_to_process.append(queue.popleft())

            # build the dialog for each item in the batch
            batch_dialogs = []
            for (qu_tree, qu_branch_input, local_history, recursion_depth) in items_to_process:
                # build or update the user prompt in the history
                if local_history:
                    # if there's existing history, just add this new user turn
                    self.add_to_history(local_history, self.PROMPT_TEMPLATE.format(qu_input=qu_branch_input), role="user")
                else:
                    # otherwise, initialize a new conversation
                    self.add_to_history(local_history, self.NEW_QUESTION_PROMPT_TEMPLATE.format(qu_input=qu_branch_input), role="user")
                
                # convert the local_history into a final LLM-ready dialog
                dialog = self.create_dialog(local_history)
                batch_dialogs.append(dialog)

            # ensure the dialogs are correctly created
            for dialog in batch_dialogs:
                is_user = True
                for turn in [t for t in dialog if not t["role"] == "system"]:
                    assert (is_user and turn["role"] == "user") or (not is_user and turn["role"] == "assistant"), f"Incorrect dialog created for qu_branch_input={qu_branch_input}: is_user={is_user}, turn={turn}, dialog={dialog}, batch_dialogs={batch_dialogs}"
                    is_user = not is_user

            # batched inference returning one output per input dialog, retaining the input order
            logger.debug(f"Running inference with {len(batch_dialogs)} inputs")
            if type(self.model) == QUModel:
                input_texts = [represent_input_seq2seq(dialog=dialog, tokenizer=self.model.tokenizer, max_length=self.qu_config.max_input_length) for dialog in batch_dialogs]
                batch_results = self.model.batch_inference(input_texts, sampling_params.copy())  # copy sampling params to retain "n" key
            elif type(self.model) == QUModelCausal:
                input_texts = [represent_input_causal(dialog=dialog, tokenizer=self.model.tokenizer, max_length=self.qu_config.max_input_length) for dialog in batch_dialogs]
                batch_results = self.model.batch_inference(input_texts, sampling_params.copy())  # copy sampling params to retain "n" key
            else:
                batch_results = self.model.batch_inference(batch_dialogs, sampling_params)

            # distribute the results back to the correct nodes, build child subtrees
            for i, (qu_tree, qu_branch_input, history, recursion_depth) in enumerate(items_to_process):
                local_history = deepcopy(history)
                dialog_result = batch_results[i]

                # if model returns single string
                if sampling_params["n"] == 1:
                    next_qu_options = [dialog_result]  
                else:
                    # or deduplicate if top-k
                    next_qu_options = []
                    for candidate in dialog_result:
                        if candidate not in next_qu_options:
                            next_qu_options.append(candidate)
                    logger.debug(f"next_qu_options={next_qu_options}")

                # for each next QU option, create a subtree:
                # (i) add it to local_history with role="assistant"
                # (ii) recursively parse further QU calls if any
                qu_childs = list()
                for qu_option in next_qu_options:
                    history_with_option = deepcopy(local_history)
                    self.add_to_history(history_with_option, qu_option, role="assistant")

                    # detect loops (same QU option seen before)
                    if self.is_loop(history_with_option):
                        logger.error(f"Detected loop 1 with qu_option={qu_option}, local_history={local_history}")
                        continue
                    
                    # create subtree
                    qu_option_subtree = QUTree(qu_option, qu_branch_input) 
                    
                    # check if we need to expand it further
                    if recursion_depth < self.MAX_RECURSION_DEPTH:
                        # If it has more QU calls, queue them up
                        next_sub_branches = self.QU_INPUT_PATTERN.findall(qu_option)
                        next_sub_branches = [f"QU{qu_params}" for qu_params in next_sub_branches]
                        for sub_branch in next_sub_branches:
                            if self.is_loop(history_with_option):
                                logger.error(f"Detected loop 2 with sub_branch={sub_branch}, local_history={local_history}")
                                continue
                            history = deepcopy(history_with_option)
                            queue.append((qu_option_subtree, sub_branch, history, recursion_depth+1))
                    
                    qu_childs.append(qu_option_subtree)
                qu_branch_node = QUTreeBranch(qu_childs)
                
                # finally, attach the branch node to the current qu_tree
                qu_tree.add_branch(qu_branch_node)

        # convert each QUTree to a list of Operator trees
        all_plans = []
        for tree in qu_trees:
            plans = tree.to_operator_trees()
            all_plans.append(plans)
        return all_plans

    @staticmethod
    def is_loop(history_with_option: List[Dict]):
        last_qu_option = history_with_option[-1]["content"]
        return any((t["content"]).strip() == last_qu_option.strip() for t in history_with_option[:-1])

    @staticmethod
    def initialize_question_to_operator_tree(qu_result_path: str, override: bool=False):
        if os.path.exists(qu_result_path) and not override:
            operator_tree_mapping = load_jsonl(qu_result_path)
            question_to_operator_tree = {key: value for mapping in operator_tree_mapping for key, value in mapping.items()}
            logger.info(f"Loaded existing QU result for {len(question_to_operator_tree)} questions from {qu_result_path}.")
        else:
            logger.info(f"Starting with new question_to_operator_tree dictionary...")
            question_to_operator_tree = dict()
        return question_to_operator_tree
