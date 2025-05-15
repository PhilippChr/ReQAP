import os
import re
import time
import json
import random
import pandas as pd
import pathlib
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, BatchEncoding
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from loguru import logger

from reqap.library.library import get_persona_names, load_json, store_json, store_jsonl_line, load_jsonl, store_jsonl, num_lines
from reqap.library.metrics import recall
from reqap.retrieval.query_execution import QueryExecution
from reqap.classes.observable_event import ObservableEvent, METADATA_KEYS
from reqap.qu.qu_module import QuestionUnderstandingModule
from reqap.qu.operator_tree import OperatorTree
from reqap.llm.crossencoder_model import DatasetCrossEncoder
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.splade.models import Splade
from reqap.retrieval.retrieval_pattern import RetrievalPattern


class DatasetRetrieval(DatasetCrossEncoder):
    def __init__(self, input_encodings: BatchEncoding, labels: List[List[int]], dataset_length: int):
        super(DatasetRetrieval, self).__init__(input_encodings, labels, dataset_length)

    @staticmethod
    def prepare_event_input(retrieve_query: str, event: ObservableEvent) -> Tuple[str, str]:
        event_data = event.event_data
        if "properties_mentioned" in event_data:
            del(event_data["properties_mentioned"])
        
        # transform to string
        event_str = ",\n".join(f"{k}: {json.dumps(v)}" for k, v in event_data.items() if not k in METADATA_KEYS)
        return retrieve_query, event_str
    
    @staticmethod
    def prepare_pattern_input(retrieve_query: str, pattern: str) -> Tuple[str, str]:
        pattern = pattern.replace(RetrievalPattern.PATTERN_MERGE_STR, ", ")
        return retrieve_query, pattern


class DatasetCrossEncoderFactory:
    QUERY_PATTERN = re.compile('query="([^"]+)"')

    def __init__(self, config: DictConfig, ce_config: DictConfig):
        self.config = config
        self.ce_config = ce_config

    
    def create(self, tokenizer: PreTrainedTokenizer, data_input_path: str, input_type: str="pattern") -> DatasetCrossEncoder:
        """
        FINAL STAGE:
        Create a dataset based on the provided tokenizer.
        Decision matix for patterns to resolve conflicts (pp: positive pattern, np: negative pattern):
                            np - positive, np - conflict, np - negative, np - NONE
            pp - positive           1              1            2               2
            pp - conflict           1              1            1               1
            pp - negative           0              1            1               1
            pp - NONE               0              1            1               -
        """
        data = load_jsonl(data_input_path)

        # filtering stage to avoid conflicts of labels for same input
        input_to_label = dict()  # store mapping from input -> label
        if input_type == "event":
            """
            Deal with events as input.
            """
            for it in tqdm(data):
                input_tuple = tuple(it["input"])
                label = it["label"]
                
                # input is event
                if it["input_type"] == "event":
                    has_conflict = label != input_to_label.get(input_tuple, label)
                    if has_conflict:
                        input_to_label[input_tuple] = [None]
                    else:
                        input_to_label[input_tuple] = label
        
        elif input_type == "pattern":
            """
            Deal with patterns as input.
            """
            for it in tqdm(data):
                input_tuple = tuple(it["input"])
                label = it["label"]

                # skip events
                if it["input_type"] == "event":
                    continue

                # init input label
                if not input_tuple in input_to_label:
                    input_to_label[input_tuple] = [None, None]  # np-label, pp-label (0: neg, 1: pos, -1: conflict)

                # negative pattern
                if it["input_type"] == "negative_pattern":
                    has_conflict = label[1] != input_to_label[input_tuple][0] and not input_to_label[input_tuple][0] is None
                    if has_conflict:
                        logger.debug(f"Conflict detected for {input_tuple}: label[1]={label[1]}, input_to_label[input_tuple][0]={input_to_label[input_tuple][0]}")
                        input_to_label[input_tuple][0] = -1
                    else:
                        input_to_label[input_tuple][0] = label[1] 
                
                # positive pattern
                elif it["input_type"] == "pattern":
                    has_conflict = label[1] != input_to_label[input_tuple][1] and not input_to_label[input_tuple][1] is None
                    if has_conflict:
                        logger.debug(f"Conflict detected for {input_tuple}: label[1]={label[1]}, input_to_label[input_tuple][1]={input_to_label[input_tuple][1]}")
                        input_to_label[input_tuple][1] = -1
                    else:
                        input_to_label[input_tuple][1] = label[1]

            # process data and derive labels
            input_to_label_new = dict()
            for input_tuple, label in input_to_label.items():
                if label[0] == 1.0 and label[1] in [0.0, None]:
                    label = [1.0, 0.0, 0.0]
                elif label[1] == 1.0 and label[0] in [0.0, None]:
                    label = [0.0, 0.0, 1.0]
                else:
                    label = [0.0, 1.0, 0.0]
                input_to_label_new[input_tuple] = label
            input_to_label = input_to_label_new

            # tmp_data = [{"input": input_tuple, "label": label} for input_tuple, label in input_to_label.items()]  
            # store_jsonl("tmp.jsonl", tmp_data)
        else:
            raise NotImplementedError(f"Input type {input_type} not implemented.")

        
        inputs = [input_tuple for input_tuple, label in input_to_label.items() if label[0] is not None]
        labels = [label for label in input_to_label.values() if label[0] is not None]
        logger.debug(f"Length(inputs): {len(inputs)}")
        logger.debug(f"Length(labels): {len(labels)}")

        # encode inputs
        max_length = self.ce_config.crossencoder_max_length
        input_encodings = tokenizer(
            [i[0] for i in inputs],
            [i[1] for i in inputs],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # check for overflowing inputs
        exceeding_count = 0
        for text in inputs:
            if len(tokenizer.encode(text)) > max_length:
                exceeding_count += 1
        logger.debug(f"{exceeding_count} inputs have been truncated.")
        
        # return result
        dataset_length = len(inputs)
        return DatasetRetrieval(input_encodings, labels, dataset_length)
    
    """
    STAGE 1: 
    Apply QU to identify retrieval calls.
    """
    def derive_retrieve_calls(self, split: str, data_output_path: str) -> None:
        # set paths
        benchmark_dir = self.config.benchmark.benchmark_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)
        
        # load QU result
        qu_result_path = self.config.qu.qu_result_paths[split]
        question_to_operator_tree = QuestionUnderstandingModule.initialize_question_to_operator_tree(
            qu_result_path=qu_result_path,
            override=False
        )

        # iterate through personas
        output_dir = os.path.dirname(data_output_path)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(data_output_path, "w") as fp_out:
            for persona in personas:
                logger.debug(f"Starting with person={persona}")
                questions_path = f"{persona_dir}/{persona}/questions.json"
                data = load_json(questions_path)
                for instance in tqdm(data):
                    if self.retain_instance(instance):
                        question = instance["question"]
                        operator_tree_dicts = question_to_operator_tree[question]
                        operator_trees = OperatorTree.from_operator_tree_dicts(operator_tree_dicts)
                        sql_query = instance["sql_query"]
                        instance["persona"] = persona

                        # derive SQL retrieval query and corresponding RETRIEVE query
                        derived_pairs = set()  # store pairs, ensure uniqueness
                        for i, operator_tree in enumerate(operator_trees):
                            retrieve_calls = list(set(call for call in operator_tree.get_retrieve_calls()))
                            if not retrieve_calls:
                                continue
                            retrieve_call = retrieve_calls[0]
                            retain_where_clause = not self.drop_where_clause(operator_tree.to_dict(), retrieve_call)
                            retrieval_sql_query = QueryExecution.derive_retrieval_query(sql_query, retain_where_clause)
                            other_retrieval_sql_query = QueryExecution.derive_retrieval_query(sql_query, not retain_where_clause)
                            if (retrieval_sql_query != other_retrieval_sql_query) or not retain_where_clause:
                                logger.debug(f"Retrieve call: `{retrieve_call}`, retrieval SQL query: `{retrieval_sql_query}`, retain_where_clause=`{retain_where_clause}`, question=`{question}`, retrieval SQL query otherwise: `{other_retrieval_sql_query}` operator_tree=`{json.dumps(operator_tree.to_dict())}`")
                            derived_pairs.add((retrieve_call, retrieval_sql_query))

                        # add data
                        for retrieve_call, retrieval_sql_query in derived_pairs:
                            new_instance = instance.copy()
                            new_instance["retrieve_call"] = retrieve_call
                            new_instance["retrieval_sql_query"] = retrieval_sql_query
                            store_jsonl_line(fp_out, new_instance)
        
    @staticmethod
    def drop_where_clause(operator_tree_dict: Dict, retrieve_call: str, parents: List[str]=list()):
        """
        Identify whether to drop the WHERE clause or not.
        Traverses the OperatorTree dict, retains the parent calls for each traversal,
        and once the actual retrieve_call is reached, checks if a sequence of SELECT + FILTER operations is before the retrieve_call.
        If so, this structure is supposed to match the WHERE clause, which indicates that the WHERE clause
        is not resolved within the RETRIEVE call in the Operator Tree.
        """
        def matches_where_clause(qu_call: str):
            # do not consider FILTER operations for any temporal condition, as these
            # are dropped from WHERE clause anyway, and never matched by RETRIEVE
            if "date" in qu_call or "day" in qu_call or "time" in qu_call: 
                return False
            else:
                return qu_call.startswith("FILTER")

        if operator_tree_dict["qu_input"] == retrieve_call:
            if len(parents) < 2:
                return False
            elif parents[-1].startswith("SELECT") and matches_where_clause(parents[-2]):
                logger.debug(f"WHERE claused matched immediately for retrieve_call={retrieve_call} because of parents={parents[-2], parents[-1]}")
                return True
            else:
                # try to match WHERE clause higher up in the plan as well (found to be effective)
                for i in range(len(parents)-1):
                    if parents[i+1].startswith("SELECT") and matches_where_clause(parents[i]):
                        logger.warning(f"WHERE claused matched higher above for retrieve_call={retrieve_call} because of parents={parents[i], parents[i+1]}")
                        return True
                return False
        else:
            new_parents = parents.copy()
            new_parents.append(operator_tree_dict["qu_input"])
            # WHERE clause in SQL query should be kept, in case there is no corresponding SELECT -> FILTER path in the Operator Tree
            # => If any structure in the Operator Tree matches the WHERE clause, the RETRIEVE call should not cover the WHERE clause itself 
            where_clause_is_matched = any(DatasetCrossEncoderFactory.drop_where_clause(
                    operator_tree_dict=child,
                    retrieve_call=retrieve_call,
                    parents=new_parents
                )
                for child in operator_tree_dict["childs"]
            )
            return where_clause_is_matched


    """
    STAGE 2:
    Apply SPLADE retrieval and derive positives and negatives, including positive and negative patterns. 
    """
    def derive_data(self, splade_model: Splade, split: str, data_input_path: str, data_output_path: str, single_persona: str=None) -> None:
        """
        The persona parameter is used to compute the result for a single person only.
        """
        # set paths
        benchmark_dir = self.config.benchmark.benchmark_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)
        logger.info(f"Starting with split {split} ({len(personas)} personas)")

        # ensure output dir exists
        if not single_persona is None:
            logger.warning(f"Running derive_data for single persona: {single_persona}")
            data_output_path = os.path.join(os.path.dirname(data_output_path), "personas", f"{single_persona}.jsonl")
            logger.warning(f"Writing output to: {data_output_path}")
        output_dir = os.path.dirname(data_output_path)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # derive inputs/outputs
        prev_persona = None
        lines_count = num_lines(data_input_path)
        with open(data_input_path, "r") as fp_in:
            for line in tqdm(fp_in, total=lines_count):
                instance = json.loads(line)
                persona = instance["persona"]

                # this line enables parallel inference for all personas
                if not single_persona is None and persona != single_persona:
                    continue

                if persona != prev_persona:
                    # initialize paths and modules
                    str_events_csv_path = f"{persona_dir}/{persona}/{persona}_str.csv"
                    obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
                    collection = CollectionDataset(data_path=obs_events_csv_path)
                    observable_events_df = collection.to_df()
                    event_data_df = pd.DataFrame(observable_events_df["event_data"].tolist())
                    splade_config = self.config.splade
                    splade_index_path = os.path.join(splade_config.splade_indices_dir, f"{persona}.splade_index")
                    splade_retrieval = SparseRetrieval(
                        splade_config=splade_config,
                        model=splade_model,
                        collection=collection,
                        dim_voc=splade_model.output_dim,
                        splade_index_path=splade_index_path
                    )
                    query_execution = QueryExecution(
                        obs_events_csv_path=obs_events_csv_path,
                        str_events_csv_path=str_events_csv_path
                    )
                    observable_events = ObservableEvent.from_csv_path(obs_events_csv_path)
                    source_distribution = defaultdict(lambda: 0)
                    for oe in observable_events:
                        source_distribution[oe.event_type] += 1
                    logger.debug(f"source_distribution={dict(source_distribution)}")
                    prev_persona = persona

                # derive inputs and labels
                inputs, input_types, labels = self._process_instance(instance, splade_retrieval, query_execution, observable_events_df, event_data_df, observable_events, source_distribution)
                assert len(inputs) == len(input_types), f"Assertion failed: {len(inputs)}, {len(input_types)}, {len(labels)}"
                assert len(inputs) == len(labels), f"Assertion failed: {len(inputs)}, {len(input_types)}, {len(labels)}"

                # store data (tokenization independent)
                data = [{"positive": l[1] == 1.0, "input": i, "input_type": it, "label": l} for i, it, l in zip(inputs, input_types, labels)]
                store_jsonl(data_output_path, data, "a")

    def _process_instance(
            self,
            instance: Dict,
            splade_retrieval: SparseRetrieval,
            query_execution: QueryExecution,
            observable_events_df: pd.DataFrame,
            event_data_df: pd.DataFrame,
            observable_events: List[ObservableEvent],
            source_distribution: Dict[str, int]
        ) -> Tuple[List, List, List]:
        question = instance["question"]

        ### Run SPLADE retrieval
        retrieve_call = instance["retrieve_call"]
            
        # identify query
        retrieve_queries = self.QUERY_PATTERN.findall(retrieve_call)
        if not retrieve_queries:
            logger.warning(f'No retrieval queries for `{question}`.')
            return [], [], []
        retrieve_query = retrieve_queries[0]
        res, _ = splade_retrieval.retrieve(retrieve_query, involve_model=True, top_k=0)

        # process result
        retrieved_event_ids = [int(e["id"]) for e in res]
        retrieval_result = {
            "retrieve_query": retrieve_query,
            "retrieval_result": res,
            "retrieved_event_ids": retrieved_event_ids,
            "retrieval_result_len": len(retrieved_event_ids),
        }

        ### Derive gold retrieval results
        inputs = list()
        input_types = list()
        labels = list()
        retrieve_query = retrieval_result["retrieve_query"]
        retrieved_event_ids = retrieval_result["retrieved_event_ids"]
        
        ### Derive ground-truth results with (and without) WHERE clause
        retrieval_sql_query = instance["retrieval_sql_query"]
        retrieval_data = self.derive_gold_retrieval_data(
            instance,
            retrieval_sql_query,
            observable_event_data=observable_events_df,
            query_execution=query_execution
        )
        if retrieval_data is None:  # SQL query failed => drop instance
            logger.warning(f'SQL query `{retrieval_sql_query}` failed for `{question}`.')
            return [], [], []

        # eval
        splade_result = retrieval_result["retrieval_result"]
        gold_obs_event_ids = set(retrieval_data["gold_obs_event_ids"])
        retrieved_event_ids = [int(e["id"]) for e in splade_result]
        rec = recall(gold_obs_event_ids, retrieved_event_ids)

        ### Derive positive / negative patterns
        patterns = RetrievalPattern.identify_candidate_positive_patterns(
            retrieval_result=retrieval_result["retrieval_result"],
            min_events_matched=self.ce_config.retrieval_pattern.min_events_matched_train
        )
        logger.debug(f"Identified {len(patterns)} patterns: {patterns}")

        # positive patterns
        max_positive_patterns = self.ce_config.crossencoder_train_num_positive_patterns
        positive_patterns, negative_patterns_positives = self.process_candidate_positive_patterns(patterns, observable_events_df, event_data_df, gold_obs_event_ids)
        logger.debug(f"Positive patterns for retrieve_query=`{retrieve_query}` retrieval_sql_query=`{retrieval_sql_query}`: `{positive_patterns}`")
        if positive_patterns:
            positive_patterns = random.sample(positive_patterns, k=min(max_positive_patterns, len(positive_patterns)))
        result = {
            "positive_patterns": positive_patterns,
            "gold_obs_event_ids": gold_obs_event_ids,
            "retrieved_event_ids": retrieved_event_ids,
            "recall": rec
        }

        ### Drop cases for which recall is too low: likely that retrieval query is of bad quality
        if result["recall"] < self.ce_config.crossencoder_train_min_recall:
            logger.warning(f'Skipping `{question}` with query `{retrieve_query}` due to low recall ({result["recall"]}).')
            return [], [], []
        positive_patterns = result["positive_patterns"]
        gold_obs_event_ids = result["gold_obs_event_ids"]
        retrieved_event_ids = result["retrieved_event_ids"]

        # PATTERNS: identify negative patterns
        max_negative_patterns = self.ce_config.crossencoder_train_num_negative_patterns
        negative_patterns = [p for p in patterns if not p in positive_patterns]
        if negative_patterns:
            negative_patterns = random.sample(negative_patterns, k=min(max_negative_patterns, len(negative_patterns)))

        # PATTERNS: construct model inputs and labels
        for pattern in positive_patterns + negative_patterns:
            input_tuples = DatasetRetrieval.prepare_pattern_input(retrieve_query, pattern)
            inputs.append(input_tuples)
        labels += [[0.0, 1.0]] * len(positive_patterns) + [[1.0, 0.0]] * len(negative_patterns)
        input_types += ["pattern"] * (len(positive_patterns)+len(negative_patterns))


        ### Derive negative patterns
        # NEGATIVE PATTERNS: identify positives+negatives
        negative_patterns_negatives = list()
        # negative patterns can also be derived from candidate positive patterns, that lead to only incorrect results
        if not self.ce_config.get("unified_negative_patterns", False):
            logger.debug("Negative patterns are not unified: if a candidate positive pattern is found to be negative instead, it will not be considered as such.")
            negative_patterns_positives = list()
        source_to_label = defaultdict(lambda: False)
        for event_id in retrieved_event_ids:
            source = observable_events[event_id].event_type
            source_to_label[source] = source_to_label[source] or event_id in gold_obs_event_ids
        for source, label in source_to_label.items():
            pattern = RetrievalPattern.NEGATIVE_PATTERN_TEMPLATE.format(source=source)
            if label:
                negative_patterns_negatives.append(pattern)
            else:
                negative_patterns_positives.append(pattern)
        for pattern in negative_patterns_positives + negative_patterns_negatives:
            input_tuples = DatasetRetrieval.prepare_pattern_input(retrieve_query, pattern)
            inputs.append(input_tuples)
        labels += [[0.0, 1.0]] * len(negative_patterns_positives) + [[1.0, 0.0]] * len(negative_patterns_negatives)
        input_types += ["negative_pattern"] * (len(negative_patterns_positives)+len(negative_patterns_negatives))


        ### Derive positives / negatives
        # EVENTS: positives
        max_pos_events = self.ce_config.crossencoder_train_num_positives
        pos_ev_ids = [e_id for e_id in retrieved_event_ids if e_id in gold_obs_event_ids]
        pos_ev_ids = random.sample(pos_ev_ids, k=min(max_pos_events, len(pos_ev_ids)))

        # EVENTS: sample negatives from SPLADE retrieval => challenging
        max_neg_events = self.ce_config.crossencoder_train_num_negatives
        neg_ev_ids = [e_id for e_id in retrieved_event_ids if e_id not in gold_obs_event_ids][:max_neg_events]  # prefer high-ranked (by SPLADE) negatives 
        # neg_ev_ids = random.sample(neg_ev_ids, k=min(max_neg_events, len(neg_ev_ids)))  # random SPLADE negatives
        
        # EVENTS: random negatives
        max_random_neg_events = self.ce_config.crossencoder_train_num_random_negatives
        start = time.time()
        random_neg_ev_ids = self.sample_random_negatives(observable_events, source_distribution, gold_obs_event_ids, neg_ev_ids, max_random_neg_events)
        neg_ev_ids += random_neg_ev_ids
        logger.debug(f"Sampling negatives took {time.time() - start} seconds...")

        # EVENTS: construct model inputs and labels
        for event_id in pos_ev_ids + neg_ev_ids:
            event = observable_events[event_id]
            input_tuples = DatasetRetrieval.prepare_event_input(retrieve_query, event)
            inputs.append(input_tuples)
        labels += [[0.0, 1.0]] * len(pos_ev_ids) + [[1.0, 0.0]] * len(neg_ev_ids)
        input_types += ["event"] * (len(pos_ev_ids)+len(neg_ev_ids))

        return inputs, input_types, labels

    @classmethod
    def derive_gold_retrieval_data(
            cls,
            instance: Dict,
            retrieval_sql_query: str,
            observable_event_data: pd.DataFrame,
            query_execution: QueryExecution
        ) -> List[Dict]:
        try:
            str_event_ids, obs_event_ids = cls.retrieve_gold_events(retrieval_sql_query, observable_event_data, query_execution)
        except QueryExecution.SQLError:
            logger.warning(f"Retrieval query `{retrieval_sql_query}` failed.")
            return None
        return {
            "id": instance["id"],
            "question": instance["question"],
            "retrieval_sql_query": retrieval_sql_query,
            "gold_str_event_ids": str_event_ids,
            "gold_obs_event_ids": obs_event_ids,
        }

    @staticmethod
    def retrieve_gold_events(retrieval_sql_query: str, observable_event_data: pd.DataFrame, query_execution: QueryExecution) -> Tuple[List[int], List[int]]:
        """
        Retrieve the gold events for the given retrieval SQL query.
        """
        def _is_obs_event_sql_query(query):
            OBS_EVENTS_TABLES = ["social_media", "mail", "calendar"]
            return any(t_name in query for t_name in OBS_EVENTS_TABLES)
        sql_query_res = query_execution.run_sql_query(retrieval_sql_query)
        str_event_ids = sql_query_res.values.tolist()
        str_event_ids = list(id_ for id_list in str_event_ids for id_ in id_list)
        if _is_obs_event_sql_query(retrieval_sql_query):  # there are no structured events for the query
            obs_event_ids = str_event_ids
            return {}, obs_event_ids
        observable_event_data["structured_event_id"] = pd.to_numeric(observable_event_data["structured_event_id"])
        obs_events = pd.merge(sql_query_res, observable_event_data, left_on='id', right_on='structured_event_id')
        obs_event_ids = pd.to_numeric(obs_events["id_y"]).tolist()
        return str_event_ids, obs_event_ids
    
    @staticmethod
    def retain_instance(instance: Dict) -> bool:
        """
        Whether to retain the instance in the dataset or not.
        Currently drops queries that are not simple (e.g., that include joins or so).
        """
        query = instance["sql_query"]
        return QueryExecution.is_simple_query(query)
    
    """
    POSITIVE/NEGATIVE SAMPLING
    """
    @staticmethod
    def sample_random_negatives(observable_events: List[ObservableEvent], source_distribution: Dict[str, int], gold_obs_event_ids: Set[int], neg_ev_ids: List[int], n: int) -> List[int]:
        sample_space = list()
        sample_space_weights = dict()
        for e in observable_events:
            sample_space_weights[int(e.id)] = 1/((source_distribution[e.event_type])*10+1)
            sample_space.append(int(e.id))
        for idx in list(gold_obs_event_ids) + neg_ev_ids:
            sample_space.remove(idx)
        sample_space_weights = [sample_space_weights[e_id] for e_id in sample_space]
        random_negatives = random.choices(
            sample_space,
            k=min(len(sample_space), n),
            weights=sample_space_weights
        )
        random_negatives = list(set(random_negatives))  # dedup (there could be duplicates due to usage of choices)
        return random_negatives
    
    @staticmethod
    def process_candidate_positive_patterns(candidate_positive_patterns: List[str], observable_events_df: pd.DataFrame, event_data_df: pd.DataFrame, gold_ids: Set[int]) -> List[str]:
        """
        Positive patterns have perfect precision (no restriction for recall).
        Negative patterns would have none of the gold IDs in the result.
        Returns all positive patterns and all negative patterns (others are dropped).
        """        
        logger.debug(f"Starting with deriving positive patterns for {len(candidate_positive_patterns)} patterns")
        pos_patterns = list()
        neg_patterns = list()
        for pattern in candidate_positive_patterns:
            filtered_df = RetrievalPattern.apply_positive_pattern(observable_events_df, event_data_df, pattern)
            ids_in_df = set(int(id_str) for id_str in filtered_df["id"])
            if not ids_in_df:
                continue
            if len(ids_in_df - gold_ids) == 0:  # perfect precision (=no incorrect ID in ids_in_df)
                pos_patterns.append(pattern)
            elif len(ids_in_df & gold_ids) == 0:
                neg_patterns.append(pattern)
        return pos_patterns, neg_patterns

    """
    STAGE 3:
    Derive equivalent RETRIEVE queries.
    These can then be used in the training loop, to derive alternative
    Operator Trees without additional (and expensive) LLM sampling.

    Writes out a JSON-dict which maps a RETRIEVE call to all equivalents,
    into the provided output_path.
    """

    def derive_equivalent_retrieve_queries(self, input_path: str, output_path: str) -> None:
        # load data
        data = load_jsonl(input_path)

        # derive mapping from query to patterns
        pattern_to_query = defaultdict(lambda: set())
        for instance in data:
            if not instance["positive"] or not instance["input_type"] == "pattern":
                continue
            query, pattern = instance["input"]
            pattern_to_query[query].add(pattern.strip())
        
        # derive mapping from query to equivalents
        query_to_equivalents = defaultdict(lambda: set())
        for q1, p1 in pattern_to_query.items():
            for q2, p2 in pattern_to_query.items():
                if q1 == q2:
                    continue
                if p1 == p2:
                    query_to_equivalents[q1].add(q2)
                    query_to_equivalents[q2].add(q1)

        # make sure to convert sets into lists
        query_to_equivalents = dict(query_to_equivalents)
        for query in query_to_equivalents:
            query_to_equivalents[query] = list(query_to_equivalents[query])

        # store result
        store_json(output_path, query_to_equivalents)
