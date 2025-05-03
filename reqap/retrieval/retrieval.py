import pandas as pd
from omegaconf import DictConfig
from loguru import logger
from typing import List

from reqap.classes.observable_event import ObservableEvent
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.crossencoder.crossencoder_module import CrossEncoder
from reqap.retrieval.retrieval_pattern import RetrievalPattern


class Retrieval():
    def __init__(self, config: DictConfig, obs_events_csv_path: str, splade_index_path: str):
        self.config = config
        self.ce_config = config.crossencoder
        self.splade_config = config.splade
        splade_model = Splade(self.splade_config.splade_model_type_or_path, agg="max")
        collection = CollectionDataset(data_path=obs_events_csv_path)
        self.observable_events_df = collection.to_df()
        self.event_data_df = pd.DataFrame(self.observable_events_df["event_data"].tolist())
        self.sparse_retrieval = SparseRetrieval(
            splade_config=self.splade_config,
            model=splade_model,
            collection=collection,
            dim_voc=splade_model.output_dim,
            splade_index_path=splade_index_path
        )
        self.splade_involve_model = self.splade_config.get("splade_involve_model", True)
        self.crossencoder = CrossEncoder(config=config, ce_config=self.ce_config)
        self.cache = dict()

    def retrieve(self, query: str, ordered: bool=False) -> List[ObservableEvent]:
        """
        Function to implement RETRIEVE function.
        Involves SPLADE for high-recall first-stage retrieval,
        and a cross-encoder classification model.

        By setting ordered = True, we can use it within RAG model.
        """
        # try to access cache
        if query in self.cache:
            return self.cache[query]
        
        """
        Step 1: Sparse Retrieval - retrieve candidates via SPLADE.
        """
        threshold = self.splade_config.get("splade_threshold", 0)
        candidates, _ = self.sparse_retrieval.retrieve(
            query,
            involve_model=self.splade_involve_model,  # whether to run model to expand query or not
            top_k=0,
            threshold=threshold
        )
        logger.debug(f"SPLADE threshold in use: {threshold}")
        logger.debug(f"{len(candidates)} events after SPLADE retrieval.")
        event_to_splade_score = {int(d["data"]["id"]): d["score"] for d in candidates}

        # avoids computing full CE result for extremely large outputs in RAG
        if ordered:
            candidates = candidates[:10000]

        # for ablation study: skip cross-encoder completely
        if self.splade_config.get("splade_only", False):
            obs_events = [ObservableEvent.from_dict(d["data"]) for d in candidates]
            return obs_events
        
        """
        Step 2: Pattern Detection - identify candidate positive and negative patterns.
        """
        if self.ce_config.retrieval_pattern.apply:
            positive_patterns = RetrievalPattern.identify_candidate_positive_patterns(
                retrieval_result=candidates,
                min_events_matched=self.ce_config.retrieval_pattern.min_events_matched_inference
            )
            negative_patterns = RetrievalPattern.identify_candidate_negative_patterns(candidates)
        else:
            positive_patterns = list()
            negative_patterns = list()
        candidate_obs_events = [ObservableEvent.from_dict(d["data"]) for d in candidates]
        candidate_obs_events_dict = {int(e.id): e for e in candidate_obs_events}

        """
        Step 3: Pattern Classification - classify candidate patterns into (0) irrelevant, (1) partially relevant, and (2) relevant.
        """
        patterns = positive_patterns + negative_patterns
        if self.ce_config.retrieval_pattern.apply:
            scored_patterns = self.crossencoder.run_for_patterns(query, patterns)
            logger.debug(f"Scored patterns: {scored_patterns}")
            if self.ce_config.get("unified_negative_patterns", False):
                accepted_positive_patterns = [(c["pattern"], c["probabilities"]) for c in scored_patterns if c["class"] == 2]
                accepted_negative_patterns = [(c["pattern"]) for c in scored_patterns if c["class"] == 0]
            else:
                accepted_positive_patterns = [(c["pattern"], c["probabilities"]) for c in scored_patterns[:len(positive_patterns)] if c["class"] == 2]
                accepted_negative_patterns = [(c["pattern"]) for c in scored_patterns[len(positive_patterns):] if c["class"] == 0]
            logger.debug(f"Found {len(accepted_positive_patterns)} positive patterns for query=`{query}`: {accepted_positive_patterns}.")
            logger.debug(f"Found {len(accepted_negative_patterns)} negative patterns for query=`{query}`: {accepted_negative_patterns}.")

            
            # Apply positive patterns
            positive_events_dict = dict()
            for pattern, pattern_probs in accepted_positive_patterns:
                filtered_df = RetrievalPattern.apply_positive_pattern(self.observable_events_df, self.event_data_df, pattern)
                
                # add positive candidates from filtered df
                positive_events = ObservableEvent.from_df(filtered_df)
                for oe in positive_events:
                    splade_score = event_to_splade_score.get(int(oe.id), "PATTERN_ONLY")
                    oe.set_retrieval_result(derived_via=pattern, splade_score=splade_score, ce_scores=pattern_probs)
                    positive_events_dict[int(oe.id)] = oe  # this helps to avoid duplicates due to same event matched by multiple patterns
            positive_events = [oe for _, oe in positive_events_dict.items()]


            # Apply negative patterns
            # => IMPORTANT: positive patterns are prioritized over conflicting negative patterns
            candidate_obs_events = [ev for ev_id, ev in candidate_obs_events_dict.items() if not int(ev_id) in positive_events_dict]
            num_events_before_negative_patterns = len(candidate_obs_events)
            for pattern in accepted_negative_patterns:
                candidate_obs_events = RetrievalPattern.apply_negative_pattern(candidate_obs_events, pattern)
            logger.debug(f"Dropped {num_events_before_negative_patterns - len(candidate_obs_events)} events with negative patterns.")
            logger.debug(f"{len(candidate_obs_events)} candidate events remaining after SPLADE retrieval and pattern matching.")
        else:
            positive_events = list()

        """
        Step 4: Event Classification - classify remaining events into (0) irrelevant, and (1) relevant.
        """
        if candidate_obs_events:
            scored_candidates = self.crossencoder.run_for_events(query, candidate_obs_events, ordered=ordered)
            for c in scored_candidates:
                if c["class"]:
                    oe = c["obs_event"]
                    splade_score = event_to_splade_score[int(oe.id)]
                    oe.set_retrieval_result(derived_via="EVENT", splade_score=splade_score, ce_scores=c["probabilities"])
                    positive_events.append(oe)
        logger.debug(f"{len(positive_events)} positive events after cross-encoder classification.")

        """
        Step 5: Event Deduplication - deduplicate events that express same information.
        Done in `create_computed_events` function, after predicting temporal information.
        """

        # store in cache
        self.cache[query] = positive_events
        return positive_events
    
    def load(self):
        self.crossencoder.load_models()

    def clear_cache(self):
        self.cache = dict()