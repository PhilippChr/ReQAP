import json
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from loguru import logger

from reqap.classes.observable_event import ObservableEvent
from reqap.retrieval.splade.index_construction import CollectionDataset


class RetrievalPattern:
    PATTERN_MERGE_STR = "|||"
    MIN_EVENTS_MATCHED = 1000
    NEGATIVE_PATTERN_TEMPLATE = "event: \"{source}\""
    
    @classmethod
    def identify_candidate_positive_patterns(cls, retrieval_result: List[Dict], min_events_matched: int=MIN_EVENTS_MATCHED) -> List[str]:
        """
        Identify candidate positive patterns.
        """
        # identify pattern
        patterns = defaultdict(lambda: set())
        for d in retrieval_result:
            event_data = d["data"]["event_data"]
            if type(event_data) == str:
                event_data = json.loads(event_data)
            for key, value in event_data.items():
                if type(value) in [int, float]:
                    continue
                if key == "properties_mentioned": 
                    continue
                #TODO: We could consider value-types or sub-strings as well here, to identify patterns (apart from exact values).
                patterns[f"{key}: {value}"].add(d["id"])
        patterns = list(filter(lambda i: len(i[1]) >= min_events_matched, patterns.items()))
        logger.debug(f"Number of positive patterns before merging: {len(patterns)}")
        patterns = cls._merge_pattern(patterns, min_events_matched)
        return patterns
    
    @classmethod
    def identify_candidate_negative_patterns(cls, retrieval_result: List[Dict]) -> List[str]:
        """
        Identify candidate negative patterns.
        """
        # identify pattern
        sources = set()
        for d in retrieval_result:
            event_source = d["data"]["event_type"]
            sources.add(event_source)
        patterns = list()
        for source in sources:
            pattern = cls.NEGATIVE_PATTERN_TEMPLATE.format(source=source)
            patterns.append(pattern)
        logger.debug(f"Number of negative patterns: {len(patterns)}")
        return patterns
    
    @classmethod
    def _merge_pattern(cls, patterns: List[str], min_events_matched: int=MIN_EVENTS_MATCHED) -> List[str]:
        # merge pattern (if doc sets are same)
        set_to_keys = defaultdict(list)
        for key, value in patterns:
            set_to_keys[frozenset(value)].append(key)
        merged_patterns = {}
        for values_set, keys in set_to_keys.items():
            new_key = cls.PATTERN_MERGE_STR.join(keys)
            merged_patterns[new_key] = set(values_set)
        logger.debug(f"Number of patterns after merging: {len(merged_patterns)}")
        merged_patterns = list(filter(lambda i: len(i[1]) >= min_events_matched, merged_patterns.items()))
        logger.debug(f"Number of patterns after filtering by num events matched: {len(merged_patterns)}")
        # drop items from patterns
        merged_patterns = [pattern for pattern, ids in merged_patterns]
        return merged_patterns
    
    @classmethod
    def prune_patterns(cls, collection: CollectionDataset, patterns: List[str]) -> List[str]:
        # filter invalid patterns, for which not all events were retrieved
        df = collection.to_df()
        filtered_pattern = list()
        ids_covered_by_pattern = set()
        # for pattern, ids in merged_pattern:
        for pattern, ids in patterns:
            new_df = cls.apply_positive_pattern(df, pattern)
            ids_in_df = set(new_df["id"])
            if ids_in_df == ids:
                filtered_pattern.append((pattern, ids))
                ids_covered_by_pattern.update(ids)
        logger.debug(f"Filtered pattern: {[p for p, _ in filtered_pattern]}")
        return filtered_pattern

    @classmethod
    def apply_positive_pattern(cls, df: pd.DataFrame, event_data_df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        conditions = dict(item.split(": ", 1) for item in pattern.split(cls.PATTERN_MERGE_STR))

        filter_mask = pd.Series(True, index=event_data_df.index)
        for column, value in conditions.items():
            filter_mask &= event_data_df[column] == value

        filtered_df = df[filter_mask]
        return filtered_df
    
    @classmethod
    def apply_negative_pattern(cls, candidate_obs_events: List[ObservableEvent], pattern: str) -> List[ObservableEvent]:
        source = pattern.replace("event: ","").replace("\"", "").strip()
        candidate_obs_events = [oe for oe in candidate_obs_events if not oe.event_type == source]
        return candidate_obs_events
