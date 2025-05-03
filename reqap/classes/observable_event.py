import json
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

from reqap.library.csv import initialize_csv_reader


METADATA_KEYS = {"derived_via", "splade_score", "ce_scores"}


class ObservableEventType(Enum):
    # structured
    MOVIE_STREAM = "movie_stream"
    MUSIC_STREAM = "music_stream"
    ONLINE_PURCHASE = "online_purchase"
    TVSERIES_STREAM = "tvseries_stream"
    WORKOUT = "workout"
    # non-structured
    CALENDAR = "calendar"
    MAIL = "mail"
    SOCIAL_MEDIA = "social_media"

    def is_structured(self):
        return not self.value in ["mail", "calendar", "social_media"]

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
        

@dataclass
class ObservableEvent:
    id: int
    structured_event_id: int
    start_date: str
    start_time: str
    end_date: str
    end_time: str
    event_type: ObservableEventType
    event_data: dict

    # OPTIONAL
    derived_via: Optional[str]="None"
    splade_score: Optional[str]="None"
    ce_scores: Optional[str]="None"

    CSV_HEADER = ["id", "structured_event_id", "start_date", "start_time", "end_date", "end_time", "event_type", "event_data", "properties_mentioned"]
    DATETIME_KEYS = ["start_date", "start_time", "end_date", "end_time"]

    def __init__(
        self,
        id: int,
        structured_event_id: int,
        start_date: str,
        start_time: str,
        end_date: str,
        end_time: str,
        event_type: ObservableEventType,
        event_data: dict,
        **kwargs
    ):
        self.id = int(id)
        self.structured_event_id = int(structured_event_id)
        self.start_date = start_date
        self.start_time = start_time
        self.end_date = end_date
        self.end_time = end_time
        self.event_type = event_type
        if "properties_mentioned" in event_data:
            del(event_data["properties_mentioned"])
        self.event_data = event_data

    @classmethod
    def from_csv_row(cls, event_row: Dict) -> "ObservableEvent":
        return cls._parse_data(event_row)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ObservableEvent":
        return cls._parse_data(data)

    @classmethod
    def from_csv_path(cls, obs_events_csv_path: str) -> List["ObservableEvent"]:
        observable_events = list()
        with open(obs_events_csv_path, "r", encoding="utf-8") as fp:
            csv_reader = initialize_csv_reader(fp)
            for row in csv_reader:
                oe = cls.from_csv_row(row)
                observable_events.append(oe)
        return observable_events
    
    @classmethod
    def _parse_data(cls, data: Dict):
        if type(data["event_data"]) == str:
            data["event_data"] = json.loads(data["event_data"])
            for k in cls.DATETIME_KEYS:
                if data.get(k) is not None:
                    data["event_data"][k] = data[k]
        return cls(**data)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> List["ObservableEvent"]:
        observable_events = list()
        for _, row in df.iterrows():
            oe = cls.from_dict(row.to_dict())
            observable_events.append(oe)
        return observable_events
    
    def to_dict(self) -> Dict:
        data = {
            **self.event_data,
            "derived_via": self.derived_via,
            "splade_score": self.splade_score,
            "ce_scores": self.ce_scores
        }
        return {
            "id": self.id,
            "start_date": self.start_date,
            "start_time": self.start_time,
            "end_date": self.end_date,
            "end_time": self.end_time,
            "event_type": self.event_type,
            "event_data": data,
        }
    
    def set_retrieval_result(self, derived_via: str, splade_score: float, ce_scores: List[float]) -> None:
        self.derived_via = str(derived_via)
        self.splade_score = str(splade_score)
        self.ce_scores = str(ce_scores)
    
    def is_structured(self):
        if isinstance(self.event_type, str):
            return ObservableEventType(self.event_type).is_structured()
        else:
            return self.event_type.is_structured()
    
    def __hash__(self):
        return hash(json.dumps(self.to_dict(), sort_keys=True))
