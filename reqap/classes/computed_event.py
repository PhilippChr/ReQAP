from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import date, time

from reqap.classes.observable_event import ObservableEvent
from reqap.library.library import to_json_dict


@dataclass
class ComputedEvent:
    EXTRACT_ATTRIBUTES = ["start_date", "start_time", "end_date", "end_time"]
    EXTRACT_TYPES = [date.fromisoformat, time.fromisoformat, date.fromisoformat, time.fromisoformat]

    attributes: Dict
    interpretation: str
    observable_event: Optional[ObservableEvent] = None
    observable_events: Optional[List[ObservableEvent]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        serialized = {}
        for field in self.__dataclass_fields__:
            if field == "attributes":
                serialized[field] = to_json_dict(getattr(self, "attributes", {}))
            else:
                value = getattr(self, field)
                serialized[field] = to_json_dict(value)
        return serialized
    
    def get_observable_events(self) -> List[ObservableEvent]:
        if self.observable_event is None:
            return self.observable_events
        else:
            return [self.observable_event]
    
    def __getattr__(self, attr_name: str) -> Any:
        if attr_name == "attributes":
            return self.__dict__.get("attributes", {})
    
        if "attributes" in self.__dict__ and attr_name in self.__dict__["attributes"]:
            return self.__dict__["attributes"][attr_name]
        
        raise AttributeError(f"'ComputedEvent' object {self} has no attribute '{attr_name}'")

    def __getitem__(self, attr_name: str):
        return self.__getattr__(attr_name)
    
    def __copy__(self) -> "ComputedEvent":
        new = type(self)(**self.__dict__)
        return new
        
    def __len__(self):
        if self.observable_events:
            return len(self.observable_events)
        return 1
    
    def __hash__(self):
        return hash(str(self.to_dict()))
