from typing import List, Any, Dict

from reqap.classes.computed_event import ComputedEvent
from reqap.library.library import to_json_dict


class QUExecutionResult:
    def __init__(self, computed_items: List[ComputedEvent] | List["Group"], result: Any=None):
        self.computed_items = computed_items
        self.result = result
        self.time_taken = None

    def to_dict(self) -> Dict:
        res = {
            "computed_items": [item.to_dict() for item in self.computed_items],
            "result": str(self.result),
            "time_taken": self.time_taken
        }
        return res
    
    def set_timing(self, time_taken: float):
        self.time_taken = time_taken
    

class Group(QUExecutionResult):
    def __init__(self, computed_items: List[ComputedEvent], attributes: Dict, result: Any=None):
        self.computed_items = computed_items
        self.attributes = attributes
        self.result = result
        self.time_taken = None

    def to_dict(self) -> Dict:
        return {
            **super().to_dict(),
            "attributes": to_json_dict(self.attributes),
        }
       
    def __getattr__(self, attr_name: str):
        if attr_name == "attributes":
            return self.__dict__.get("attributes", {})
        
        if "attributes" in self.__dict__ and attr_name in self.__dict__["attributes"]:
            return self.__dict__["attributes"][attr_name]
        
        raise AttributeError(f"'ComputedEvent' object {self} has no attribute '{attr_name}'")
    
    def __getitem__(self, key):
        return self.__getattr__(key)


class GroupByResult(QUExecutionResult):
    def __init__(self, groups: List[Group]):
        self.groups = groups
        self.time_taken = None

    def to_dict(self) -> Dict:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "time_taken": self.time_taken
        }
