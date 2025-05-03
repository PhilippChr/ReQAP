
import itertools
from typing import Dict, List

from reqap.qu.operator_tree import OperatorTree


class QUTree:
    """
    QU tree is the outcome from the QU stage.
    The QU tree may provide multiple options for each
    QU step, which means that a single QU tree can be
    interpreted as multiple Operator Trees.
    """
    def __init__(self, qu_input: str, qu_branch_input: str):
        self.qu_branch_input = qu_branch_input  # previous QU call
        self.qu_input = qu_input  # current QU call
        self.child_nodes: List["QUTreeBranch"] = []

    def add_branch(self, qu_tree_branch: "QUTreeBranch"):
        self.child_nodes.append(qu_tree_branch)

    def to_operator_trees(self) -> List[OperatorTree]:
        """
        Function that derives all possible Operator Trees for the QU tree.
        """
        operator_tree_dicts = self._to_operator_tree_dicts()
        return [OperatorTree(operator_tree_dict) for operator_tree_dict in operator_tree_dicts]

    def _to_operator_tree_dicts(self) -> List[Dict]:
        ## check stopping criteria (reached a leaf node)
        if not self.child_nodes:  
            return [{
                "qu_branch_input": self.qu_branch_input,
                "qu_input": self.qu_input,
                "childs": []
            }]

        ## obtain plans for each branch 
        operator_tree_branches = list()
        for child in self.child_nodes:
            operator_trees = list()
            for _, qu_option in enumerate(child.qu_options):
                res = qu_option._to_operator_tree_dicts()
                operator_trees += res
            operator_tree_branches.append(operator_trees)

        ## cartesian product of Operator Trees for branches
        operator_tree_dicts = list()
        for operator_tree_childs in itertools.product(*operator_tree_branches):
            operator_tree_dict = {
                "qu_branch_input": self.qu_branch_input,
                "qu_input": self.qu_input,
                "childs": list(operator_tree_childs)
            }
            operator_tree_dicts.append(operator_tree_dict)
        return operator_tree_dicts


class QUTreeBranch:
    """
    Branches are used to represent multiple QU calls
    at one tree node.
    """
    def __init__(self, qu_options: List[QUTree]):
        self.qu_options: List[QUTree] = qu_options
