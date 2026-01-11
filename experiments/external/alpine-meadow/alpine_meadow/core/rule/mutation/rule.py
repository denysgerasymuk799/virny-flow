"""Rules for mutating the search space."""
from abc import ABC, abstractmethod


class MutationRule(ABC):
    """
    The base class for mutation rules, which mutate pipeline arms to create new data-specific pipeline arms
    """

    @abstractmethod
    def apply(self, search_space):
        pass

    def __str__(self):
        return type(self).__name__


def register_rules(rule_executor):
    from .rules import mutation_rules

    mutation_rules.register_rules(rule_executor)
