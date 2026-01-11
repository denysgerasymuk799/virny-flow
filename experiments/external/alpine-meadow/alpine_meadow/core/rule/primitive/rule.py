"""Primitive rules for adding new primitives into the pipelines."""
from abc import ABC, abstractmethod


class PrimitiveRule(ABC):
    """
    The base class for primitive rules, which construct the structures (steps)
    of pipeline arms by adding/changing primitive.
    """

    @abstractmethod
    def predicate(self, task):
        pass

    @abstractmethod
    def apply(self, search_space):
        pass

    def __str__(self):
        return type(self).__name__


def register_rules(rule_executor):
    """
    Register all primitive rules.
    :param rule_executor:
    :return:
    """

    # data
    from .rules.data import dataset_rules

    dataset_rules.register_rules(rule_executor)

    # feature
    from .rules.feature import preprocessing_rules

    preprocessing_rules.register_rules(rule_executor)

    # classification/regression/clustering models etc.
    from .rules.model import classification_rules
    from .rules.model import regression_rules

    classification_rules.register_rules(rule_executor)
    regression_rules.register_rules(rule_executor)

    # optimizations
    from .rules.optimization import optimization_rules

    optimization_rules.register_rules(rule_executor)
