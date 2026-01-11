"""Base class for parameter rule, which creates the hyper-parameter space
for each primitive."""
from abc import ABC, abstractmethod


class ParameterRule(ABC):
    """
    The base class for parameter rules, which generate the configuration space of hyper-parameters for utils
    """

    @abstractmethod
    def predicate(self, task, step):
        pass

    @abstractmethod
    def apply(self, task, step):
        pass

    def __str__(self):
        return type(self).__name__


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    # preprocessing
    from .rules import preprocessing_rules

    preprocessing_rules.register_rules(rule_executor)

    # models
    from .rules import classification_rules
    from .rules import regression_rules

    classification_rules.register_rules(rule_executor)
    regression_rules.register_rules(rule_executor)
