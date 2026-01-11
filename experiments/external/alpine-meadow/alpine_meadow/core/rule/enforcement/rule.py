"""Base class for enforcement rule."""
from abc import ABC, abstractmethod


class EnforcementRule(ABC):
    """
    The base class for enforcement rules, which filter out pipeline arms based on some conditions
    """

    def __init__(self):
        pass

    @staticmethod
    def predicate(task):  # pylint: disable=unused-argument
        return True

    @staticmethod
    @abstractmethod
    def enforce(task, pipeline_arm):
        pass


def register_rules(rule_executor):
    from .rules import basic_rules

    basic_rules.register_rules(rule_executor)
