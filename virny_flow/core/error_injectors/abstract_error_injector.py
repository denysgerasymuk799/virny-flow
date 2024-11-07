import pandas as pd
from abc import ABCMeta, abstractmethod


class AbstractErrorInjector(metaclass=ABCMeta):
    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def fit(self, df: pd.DataFrame, columns_with_nulls: list, null_percentage: float, condition: str = None):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, columns_with_nulls: list, null_percentage: float, condition: str = None):
        pass
