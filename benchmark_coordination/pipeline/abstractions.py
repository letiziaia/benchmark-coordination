from abc import ABC, abstractmethod
import pandas as pd


class IPipeline(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
