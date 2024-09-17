from abc import ABC, abstractmethod
from typing import List, Optional, Union
import pandas as pd
import uuid


class IPipeline(ABC):
    def __init__(self, verbose: bool = False, pipeline_id: Optional[str] = None):
        self.verbose = verbose
        self._pipeline_id = (
            "pipeline" + str(uuid.uuid4()) if pipeline_id is None else pipeline_id
        )

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        pass

    @property
    def pipeline_id(self) -> str:
        """
        Returns the pipeline ID
        :return: str
        """
        return self._pipeline_id
