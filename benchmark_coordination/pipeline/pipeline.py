from functools import partial
from typing import Dict, List, Optional, Tuple
import pandas as pd

from benchmark_coordination.pipeline.abstractions import IPipeline
from benchmark_coordination.utils.logging import logger


class Pipeline(IPipeline):
    """
    A sequence of data transformers.

    `Pipeline` allows you to sequentially apply a list of transformers to
    preprocess the data.

    The purpose of the pipeline is to assemble several steps with different parameters.

    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator, dictionary of params) tuples that are to be chained in
        sequential order. All steps must return a pd.DataFrame.
    pipeline_id : str, optional
        The ID of the pipeline. If not provided, a random ID will be generated.
    verbose : bool, optional
        Whether to log the steps of the pipeline. Default is False.

    Attributes
    ----------
    named_steps : Dictionary-like object. Read-only attribute to access any step parameter
        by name. Keys are step names and values are steps parameters.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "author_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    ...         "trace": ["a", "b", "c", "b", "c", "d", "e", "f", "d", "e"],
    ...         "timestamp": ["2024-01-01 08:00:00", "2024-01-01 08:01:00",\
    ...             "2024-01-01 08:02:00", "2024-01-01 08:03:00", "2024-01-01 08:04:00",\
    ...             "2024-01-01 08:05:00", "2024-01-01 08:06:00", "2024-01-01 08:07:00",\
    ...             "2024-01-01 08:08:00", "2024-01-01 08:09:00"],
    ...     }
    ... )
    >>> df["timestamp"] = pd.to_datetime(df["timestamp"])
    >>> from benchmark_coordination.network_builder.similarity_net import build_similarity_network
    >>> from benchmark_coordination.network_builder.thresholding import filter_edgelist
    >>> pipe = Pipeline(
    ...     steps=[
    ...         ("build_similarity_network", build_similarity_network, {"score": "jaccard", "symmetric": True}),
    ...         ("filter_edgelist", filter_edgelist, {"column_name": "similarity", "threshold": 0.3, "comparison": ">="}),  
    ...     ],
    ... )
    >>> pipe.fit(df)
         source  target  similarity
    2        1       4    0.333333
    5        2       4    0.333333
    6        2       5    0.333333
    8        3       5    0.333333
    """

    def __init__(
        self,
        steps: List[Tuple],
        *,
        pipeline_id: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose, pipeline_id=pipeline_id)

        self._steps = steps

    @property
    def steps(self) -> List[Tuple]:
        """
        Returns the steps of the Pipeline
        :return: List[Tuple]
        """
        return self._steps

    @property
    def named_steps(self) -> Dict[str, partial]:
        """
        Returns the named steps of the Pipeline
        :return: Dict[str, partial]
        """
        named_steps = {}
        for name, step, params in self.steps:
            named_steps[name] = partial(step, **params)
        return named_steps

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline on the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data to run through the pipeline.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        with logger.contextualize(pipeline=self._pipeline_id):
            for name, step, params in self.steps:
                data = step(data, **params)
                if self.verbose:
                    logger.info(f"Step {name}: completed.")
            return data

    def __len__(self) -> int:
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    def __repr__(self) -> str:
        return f"Pipeline(steps={self.steps})"
