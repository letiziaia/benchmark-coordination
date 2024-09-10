import pandas as pd
import multiprocessing

from benchmark_coordination.pipeline.abstractions import IPipeline
from benchmark_coordination.pipeline.pipeline import Pipeline


class MultiPipeline(IPipeline):
    """
    MultiPipeline is a class that allows
    you to run multiple pipelines in parallel.
    """

    def __init__(self, pipelines: list[Pipeline], verbose: bool = False):
        super().__init__(verbose=verbose)
        self.pipelines = pipelines

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline on the data, running each pipeline in parallel.

        Parameters
        ----------
        data : pd.DataFrame
            The data to run through the pipeline.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        with multiprocessing.Pool(len(self.pipelines)) as pool:
            # TODO: modfidy Pipeline to add a column to the output dataframe
            # with the name of the pipeline that generated the row
            results = pool.map(lambda pipeline: pipeline.fit(data), self.pipelines)
        return pd.concat(results)
