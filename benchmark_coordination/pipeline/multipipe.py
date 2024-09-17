import pandas as pd
from typing import List, Optional, Tuple
import multiprocessing

from benchmark_coordination.pipeline.abstractions import IPipeline
from benchmark_coordination.pipeline.pipeline import Pipeline


class MultiPipeline(IPipeline):
    """
    MultiPipeline is a class that allows
    you to run multiple pipelines in parallel.

    Parameters
    ----------
    pipelines : list of Pipelines

    multipipeline_id : str, optional
        The ID of the multipipeline. If not provided, a random ID will be generated.

    verbose : bool, optional
        Whether to log the steps of the pipeline. Default is False.

    Attributes
    ----------
    named_pipelines : Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    """

    def __init__(
        self,
        pipelines: List[Pipeline],
        multipipeline_id: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose, pipeline_id=multipipeline_id)
        self._pipelines = pipelines

    @property
    def pipelines(self) -> List[Pipeline]:
        """
        Returns the pipelines
        :return: List[Pipeline]
        """
        return self._pipelines

    @property
    def named_pipelines(self) -> List[str]:
        """
        Returns the names of the pipelines
        :return: List[str]
        """
        return [pipeline.pipeline_id for pipeline in self.pipelines]

    def _pipeline_ids(self) -> List[str]:
        """
        Returns the IDs of the pipelines, in the order they were added
        :return: List[str]
        """
        return [pipeline.pipeline_id for pipeline in self.pipelines]

    def _fit_pipeline(
        self, pipeline: Pipeline, data: pd.DataFrame
    ) -> Tuple[str, pd.DataFrame]:
        """
        Fit a pipeline on the data
        :param pipeline: Pipeline, the pipeline to fit
        :param data: pd.DataFrame, the data to fit the pipeline on
        :return: Tuple[str, pd.DataFrame], the ID of the pipeline and the transformed data
        """
        return pipeline.pipeline_id, pipeline.fit(data)

    def fit(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Fit each pipeline on the data, running each pipeline in parallel.

        Parameters
        ----------
        data : pd.DataFrame
            The data to run through the pipeline.

        Returns
        -------
        List[pd.DataFrame]
            The transformed data.
        """
        n_processes = max(multiprocessing.cpu_count() - 1, len(self.pipelines))
        with multiprocessing.Pool(n_processes) as pool:
            results = pool.starmap(
                self._fit_pipeline, [(pipeline, data) for pipeline in self.pipelines]
            )
        # return the data in the order of the pipelines
        df_dict = {pipeline_id: result for pipeline_id, result in results}
        return [df_dict[pipeline_id] for pipeline_id in self._pipeline_ids()]

    def __len__(self) -> int:
        """
        Returns the size of the MultiPipeline
        :return: int, the number of pipelines
        """
        return len(self.pipelines)

    def __getitem__(self, index: int) -> Pipeline:
        """
        Returns the pipeline at the given index
        :param index: int, the index of the pipeline
        :return: Pipeline
        """
        return self.pipelines[index]

    def __iter__(self):
        """
        Returns an iterator over the pipelines
        :return: Iterator
        """
        return iter(self.pipelines)

    def __repr__(self) -> str:
        return f"MultiPipeline(steps={self.pipelines})"
