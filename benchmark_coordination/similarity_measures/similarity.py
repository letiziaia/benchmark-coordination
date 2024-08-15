from typing import Callable, Dict, List

import benchmark_coordination.similarity_measures.scores as scores
from benchmark_coordination.types.similarity import Sim


class SimilarityCalculator:
    """
    A class to calculate similarity between two vectors using different similarity measures.
    ----------------
    Example:
    ----------------
    >>> similarity_calculator = SimilarityCalculator("cosine")
    >>> vector1 = [1, 2, 3]
    >>> vector2 = [4, 5, 6]
    >>> similarity_calculator.calculate_similarity(vector1, vector2)
    0.9746318461970762
    """

    def __init__(self, similarity_score: Sim) -> None:
        """
        Initialize the SimilarityCalculator object with the similarity score to be used.
        :param similarity_score: str, the similarity score to be used.
            If the similarity score is not one of the following: "cosine", "jaccard",
            "ratcliff-obershelp", a ValueError will be raised.
        :return: None
        """
        self.similarity_score = similarity_score

        self.similarity_measures: Dict[str, Callable] = {
            "cosine": scores.cosine_similarity,
            "jaccard": scores.jaccard_similarity,
            "ratcliff-obershelp": scores.ratcliff_obershelp_similarity,
        }

        if self.similarity_score not in self.similarity_measures:
            raise ValueError("Invalid similarity score")

    def calculate_similarity(self, vector1: List, vector2: List, **kwargs) -> float:
        """
        Calculate the similarity between two vectors using the specified similarity score.
        :param vector1: list, the first vector.
        :param vector2: list, the second vector.
        :param kwargs: dict, additional arguments to be passed to the similarity measure function.
        :return: float, the similarity score between the two vectors.
        """
        return self.similarity_measures[self.similarity_score](
            vector1, vector2, **kwargs
        )
