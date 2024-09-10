import pandas as pd

from benchmark_coordination.similarity_calculator.calculator import SimilarityCalculator
from benchmark_coordination.types.similarity_types import SimilarityMeasure


def build_similarity_network(
    dataframe: pd.DataFrame, score: SimilarityMeasure, symmetric: bool = True
) -> pd.DataFrame:
    """
    Build a similarity network from a dataframe using the specified similarity score.
    :param dataframe: pd.DataFrame, the dataframe containing the data to be used to build the similarity network.
        The dataframe should have column 'author_id' containing the source nodes, and column 'trace'
        containing the activity trace to compare.
    :param score: str, the similarity score to be used.
        If the similarity score is not one from SimilarityMeasure, a ValueError will be raised.
    :param symmetric: bool, whether the similarity network should be symmetric.
    :return: pd.DataFrame, the edge list for the similarity network.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "author_id": [1, 1, 2, 2],
    ...     "trace": ["A", "B", "A", "C"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> build_similarity_network(df, "jaccard")
        source  target  similarity
    0       1       2    0.333333
    """
    sim = SimilarityCalculator(similarity_score=score)
    users = sorted(dataframe["author_id"].unique())
    similarity_network = []
    for u1 in users:
        for u2 in users:
            if u1 == u2:
                continue
            if symmetric and u1 > u2:
                continue
            u1_data = dataframe[dataframe["author_id"] == u1]["trace"]
            u2_data = dataframe[dataframe["author_id"] == u2]["trace"]
            s = sim.calculate_similarity(
                vector1=u1_data,
                vector2=u2_data,
            )
            similarity_network.append({"source": u1, "target": u2, "similarity": s})

    return pd.DataFrame(similarity_network)
