import pandas as pd

from benchmark_coordination.similarity_measures.similarity import SimilarityCalculator


def build_similarity_network(
    dataframe: pd.DataFrame, score: str, simmetric: bool = True
) -> pd.DataFrame:
    """
    Build a similarity network from a dataframe using the specified similarity score.
    :param dataframe: pd.DataFrame, the dataframe containing the data to be used to build the similarity network.
        The dataframe should have column 'author_id' containing the source nodes, and column 'trace'
        containing the activity trace to compare.
    :param score: str, the similarity score to be used.
        If the similarity score is not one of the following: "cosine", "jaccard",
        a ValueError will be raised.
    :param simmetric: bool, whether the similarity network should be simmetric.
    :return: pd.DataFrame, the edge list for the similarity network.
    """
    sim = SimilarityCalculator(similarity_score=score)
    users = sorted(dataframe["author_id"].unique())
    similarity_network = []
    for u1 in users:
        for u2 in users:
            if u1 == u2:
                continue
            if simmetric and u1 > u2:
                continue
            u1_data = dataframe[dataframe["author_id"] == u1]["trace"]
            u2_data = dataframe[dataframe["author_id"] == u2]["trace"]
            s = sim.calculate_similarity(
                vector1=u1_data,
                vector2=u2_data,
            )
            similarity_network.append({"source": u1, "target": u2, "similarity": s})

    return pd.DataFrame(similarity_network)
