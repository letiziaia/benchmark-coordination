import numpy as np


def cosine_similarity(vector1, vector2) -> float:
    """
    Calculate the cosine similarity between two vectors.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :return: The cosine similarity between the two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity


def jaccard_similarity(vector1, vector2) -> float:
    """
    Calculate the Jaccard similarity between two vectors.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :return: The Jaccard similarity between the two vectors.
    """
    intersection = np.logical_and(vector1, vector2)
    union = np.logical_or(vector1, vector2)
    similarity = intersection.sum() / float(union.sum())
    return similarity
