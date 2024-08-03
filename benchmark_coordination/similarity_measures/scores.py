import numpy as np


def cosine_similarity(vector1, vector2, return_normalized: bool = False) -> float:
    """
    Calculate the cosine similarity between two vectors.
    Cosine similarity is only defined for non-zero vectors.
    Note that the result is between -1 and 1.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :param return_normalized: Whether to return the normalized similarity (between 0 and 1).
        Default is False (return the raw similarity).
    :return: The cosine similarity between the two vectors.
    ----------------
    Example:
    >>> vector1 = [1, 2, 3]
    >>> vector2 = [-1, -2, -3]
    >>> cosine_similarity(vector1, vector1)
    1.0
    >>> cosine_similarity(vector1, vector1, return_normalized=True)
    1.0
    >>> cosine_similarity(vector1, vector2)
    -1.0
    >>> cosine_similarity(vector1, vector2, return_normalized=True)
    0.0
    >>> vector1 = [1, 0, 1]
    >>> vector2 = [1, 1, 1]
    >>> cosine_similarity(vector1, vector2)
    0.816496580927726
    >>> cosine_similarity(vector1, vector2, return_normalized=True)
    0.9082482904638631
    >>> vector1 = [1, 0, 0]
    >>> vector2 = [0, 0, 1]
    >>> cosine_similarity(vector1, vector2)
    0.0
    >>> cosine_similarity(vector1, vector2, return_normalized=True)
    0.5
    """
    # non-zero vectors
    assert (
        np.linalg.norm(vector1) > 0 and np.linalg.norm(vector2) > 0
    ), "Vectors must be non-zero"
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    if return_normalized:
        similarity = (similarity + 1) / 2
    return similarity


def jaccard_similarity(vector1, vector2) -> float:
    """
    Calculate the Jaccard similarity between two vectors.
    Works for non-binary vectors, but only accounts for the
    number of common elements (i.e. the vectors are treated as sets).
    :param vector1: The first vector.
    :param vector2: The second vector.
    :return: The Jaccard similarity between the two vectors.
    ----------------
    Example:
    >>> vector1 = [1, 2, 3]
    >>> vector2 = [2, 3, 4]
    >>> jaccard_similarity(vector1, vector1)
    1.0
    >>> jaccard_similarity(vector1, vector2)
    0.5
    >>> vector1 = [1, 0, 0]
    >>> vector2 = [0, 0, 1]
    >>> jaccard_similarity(vector1, vector2)
    1.0
    """
    set1 = set(vector1)
    set2 = set(vector2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union)
    return similarity


def jaccard_binary_similarity(vector1, vector2) -> float:
    """
    Calculate the Jaccard similarity between two binary vectors.
    Elements are considered common if they are the same and in the same position.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :return: The Jaccard similarity between the two vectors.
    ----------------
    Example:
    ----------------
    >>> vector1 = [1, 0, 1]
    >>> vector2 = [0, 0, 1]
    >>> jaccard_binary_similarity(vector1, vector1)
    1.0
    >>> jaccard_binary_similarity(vector1, vector2)
    0.0
    >>> vector1 = [1, 0, 0]
    >>> vector2 = [1, 0, 1]
    >>> jaccard_binary_similarity(vector1, vector2)
    0.5
    """
    # cast to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    # check that the vectors are binary
    assert all(
        np.logical_or(vector1 == 0, vector1 == 1)
    ), "Vector1 must be binary (0 or 1)"
    # jaccard similarity (binary, element-wise)
    intersection = np.logical_and(vector1, vector2)
    union = np.logical_or(vector1, vector2)
    similarity = intersection.sum() / float(union.sum())
    return similarity
