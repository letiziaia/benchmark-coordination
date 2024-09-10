from typing import Any, Literal
from numpy.typing import NDArray

import numpy as np


def cardinality_similarity(vector1: NDArray[Any], vector2: NDArray[Any]) -> float:
    """
    Calculate the cardinality similarity between two vectors.
    Cardinality similarity is the number of common elements between the two vectors.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :return: The cardinality similarity between the two vectors.
    ----------------
    Example:
    >>> vector1 = [1, 2, 3]
    >>> vector2 = [2, 3, 4]
    >>> cardinality_similarity(vector1, vector1)
    3
    >>> cardinality_similarity(vector1, vector2)
    2
    """
    set1 = set(vector1)
    set2 = set(vector2)
    similarity = len(set1.intersection(set2))
    return similarity


def cosine_similarity(
    vector1: NDArray[Any],
    vector2: NDArray[Any],
    return_normalized: bool = False,
    coalesce: Literal["pad", "cut", "raise"] = "raise",
) -> float:
    """
    Calculate the cosine similarity between two vectors.
    Cosine similarity is only defined for non-zero vectors.
    Note that the result is between -1 and 1.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :param return_normalized: Whether to return the normalized similarity (between 0 and 1).
        Default is False (return the raw similarity).
    :param coalesce: one of 'pad', 'cut', 'raise'. If 'pad', the shorter vector is padded with 0.
        If 'cut', the longer vector is truncated to the length of the shorter vector.
        If 'raise', an error is raised if the vectors have different lengths.
        Default is 'raise'.
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
    # check that the vectors have the same length
    same_length = len(vector1) == len(vector2)
    if not same_length and coalesce == "pad":
        # pad the shorter vector with 0
        max_length = max(len(vector1), len(vector2))
        # pad the end of the vectors
        vector1 = np.pad(vector1, (0, max_length - len(vector1)), constant_values=0)
        vector2 = np.pad(vector2, (0, max_length - len(vector2)), constant_values=0)
        print(vector1, vector2)
    if not same_length and coalesce == "cut":
        # cut the longer vector
        min_length = min(len(vector1), len(vector2))
        vector1 = vector1[:min_length]
        vector2 = vector2[:min_length]

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    if return_normalized:
        similarity = (similarity + 1) / 2
    return similarity


def jaccard_similarity(vector1: NDArray[Any], vector2: NDArray[Any]) -> float:
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


def jaccard_binary_similarity(vector1: NDArray[Any], vector2: NDArray[Any]) -> float:
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


def ratcliff_obershelp_similarity(
    vector1: NDArray[Any], vector2: NDArray[Any]
) -> float:
    """
    Calculate the Ratcliff/Obershelp similarity between two vectors.
    The Ratcliff/Obershelp similarity is twice the number of matching characters
    divided by the total number of characters in the two strings.
    Matching characters are those in the longest common substring plus,
    recursively, matching characters in the unmatched region on either side
    of the longest common substring.
    :param vector1: The first vector.
    :param vector2: The second vector.
    :return: The Ratcliff/Obershelp similarity between the two vectors.
    ----------------
    Example:
    ----------------
    >>> vector1 = [1, 2, 3]
    >>> vector2 = [2, 3, 4]
    >>> ratcliff_obershelp_similarity(vector1, vector1)
    0.6666666666666666
    """
    total_elements = len(vector1) + len(vector2)
    s1 = "".join(map(str, vector1))
    s2 = "".join(map(str, vector2))
    matching_chars = _matching_characters(s1, s2)
    return 2 * matching_chars / total_elements if total_elements > 0 else 0.0


def _longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common subsequence between two vectors.
    :param s1: First string.
    :param s2: Second string.
    :return: The longest common subsequence.
    ----------------
    Example:
    ----------------
    >>> s1 = "abcde"
    >>> s2 = "bcdf"
    >>> _longest_common_substring(s1, s2)
    'bcd'
    """
    m = len(s1)
    n = len(s2)
    max_len = 0
    ending_index = m
    length = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                length[i][j] = length[i - 1][j - 1] + 1
                if length[i][j] > max_len:
                    max_len = length[i][j]
                    ending_index = i
            else:
                length[i][j] = 0

    return s1[ending_index - max_len : ending_index]


def _matching_characters(s1: str, s2: str) -> int:
    """
    Compute the number of matching characters between two sequences based on
    the longest common subsequence and recursively matching characters in the
    unmatched regions.
    :param s1: First string.
    :param s2: Second string.
    :return: Number of matching characters.
    ----------------
    Example:
    ----------------
    >>> s1 = "123"
    >>> s2 = "234"
    >>> _matching_characters(s1, s2)
    2
    """
    if not s1 or not s2:
        return 0

    lcs = _longest_common_substring(s1, s2)
    if not lcs:
        return 0

    lcs_len = len(lcs)
    lcs_start_s1 = s1.find(lcs)
    lcs_start_s2 = s2.find(lcs)

    # Recursively count matching characters in the unmatched regions
    left_match = _matching_characters(s1[:lcs_start_s1], s2[:lcs_start_s2])
    right_match = _matching_characters(
        s1[lcs_start_s1 + lcs_len :], s2[lcs_start_s2 + lcs_len :]
    )
    print(lcs_len, left_match, right_match)

    return lcs_len + left_match + right_match
