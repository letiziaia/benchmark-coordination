import numpy as np

from benchmark_coordination.similarity_measures.scores import (
    cosine_similarity,
    jaccard_similarity,
    jaccard_binary_similarity,
    ratcliff_obershelp_similarity,
    _longest_common_substring,
    _matching_characters,
)


def test_cosine_similarity():
    """
    Test the cosine_similarity function.
    """
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    result = cosine_similarity(vector1, vector2)
    expected_result = 0.9746318461970762
    assert np.isclose(
        result, expected_result
    ), f"Expected cosine_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 2, 3])
    vector2 = np.array([1, 2, 3])
    result = cosine_similarity(vector1, vector2)
    expected_result = 1.0
    assert np.isclose(
        result, expected_result
    ), f"Expected cosine_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 2, 3])
    vector2 = np.array([-1, -2, -3])
    result = cosine_similarity(vector1, vector2)
    expected_result = -1.0
    assert np.isclose(
        result, expected_result
    ), f"Expected cosine_similarity {expected_result}, got {result}"


def test_jaccard_similarity_text():
    """
    Test the jaccard_similarity function on text.
    """
    vector1 = np.array(["a", "b", "c", "d", "e"])
    vector2 = np.array(["a", "b", "c", "d", "f"])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 0.6666666
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"

    vector1 = np.array(["a", "b", "c", "d", "e"])
    vector2 = np.array(["a", "b", "c", "d", "e"])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 1.0
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"

    vector1 = np.array(["a", "b", "c", "d", "e"])
    vector2 = np.array(["f", "g", "h", "i", "j"])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 0.0
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"


def test_jaccard_similarity():
    """
    Test the jaccard_similarity function.
    """
    vector1 = np.array([1, 2, 3, 4, 5])
    vector2 = np.array([2, 3, 4, 5, 6])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 0.6666666
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 0, 1])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 1.0
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 0, 0])
    vector2 = np.array([1, 1, 1])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 0.5
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 1, 1, 1, 1])
    vector2 = np.array([0, 0, 0, 0, 0])
    result = jaccard_similarity(vector1, vector2)
    expected_result = 0.0
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"


def test_jaccard_binary_similarity():
    """
    Test the jaccard_binary_similarity function.
    """
    vector1 = np.array([1, 1, 0, 0, 1])
    vector2 = np.array([1, 0, 1, 0, 1])
    result = jaccard_binary_similarity(vector1, vector2)
    expected_result = 0.5
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_binary_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 0, 1, 0, 1])
    vector2 = np.array([1, 0, 1, 0, 1])
    result = jaccard_binary_similarity(vector1, vector2)
    expected_result = 1.0
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_binary_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 0, 1, 0, 1])
    vector2 = np.array([0, 0, 0, 0, 0])
    result = jaccard_binary_similarity(vector1, vector2)
    expected_result = 0.0
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_binary_similarity {expected_result}, got {result}"


def test__longest_common_substring():
    """
    Test the _longest_common_substring function.
    """
    s1 = "abcdfgh"
    s2 = "abedfhr"
    result = _longest_common_substring(s1, s2)
    expected_result = "ab"
    assert result == expected_result, f"Expected {expected_result}, got {result}"

    s1 = "abcdfgh"
    s2 = "abcdfgh"
    result = _longest_common_substring(s1, s2)
    expected_result = "abcdfgh"
    assert result == expected_result, f"Expected {expected_result}, got {result}"

    s1 = "abcdfgh"
    s2 = "fghijkl"
    result = _longest_common_substring(s1, s2)
    expected_result = "fgh"
    assert result == expected_result, f"Expected {expected_result}, got {result}"


def test__matching_characters():
    """
    Test the _matching_characters function.
    """
    s1 = "abcdfgh"
    s2 = "abedfhr"
    result = _matching_characters(s1, s2)
    expected_result = 5
    assert result == expected_result, f"Expected {expected_result}, got {result}"

    s1 = "abcdfgh"
    s2 = "abcdfgh"
    result = _matching_characters(s1, s2)
    expected_result = 7
    assert result == expected_result, f"Expected {expected_result}, got {result}"


def test_ratcliff_obershelp_similarity():
    """
    Test the ratcliff_obershelp_similarity function.
    """
    vector1 = np.array([1, 2, 3, 4, 5])
    vector2 = np.array([1, 2, 3, 4, 6])
    result = ratcliff_obershelp_similarity(vector1, vector2)
    expected_result = 0.8
    assert np.isclose(
        result, expected_result
    ), f"Expected ratcliff_obershelp_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 2, 3, 4, 5])
    vector2 = np.array([2, 3, 4, 5, 6])
    result = ratcliff_obershelp_similarity(vector1, vector2)
    expected_result = 0.8
    assert np.isclose(
        result, expected_result
    ), f"Expected ratcliff_obershelp_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 2, 3, 4, 5])
    vector2 = np.array([0, 0, 0, 0, 0])
    result = ratcliff_obershelp_similarity(vector1, vector2)
    expected_result = 0.0
    assert np.isclose(
        result, expected_result
    ), f"Expected ratcliff_obershelp_similarity {expected_result}, got {result}"

    vector1 = np.array([1, 2, 3, 4, 5])
    vector2 = np.array([1, 2, 3, 4, 5])
    result = ratcliff_obershelp_similarity(vector1, vector2)
    expected_result = 1.0
    assert np.isclose(
        result, expected_result
    ), f"Expected ratcliff_obershelp_similarity {expected_result}, got {result}"
