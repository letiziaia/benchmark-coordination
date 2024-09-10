import numpy as np
import pytest
from benchmark_coordination.similarity_calculator.scores import (
    cardinality_similarity,
    cosine_similarity,
    jaccard_similarity,
    jaccard_binary_similarity,
    ratcliff_obershelp_similarity,
    _longest_common_substring,
    _matching_characters,
)


@pytest.mark.parametrize(
    "vector1, vector2, expected_result",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.0),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 3.0),
        (np.array([1, 2, 3]), np.array([-1, -2, -3]), 0.0),
        (np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5]), 3.0),
    ],
)
def test_cardinality_similarity(vector1, vector2, expected_result):
    """
    Test the cardinality_similarity function.
    """
    result = cardinality_similarity(vector1, vector2)
    assert np.isclose(
        result, expected_result
    ), f"Expected cardinality_similarity {expected_result}, got {result}"


@pytest.mark.parametrize(
    "vector1, vector2, expected_result",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.9746318461970762),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 1.0),
        (np.array([1, 2, 3]), np.array([-1, -2, -3]), -1.0),
    ],
)
def test_cosine_similarity(vector1, vector2, expected_result):
    """
    Test the cosine_similarity function.
    """
    result = cosine_similarity(vector1, vector2)
    assert np.isclose(
        result, expected_result
    ), f"Expected cosine_similarity {expected_result}, got {result}"


@pytest.mark.parametrize(
    "vector1, vector2, expected_result",
    [
        (
            np.array(["a", "b", "c", "d", "e"]),
            np.array(["a", "b", "c", "d", "f"]),
            0.6666666,
        ),
        (np.array(["a", "b", "c", "d", "e"]), np.array(["a", "b", "c", "d", "e"]), 1.0),
        (np.array(["a", "b", "c", "d", "e"]), np.array(["f", "g", "h", "i", "j"]), 0.0),
    ],
)
def test_jaccard_similarity_text(vector1, vector2, expected_result):
    """
    Test the jaccard_similarity function on text.
    """
    result = jaccard_similarity(vector1, vector2)
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"


@pytest.mark.parametrize(
    "vector1, vector2, expected_result",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6]), 0.6666666),
        (np.array([1, 0, 0]), np.array([0, 0, 1]), 1.0),
        (np.array([1, 0, 0]), np.array([1, 1, 1]), 0.5),
        (np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0]), 0.0),
    ],
)
def test_jaccard_similarity(vector1, vector2, expected_result):
    """
    Test the jaccard_similarity function.
    """
    result = jaccard_similarity(vector1, vector2)
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_similarity {expected_result}, got {result}"


@pytest.mark.parametrize(
    "vector1, vector2, expected_result",
    [
        (np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1]), 0.5),
        (np.array([1, 0, 1, 0, 1]), np.array([1, 0, 1, 0, 1]), 1.0),
        (np.array([1, 0, 1, 0, 1]), np.array([0, 0, 0, 0, 0]), 0.0),
    ],
)
def test_jaccard_binary_similarity(vector1, vector2, expected_result):
    """
    Test the jaccard_binary_similarity function.
    """
    result = jaccard_binary_similarity(vector1, vector2)
    assert np.isclose(
        result, expected_result
    ), f"Expected jaccard_binary_similarity {expected_result}, got {result}"


@pytest.mark.parametrize(
    "s1, s2, expected_result",
    [
        ("abcdfgh", "abedfhr", "ab"),
        ("abcdfgh", "abcdfgh", "abcdfgh"),
        ("abcdfgh", "fghijkl", "fgh"),
    ],
)
def test__longest_common_substring(s1, s2, expected_result):
    """
    Test the _longest_common_substring function.
    """
    result = _longest_common_substring(s1, s2)
    assert result == expected_result, f"Expected {expected_result}, got {result}"


@pytest.mark.parametrize(
    "s1, s2, expected_result",
    [
        ("abcdfgh", "abedfhr", 5),
        ("abcdfgh", "abcdfgh", 7),
    ],
)
def test__matching_characters(s1, s2, expected_result):
    """
    Test the _matching_characters function.
    """
    result = _matching_characters(s1, s2)
    assert result == expected_result, f"Expected {expected_result}, got {result}"


@pytest.mark.parametrize(
    "vector1, vector2, expected_result",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 6]), 0.8),
        (np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6]), 0.8),
        (np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 0]), 0.0),
        (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]), 1.0),
    ],
)
def test_ratcliff_obershelp_similarity(vector1, vector2, expected_result):
    """
    Test the ratcliff_obershelp_similarity function.
    """
    result = ratcliff_obershelp_similarity(vector1, vector2)
    assert np.isclose(
        result, expected_result
    ), f"Expected ratcliff_obershelp_similarity {expected_result}, got {result}"
