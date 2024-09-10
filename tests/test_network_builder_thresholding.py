import pytest
import pandas as pd

from benchmark_coordination.network_builder.thresholding import (
    filter_edgelist,
    filter_edgelist_by_percentile,
)


@pytest.mark.parametrize(
    "threshold, comparison, expected_result",
    [
        (3.0, "<", pd.DataFrame({"weight": [1.0, 2.0]})),
        (3.0, ">=", pd.DataFrame({"weight": [3.0, 4.0, 5.0]})),
        (3.0, "==", pd.DataFrame({"weight": [3.0]})),
        (3.0, "<=", pd.DataFrame({"weight": [1.0, 2.0, 3.0]})),
        (3.0, ">", pd.DataFrame({"weight": [4.0, 5.0]})),
    ],
)
def test_filter_edgelist(threshold, comparison, expected_result):
    """
    Test the filter_edgelist function.
    """
    df = pd.DataFrame({"weight": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = filter_edgelist(df, "weight", threshold, comparison)

    assert list(result["weight"]) == list(
        expected_result["weight"]
    ), f"Expected {expected_result}, got {result}"


# Test filter_edgelist_by_percentile function
@pytest.mark.parametrize(
    "percentile, comparison, expected_result",
    [
        (50, "<", pd.DataFrame({"weight": [1.0, 2.0]})),
        (50, ">=", pd.DataFrame({"weight": [3.0, 4.0, 5.0]})),
        (50, "==", pd.DataFrame({"weight": [3.0]})),
        (50, "<=", pd.DataFrame({"weight": [1.0, 2.0, 3.0]})),
        (50, ">", pd.DataFrame({"weight": [4.0, 5.0]})),
    ],
)
def test_filter_edgelist_by_percentile(percentile, comparison, expected_result):
    """
    Test the filter_edgelist_by_percentile function.
    """
    df = pd.DataFrame({"weight": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = filter_edgelist_by_percentile(df, "weight", percentile, comparison)

    assert list(result["weight"]) == list(
        expected_result["weight"]
    ), f"Expected {expected_result}, got {result}"
