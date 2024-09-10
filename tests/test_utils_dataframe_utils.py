import pytest
import pandas as pd
from datetime import datetime

from benchmark_coordination.utils.dataframe_utils import (
    contains_columns,
    cast_columns_to_datetime,
    cast_columns_to_str,
)


@pytest.mark.parametrize(
    "df, columns, expected",
    [
        (pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), ["A", "B"], True),
        (pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), ["A", "C"], False),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
            ["A", "B", "C"],
            True,
        ),
        (pd.DataFrame({"A": [1, 2, 3]}), ["A"], True),
        (pd.DataFrame({"A": [1, 2, 3]}), ["B"], False),
    ],
)
def test_contains_columns(df, columns, expected):
    """
    Test the contains_columns function.
    """
    result = contains_columns(df, columns)
    assert result == expected, f"Expected {expected} but got {result}"


def test_cast_columns_to_datetime():
    """
    Test the cast_columns_to_datetime function.
    """
    df = pd.DataFrame(
        {
            "date1": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "date2": ["2022-02-01", "2022-02-02", "2022-02-03"],
        }
    )
    columns = ["date1", "date2"]
    result = cast_columns_to_datetime(df, columns)
    for column in columns:
        assert pd.api.types.is_datetime64_any_dtype(result[column]), (
            f"Expected column {column} to be of type datetime64 but got "
            f"{result[column].dtype}"
        )


def test_cast_columns_to_datetime_raise():
    """
    Test that cast_columns_to_datetime raises a ValueError when columns are not found.
    """
    df = pd.DataFrame({"A": [1, 2, 3]})
    columns = ["B"]
    with pytest.raises(ValueError):
        cast_columns_to_datetime(df, columns)


def test_cast_columns_to_str():
    """
    Test the cast_columns_to_str function.
    """
    df = pd.DataFrame(
        {
            "date1": [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
            "date2": [datetime(2022, 2, 1), datetime(2022, 2, 2), datetime(2022, 2, 3)],
        }
    )
    columns = ["date1", "date2"]
    result = cast_columns_to_str(df, columns)
    for column in columns:
        assert pd.api.types.is_string_dtype(result[column]), (
            f"Expected column {column} to be of type string but got "
            f"{result[column].dtype}"
        )


def test_cast_columns_to_str_raise():
    """
    Test that cast_columns_to_str raises a ValueError when columns are not found.
    """
    df = pd.DataFrame({"A": [1, 2, 3]})
    columns = ["B"]
    with pytest.raises(ValueError):
        cast_columns_to_str(df, columns)
