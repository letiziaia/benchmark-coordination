import pytest
from datetime import datetime

from benchmark_coordination.utils.datetime_utils import (
    timestamp_str_to_datetime,
    datetime_to_timestamp_str,
)


def test_timestamp_str_to_datetime():
    """
    Test the timestamp_str_to_datetime function.
    """
    # Test parsing a valid datetime string
    input_str = "2022-01-01 12:00:00"
    result = timestamp_str_to_datetime(input_str)
    expected_dt = datetime(2022, 1, 1, 12, 0, 0)
    assert (
        result == expected_dt
    ), f"Expected {expected_dt} but got {result} for input {input_str}"

    # Test parsing an invalid datetime string
    with pytest.raises(ValueError):
        timestamp_str_to_datetime("foobar")


def test_datetime_to_timestamp_str():
    """
    Test the datetime_to_timestamp_str function.
    """
    # Test formatting a datetime object
    dt = datetime(2022, 1, 1, 12, 0, 0)
    result = datetime_to_timestamp_str(dt)
    expected_str = "2022-01-01 12:00:00"
    assert (
        result == expected_str
    ), f"Expected {expected_str} as formatted string but got {result}"
