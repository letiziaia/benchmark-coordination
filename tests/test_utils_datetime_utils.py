import pytest
from datetime import datetime

from benchmark_coordination.utils.datetime_utils import (
    timestamp_str_to_datetime,
    datetime_to_timestamp_str,
)


def test_timestamp_str_to_datetime():
    # Test parsing a valid datetime string
    assert timestamp_str_to_datetime("2022-01-01 12:00:00") == datetime(
        2022, 1, 1, 12, 0, 0
    )

    # Test parsing an invalid datetime string
    with pytest.raises(ValueError):
        timestamp_str_to_datetime("foobar")


def test_datetime_to_timestamp_str():
    # Test formatting a datetime object
    dt = datetime(2022, 1, 1, 12, 0, 0)
    assert datetime_to_timestamp_str(dt) == "2022-01-01 12:00:00"
