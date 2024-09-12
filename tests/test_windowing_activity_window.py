import pytest
import pandas as pd
from benchmark_coordination.windowing.activity_window import (
    filter_dataframe,
    slide_dataframe,
)


@pytest.fixture
def sample_dataframe():
    data = {
        "timestamp": [
            "2022-09-01 00:00:00",
            "2022-09-01 00:01:00",
            "2022-09-01 00:02:00",
            "2022-09-01 00:03:00",
            "2022-09-01 00:04:00",
        ],
        "value": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


def test_filter_dataframe(sample_dataframe):
    """
    Test the filter_dataframe function.
    """
    start_idx = 1
    end_idx = 3
    filtered_df = filter_dataframe(sample_dataframe, start_idx, end_idx)
    assert (
        len(filtered_df) == end_idx - start_idx
    ), f"Expected {end_idx - start_idx} rows but got {len(filtered_df)}"
    assert filtered_df["value"].tolist() == [
        2,
        3,
    ], f"Expected [2, 3] but got {filtered_df['value'].tolist()}"


def test_slide_dataframe(sample_dataframe):
    """
    Test the slide_dataframe function.
    """
    window_size = 2
    step_size = 1
    windows = list(
        slide_dataframe(
            sample_dataframe,
            window_size,
            step_size,
        )
    )
    assert len(windows) == 5, f"Expected 5 windows but got {len(windows)}"
    assert all(
        isinstance(window, pd.DataFrame) for window in windows
    ), "All windows should be of type pd.DataFrame"
    assert windows[0]["value"].tolist() == [
        1,
        2,
    ], f"Expected [1, 2] but got {windows[0]['value'].tolist()}"

    windows = list(
        slide_dataframe(
            sample_dataframe,
            window_size,
            step_size,
            start_idx=1,
            end_idx=3,
        )
    )
    assert len(windows) == 2, f"Expected 2 windows but got {len(windows)}"
    assert all(
        isinstance(window, pd.DataFrame) for window in windows
    ), "All windows should be of type pd.DataFrame"
    assert windows[0]["value"].tolist() == [
        2,
        3,
    ], f"Expected [2, 3] but got {windows[0]['value'].tolist()}"
