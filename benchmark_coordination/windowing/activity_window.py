import pandas as pd
from typing import Generator


def filter_dataframe(data: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """
    Filter data based on the given indices.
    :param data: pd.DataFrame, the data to be filtered
    :param start_idx: int, the start index of the filter (inclusive).
    :param end_idx: int, the end index of the filter (exclusive).
    :return: pd.DataFrame, the filtered data.
    """
    assert start_idx < end_idx, "Start index should be less than end index"
    assert (
        data.index.is_monotonic_increasing
    ), "Data index should be monotonically increasing"
    return data.iloc[start_idx:end_idx]


def slide_dataframe(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    start_idx: int = 0,
    end_idx: int = -1,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generate a time window that slides through the activities.
    :param data: pd.DataFrame, the data to be filtered.
    :param window_size: int, the size of the time window as number of activities.
    :param step_size: int, the size of the step to slide the window.
    :param start_idx: int, the start index of the data to consider.
        Default is 0.
    :param end_idx: int, the end index of the data to consider.
        Default is -1, which means the last index.
    :return: generator, a generator that yields the data in each window.
    """
    final_index = len(data) if end_idx == -1 else end_idx
    current_window_start = max(start_idx, 0)
    current_window_end = current_window_start + window_size
    while current_window_start < final_index:
        yield filter_dataframe(data, current_window_start, current_window_end)
        current_window_start += step_size
        current_window_end = current_window_start + window_size
