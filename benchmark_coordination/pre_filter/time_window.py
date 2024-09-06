from datetime import datetime
import pandas as pd
import networkx as nx
from typing import Generator

import benchmark_coordination.dataloader.dataframe as dataframe


def slide_df(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    start_time: datetime,
    end_time: datetime,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generate a time window that slides through the time range.
    :param data: pd.DataFrame, the data to be filtered.
    :param window_size: int, the size of the time window in minutes.
    :param step_size: int, the size of the step to slide the window in minutes.
    :param start_time: datetime, the start time of the time range.
    :param end_time: datetime, the end time of the time range.
    :return: generator, a generator that yields the data in each window.
    """
    current_window_start = start_time
    current_window_end = current_window_start + pd.Timedelta(minutes=window_size)
    while current_window_start <= end_time:
        yield dataframe.filter_data(data, current_window_start, current_window_end)
        current_window_start += pd.Timedelta(minutes=step_size)
        current_window_end = current_window_start + pd.Timedelta(minutes=window_size)


def slide_graph(
    graph: nx.Graph,
    window_size: int,
    step_size: int,
    start_time: datetime,
    end_time: datetime,
) -> Generator[nx.Graph, None, None]:
    """
    Generate a time window that slides through the time range.
    :param graph: nx.Graph, the graph to be filtered.
    :param window_size: int, the size of the time window in minutes.
    :param step_size: int, the size of the step to slide the window in minutes.
    :param start_time: datetime, the start time of the time range.
    :param end_time: datetime, the end time of the time range.
    :return: generator, a generator that yields the graph in each window.
    """
    current_window_start = start_time
    current_window_end = current_window_start + pd.Timedelta(minutes=window_size)
    while current_window_start <= end_time:
        yield graph.filter_graph(graph, current_window_start, current_window_end)
        current_window_start += pd.Timedelta(minutes=step_size)
        current_window_end = current_window_start + pd.Timedelta(minutes=window_size)
