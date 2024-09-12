import pandas as pd
from datetime import datetime
from typing import Any, Generator
import networkx as nx

import benchmark_coordination.utils.dataframe_utils as df_utils
import benchmark_coordination.utils.datetime_utils as dt_utils


def filter_dataframe(
    data: pd.DataFrame, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """
    Filter data based on the timestamp.
    :param data: pd.DataFrame, the data to be filtered, having a 'timestamp' column,
        which contains the timestamp of the data in datetime format (e.g. 2022-09-01 00:00:09).
    :param start_time: datetime, the start time of the filter (inclusive).
    :param end_time: datetime, the end time of the filter (inclusive).
    :return: pd.DataFrame, the filtered data.
    """
    data = df_utils.cast_columns_to_datetime(data.copy(), ["timestamp"])
    return data[data["timestamp"].between(start_time, end_time, inclusive="both")]


def slide_dataframe(
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
        yield filter_dataframe(data, current_window_start, current_window_end)
        current_window_start += pd.Timedelta(minutes=step_size)
        current_window_end = current_window_start + pd.Timedelta(minutes=window_size)


def filter_graph(
    bipartite: nx.Graph, start_time: datetime, end_time: datetime
) -> nx.Graph:
    """
    Parameters
    ----------
    bipartite : NetworkX graph
      The input graph should be bipartite, and
      all the edges should have attribute 'timestamps',
      with value a list of timestamps as strings.

    start_time : datetime, a timestamp that can be converted
      to be in the same string format as the items in 'timestamps'
      edge attribute. This is used in the filter as left boundary
      (select the edge if any timestamp in the edge is >= 'start_time')

    end_time : datetime, a timestamp that can be converted
      to be in the same string format as the items in 'timestamps'
      edge attribute. This is used in the filter as right boundary
      (select the edge if any timestamp in the edge is <= 'start_time')

    Returns
    -------
    Graph : NetworkX graph
       A graph obtained from filtering the input graph

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.Graph()
    >>> B.add_nodes_from(["1", "2", "3", "4", "5", "6"], bipartite=0)
    >>> B.add_nodes_from(["a", "b", "c"], bipartite=1)
    >>> B.add_edge("1", "a", timestamps=["2023-01-02 05:55:00", "2023-01-02 06:00:00"])
    >>> B.add_edge("2", "a", timestamps=["2023-01-01 06:05:00"])
    >>> B.add_edge("2", "b", timestamps=["2023-01-01 06:10:00"])
    >>> B.add_edge("3", "b", timestamps=["2023-01-01 05:50:00"])
    >>> B.add_edge("3", "c", timestamps=["2023-01-01 15:55:00"])
    >>> B.add_edge("4", "b", timestamps=["2023-01-01 06:00:00"])
    >>> B.add_edge("5", "c", timestamps=["2023-01-01 06:10:00"])
    >>> B.add_edge("6", "c", timestamps=["2023-01-01 06:10:00"])

    >>> BB = filter_graph(B, "2023-01-02 00:00:00", "2023-01-02 10:00:00")
    >>> for e in BB.edges(data=True):
    ...     print(e)
    ...
    ('1', 'a', {'timestamps': ['2023-01-02 05:55:00', '2023-01-02 06:00:00']})
    """

    def _filter_edge(n1: Any, n2: Any) -> bool:
        """
        Filter edges based on the 'timestamps' values
        :param n1: node 1
        :param n2: node 2
        :return: bool, True if the edge should be included, False otherwise
        """
        ts = bipartite[n1][n2].get("timestamps", [])
        after_time = dt_utils.datetime_to_timestamp_str(start_time)
        before_time = dt_utils.datetime_to_timestamp_str(end_time)
        return any(after_time <= t <= before_time for t in ts)

    return nx.subgraph_view(bipartite, filter_edge=_filter_edge)


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
        yield filter_graph(graph, current_window_start, current_window_end)
        current_window_start += pd.Timedelta(minutes=step_size)
        current_window_end = current_window_start + pd.Timedelta(minutes=window_size)
