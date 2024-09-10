import pytest
import pandas as pd
import networkx as nx
from datetime import datetime
from benchmark_coordination.time_window import (
    filter_dataframe,
    slide_dataframe,
    filter_graph,
    slide_graph,
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


@pytest.fixture
def sample_graph():
    B = nx.Graph()
    B.add_nodes_from(["1", "2", "3", "4", "5", "6"], bipartite=0)
    B.add_nodes_from(["a", "b", "c"], bipartite=1)
    B.add_edge("1", "a", timestamps=["2023-01-02 05:55:00", "2023-01-02 06:00:00"])
    B.add_edge("2", "a", timestamps=["2023-01-01 06:05:00"])
    B.add_edge("2", "b", timestamps=["2023-01-01 06:10:00"])
    B.add_edge("3", "b", timestamps=["2023-01-01 05:50:00"])
    B.add_edge("3", "c", timestamps=["2023-01-01 15:55:00"])
    B.add_edge("4", "b", timestamps=["2023-01-01 06:00:00"])
    B.add_edge("5", "c", timestamps=["2023-01-01 06:10:00"])
    B.add_edge("6", "c", timestamps=["2023-01-01 06:10:00"])
    return B


def test_filter_dataframe(sample_dataframe):
    """
    Test the filter_dataframe function.
    """
    start_time = datetime(2022, 9, 1, 0, 1, 0)
    end_time = datetime(2022, 9, 1, 0, 3, 0)
    filtered_df = filter_dataframe(sample_dataframe, start_time, end_time)
    assert len(filtered_df) == 3, f"Expected 3 rows but got {len(filtered_df)}"
    assert filtered_df["value"].tolist() == [
        2,
        3,
        4,
    ], f"Expected [2, 3, 4] but got {filtered_df['value'].tolist()}"


def test_slide_dataframe(sample_dataframe):
    """
    Test the slide_dataframe function.
    """
    start_time = datetime(2022, 9, 1, 0, 0, 0)
    end_time = datetime(2022, 9, 1, 0, 4, 0)
    window_minutes = 2
    step_minutes = 1
    windows = list(
        slide_dataframe(
            sample_dataframe, window_minutes, step_minutes, start_time, end_time
        )
    )
    assert len(windows) == 5, f"Expected 5 windows but got {len(windows)}"
    assert all(
        isinstance(window, pd.DataFrame) for window in windows
    ), "All windows should be of type pd.DataFrame"


def test_filter_graph(sample_graph):
    """
    Test the filter_graph function.
    """
    start_time = datetime(2023, 1, 2, 0, 0, 0)
    end_time = datetime(2023, 1, 2, 10, 0, 0)
    filtered_graph = filter_graph(sample_graph, start_time, end_time)
    assert (
        len(filtered_graph.edges) == 1
    ), f"Expected 1 edge but got {len(filtered_graph.edges)}"
    assert (
        "1",
        "a",
    ) in filtered_graph.edges, "Expected edge ('1', 'a') to be in the graph"


def test_slide_graph(sample_graph):
    """
    Test the slide_graph function.
    """
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    end_time = datetime(2023, 1, 2, 0, 0, 0)
    window_size = 1440  # 1 day in minutes
    step_size = 720  # 12 hours in minutes
    windows = list(
        slide_graph(sample_graph, window_size, step_size, start_time, end_time)
    )
    assert len(windows) == 3, f"Expected 3 windows but got {len(windows)}"
    assert all(
        isinstance(window, nx.Graph) for window in windows
    ), "All windows should be of type nx.Graph"
