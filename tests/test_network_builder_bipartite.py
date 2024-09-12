import pytest
import pandas as pd
import networkx as nx
from benchmark_coordination.network_builder.bipartite import (
    build_bipartite_graph,
    project_on_nodes,
)


def test_build_bipartite_graph_notime():
    """
    Test the build_bipartite_graph function without timestamps.
    """
    df = pd.DataFrame(
        {"user_id": [1, 1, 1, 2, 2], "hashtag": ["#A", "#B", "#A", "#A", "#C"]}
    )
    result = build_bipartite_graph(
        data=df, source_column="user_id", target_column="hashtag"
    )
    expected_edges = [
        (1, "#A", {"weight": 2, "timestamps": []}),
        (1, "#B", {"weight": 1, "timestamps": []}),
        ("#A", 2, {"weight": 1, "timestamps": []}),
        (2, "#C", {"weight": 1, "timestamps": []}),
    ]
    assert isinstance(
        result, nx.Graph
    ), f"Expected a NetworkX graph, got {type(result)}"
    assert (
        list(result.edges(data=True)) == expected_edges
    ), f"Expected {expected_edges}, got {result.edges(data=True)}"


def test_build_bipartite_graph_withtime():
    """
    Test the build_bipartite_graph function with timestamps.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "hashtag": ["#A", "#B", "#A", "#A", "#C"],
            "timestamp": [
                "2024-01-01 08:00:00",
                "2024-01-01 08:00:00",
                "2024-01-01 08:01:00",
                "2024-01-01 08:02:00",
                "2024-01-01 08:03:00",
            ],
        }
    )
    result = build_bipartite_graph(
        data=df, source_column="user_id", target_column="hashtag", with_timestamps=True
    )

    expected_edges = [
        (
            1,
            "#A",
            {"weight": 2, "timestamps": ["2024-01-01 08:00:00", "2024-01-01 08:01:00"]},
        ),
        (1, "#B", {"weight": 1, "timestamps": ["2024-01-01 08:00:00"]}),
        ("#A", 2, {"weight": 1, "timestamps": ["2024-01-01 08:02:00"]}),
        (2, "#C", {"weight": 1, "timestamps": ["2024-01-01 08:03:00"]}),
    ]

    assert isinstance(
        result, nx.Graph
    ), f"Expected a NetworkX graph, got {type(result)}"

    assert (
        list(result.edges(data=True)) == expected_edges
    ), f"Expected {expected_edges}, got {result.edges(data=True)}"


@pytest.fixture
def bipartite_graph():
    B = nx.Graph()
    B.add_nodes_from(["1", "2", "3", "4", "5", "6"], bipartite=0)
    B.add_nodes_from(["A", "B", "C", "D"], bipartite=1)
    B.add_edges_from(
        [
            ("1", "A", {"weight": 1}),
            ("1", "B", {"weight": 2}),
            ("2", "B", {"weight": 1}),
            ("2", "C", {"weight": 1}),
            ("3", "C", {"weight": 3}),
            ("3", "D", {"weight": 1}),
        ]
    )
    return B


def test_project_on_nodes_unweighted(bipartite_graph):
    """
    Test the project_on_nodes function without weights.
    """
    nodes = ["1", "2", "3"]
    result = project_on_nodes(bipartite=bipartite_graph, nodes=nodes)
    expected_edges = [("1", "2", {"weight": 1}), ("2", "3", {"weight": 1})]

    assert isinstance(
        result, nx.Graph
    ), f"Expected a NetworkX graph, got {type(result)}"
    assert (
        list(result.edges(data=True)) == expected_edges
    ), f"Expected {expected_edges}, got {result.edges(data=True)}"


def test_project_on_nodes_weighted(bipartite_graph):
    """
    Test the project_on_nodes function with weights.
    """
    nodes = ["1", "2", "3"]
    result = project_on_nodes(bipartite=bipartite_graph, nodes=nodes, weight="weight")
    expected_edges = [("1", "2", {"weight": 3}), ("2", "3", {"weight": 4})]

    assert isinstance(
        result, nx.Graph
    ), f"Expected a NetworkX graph, got {type(result)}"
    assert (
        list(result.edges(data=True)) == expected_edges
    ), f"Expected {expected_edges}, got {result.edges(data=True)}"
