import networkx as nx


def filter_graph(bipartite: nx.Graph, after_time: str, before_time: str) -> nx.Graph:
    """
    Parameters
    ----------
    bipartite : NetworkX graph
      The input graph should be bipartite, and
      all the edges should have attribute 'timestamp',
      with value a list of timestamps as strings.

    after_time : str, a timestamp in the same format as
      in 'timestamp'. This is used in the filter as
      left boundary (select the edge if any timestamp
      in the edge is >= 'after_time')

    before_time : str, a timestamp in the same format as
      in 'timestamp'. This is used in the filter as
      right boundary (select the edge if any timestamp
      in the edge is <= 'after_time')

    Returns
    -------
    Graph : NetworkX graph
       A graph obtained from filtering the input graph B

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.Graph()
    >>> B.add_nodes_from(["1", "2", "3", "4", "5", "6"], bipartite=0)
    >>> B.add_nodes_from(["a", "b", "c"], bipartite=1)
    >>> B.add_edge("1", "a", timestamp=["2023-01-02 05:55:00", "2023-01-02 06:00:00"])
    >>> B.add_edge("2", "a", timestamp=["2023-01-01 06:05:00"])
    >>> B.add_edge("2", "b", timestamp=["2023-01-01 06:10:00"])
    >>> B.add_edge("3", "b", timestamp=["2023-01-01 05:50:00"])
    >>> B.add_edge("3", "c", timestamp=["2023-01-01 15:55:00"])
    >>> B.add_edge("4", "b", timestamp=["2023-01-01 06:00:00"])
    >>> B.add_edge("5", "c", timestamp=["2023-01-01 06:10:00"])
    >>> B.add_edge("6", "c", timestamp=["2023-01-01 06:10:00"])

    >>> BB = filter_graph(B, "2023-01-02 00:00:00", "2023-01-02 10:00:00")
    >>> for e in BB.edges(data=True):
    ...     print(e)
    ...
    ('1', 'a', {'timestamp': ['2023-01-02 05:55:00', '2023-01-02 06:00:00']})
    """

    def _filter_edge(n1, n2):
        ts = bipartite[n1][n2].get("timestamp", [])
        return any(after_time <= t <= before_time for t in ts)

    return nx.subgraph_view(bipartite, filter_edge=_filter_edge)
