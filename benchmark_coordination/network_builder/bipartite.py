import networkx as nx
import pandas as pd


def build_bipartite_graph(
    data: pd.DataFrame,
    source_column: str,
    target_column: str,
    with_timestamps: bool = False,
) -> nx.Graph:
    """
    Build a bipartite graph from a dataframe.
    :param data: pd.DataFrame, the dataframe containing the data to be used to build the bipartite graph.
        The dataframe should have columns as specified in source_column and target_column.
    :param source_column: str, the name of the column containing the source nodes (e.g. "user_id").
    :param target_column: str, the name of the column containing the target nodes (e.g. "hashtag").
    :param with_timestamps: bool, whether the dataframe contains timestamps. If True, the function will
        look for a 'timestamp' column in the dataframe, and use it to add the timestamps as edge attributes.
    :return: nx.Graph, the bipartite graph.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [1, 1, 1, 2, 2],
    ...     "hashtag": ["#A", "#B", "#A", "#A", "#C"]
    ... })
    >>> build_bipartite_graph(df).edges(data=True)
    EdgeDataView([(1, '#A', {'weight': 2, 'timestamps': []}), (1, '#B', {'weight': 1, 'timestamps': []}), ('#A', 2, {'weight': 1, 'timestamps': []}), (2, '#C', {'weight': 1, 'timestamps': []})])
    """
    # group by source and target columns, count the number of occurrences
    # if the dataframe has timestamps, also aggregate the timestamps
    aggregation = (
        {"count": "sum", "timestamp": list} if with_timestamps else {"count": "sum"}
    )
    data["count"] = 1
    data = data.groupby([source_column, target_column]).agg(aggregation).reset_index()  # type: ignore
    G = nx.Graph()
    for _, row in data.iterrows():
        source = row[source_column]
        target = row[target_column]
        G.add_node(source, bipartite=0)
        G.add_node(target, bipartite=1)
        G.add_edge(
            source, target, weight=row["count"], timestamps=row.get("timestamp", [])
        )
    return G
