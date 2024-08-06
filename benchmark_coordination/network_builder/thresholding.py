import pandas as pd
from typing import Literal


def filter_edgelist(
    df: pd.DataFrame,
    column_name: str,
    threshold: float,
    comparison: Literal["<", "<=", "==", ">=", ">"],
) -> pd.DataFrame:
    """
    Filter edges based on a threshold value and a comparison operator.
    :param df: pd.DataFrame, the dataframe containing the edges to be filtered.
        The dataframe should have columns 'source', 'target', and column_name.
    :param column_name: str, the name of the column to use for filtering.
    :param threshold: float, the threshold value to compare against.
    :param comparison: str, the comparison operator to use.
        The comparison operators available are: "<", "<=", "==", ">=", ">".
    :return: pd.DataFrame, the filtered dataframe.
    ----------------
    Example:
    >>> import pandas as pd
    >>> data = {
    ...     "source": [1, 2, 3, 4, 5],
    ...     "target": [6, 7, 8, 9, 10],
    ...     "weight": [0.1, 0.2, 0.3, 0.4, 0.5]
    ... }
    >>> df = pd.DataFrame(data)
    >>> filter_edgelist(df, "weight", 0.3, ">=")
       source  target  weight
    2       3       8     0.3
    3       4       9     0.4
    4       5      10     0.5
    """
    return df.query(f"{column_name} {comparison} {threshold}")


def filter_edgelist_by_percentile(
    df: pd.DataFrame,
    column_name: str,
    percentile: float,
    comparison: Literal["<", "<=", "==", ">=", ">"],
) -> pd.DataFrame:
    """
    Filter edges based on a threshold value and a comparison operator.
    :param df: pd.DataFrame, the dataframe containing the edges to be filtered.
        The dataframe should have columns 'source', 'target', and column_name.
    :param column_name: str, the name of the column to use for filtering.
    :param percentile: float, the percentile value to compare against.
    :param comparison: str, the comparison operator to use.
        The comparison operators available are: "<", "<=", "==", ">=", ">".
    :return: pd.DataFrame, the filtered dataframe.
    ----------------
    Example:
    >>> import pandas as pd
    >>> data = {
    ...     "source": [1, 2, 3, 4, 5],
    ...     "target": [6, 7, 8, 9, 10],
    ...     "weight": [0.1, 0.2, 0.3, 0.4, 0.5]
    ... }
    >>> df = pd.DataFrame(data)
    >>> filter_edgelist_by_percentile(df, "weight", 50, ">")
       source  target  weight
    0       5      10     0.5
    1       4       9     0.4
    2       3       8     0.3
    3       2       7     0.2
    4       1       6     0.1
    """
    threshold = df[column_name].quantile(percentile / 100)
    return filter_edgelist(df, column_name, threshold, comparison)
