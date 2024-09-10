import pandas as pd


def contains_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """
    Check if the DataFrame contains the specified columns.
    :param df: pd.DataFrame, the DataFrame to check.
    :param columns: list[str], the list of column names to check.
    :return: bool, True if all columns are present, False otherwise.
    """
    return all(column in df.columns for column in columns)


def cast_columns_to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Cast the specified columns in the DataFrame to datetime.
    :param df: pd.DataFrame, the DataFrame to cast.
    :param columns: list[str], the list of column names to cast.
    :return: pd.DataFrame, the DataFrame with the specified columns cast to datetime.
    """
    if not contains_columns(df, columns):
        raise ValueError("Columns not found in DataFrame")
    for column in columns:
        df[column] = pd.to_datetime(df[column])
    return df


def cast_columns_to_str(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Cast the specified columns in the DataFrame to string.
    :param df: pd.DataFrame, the DataFrame to cast.
    :param columns: list[str], the list of column names to cast.
    :return: pd.DataFrame, the DataFrame with the specified columns cast to string.
    """
    if not contains_columns(df, columns):
        raise ValueError("Columns not found in DataFrame")
    for column in columns:
        df[column] = df[column].astype(str)
    return df
