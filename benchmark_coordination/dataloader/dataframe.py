import pandas as pd
from datetime import datetime


def read_data(file_path: str) -> pd.DataFrame:
    """
    Read data from a parquet file.
    :param file_path: str, the path to the parquet file.
        e.g. 'scratch/cs/ecanet/coordination_sim/all_real.parquet.gzip'
    :return: pd.DataFrame, the data read from the parquet file.
    """
    return pd.read_parquet(
        file_path,
        columns=[
            "author_id",
            "author",
            "tweet_text",
            "timestamp",
            "links",
            "is_retweet",
            "original_author",
            "mentioned_usernames",
            "mentioned_hashtags",
        ],
    )


def filter_data(
    data: pd.DataFrame, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """
    Filter data based on the timestamp.
    :param data: pd.DataFrame, the data to be filtered.
    :param start_time: datetime, the start time of the filter (inclusive).
    :param end_time: datetime, the end time of the filter (inclusive).
    :return: pd.DataFrame, the filtered data.
    """
    return data[data["timestamp"].between(start_time, end_time, inclusive="both")]
