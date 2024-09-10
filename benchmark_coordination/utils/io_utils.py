import pandas as pd
from benchmark_coordination.utils.logging import logger


def read_from_parquet(file_path: str) -> pd.DataFrame:
    """
    Read data from a parquet file.
    :param file_path: str, the path to the parquet file.
        e.g. 'scratch/cs/ecanet/coordination_sim/all_real.parquet.gzip'
    :return: pd.DataFrame, the data read from the parquet file.
    """
    logger.debug(f"Reading data from {file_path}")
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


def save_to_parquet(data: pd.DataFrame, file_path: str) -> None:
    """
    Save data to a parquet file.
    :param data: pd.DataFrame, the data to be saved.
    :param file_path: str, the path to the parquet file.
        e.g. 'scratch/cs/ecanet/coordination_sim/all_real.parquet.gzip'
    :return: None
    """
    assert not data.empty, "Data is empty"
    data.to_parquet(file_path, index=False)
    logger.debug(f"Data saved to {file_path}")
