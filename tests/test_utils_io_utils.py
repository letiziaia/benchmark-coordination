import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from benchmark_coordination.utils.io_utils import read_from_parquet, save_to_parquet


@patch("benchmark_coordination.utils.io_utils.pd.read_parquet")
def test_read_from_parquet(mock_read_parquet):
    """
    Test the read_from_parquet function.
    """
    mock_df = pd.DataFrame(
        {
            "author_id": [1, 2, 3],
            "author": ["Alice", "Bob", "Charlie"],
            "tweet_text": ["Hello", "Hi", "Hey"],
            "timestamp": [
                datetime(2022, 1, 1, 12, 0, 0),
                datetime(2022, 1, 2, 12, 0, 0),
                datetime(2022, 1, 3, 12, 0, 0),
            ],
            "links": [None, None, None],
            "is_retweet": [False, False, False],
            "original_author": [None, None, None],
            "mentioned_usernames": [None, None, None],
            "mentioned_hashtags": [None, None, None],
        }
    )
    mock_read_parquet.return_value = mock_df

    # call the function
    file_path = "dummy_path"
    result = read_from_parquet(file_path)

    # check that the function called pd.read_parquet with the expected arguments
    mock_read_parquet.assert_called_with(
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
    ), "pd.read_parquet was not called with the expected arguments"
    assert isinstance(
        result, pd.DataFrame
    ), "Expected a DataFrame but got something else"


def test_save_to_parquet_raise():
    """
    Test that save_to_parquet raises an AssertionError when data is empty.
    """
    data = pd.DataFrame()
    file_path = "tests/data/test_data.parquet"
    with pytest.raises(AssertionError):
        save_to_parquet(data, file_path)


@patch("benchmark_coordination.utils.io_utils.pd")
def test_save_to_parquet(mock_pd):
    """
    Test the save_to_parquet function.
    """
    data = pd.DataFrame(
        {
            "author_id": [1, 2, 3],
            "author": ["Alice", "Bob", "Charlie"],
            "tweet_text": ["Hello", "Hi", "Hey"],
            "timestamp": [
                datetime(2022, 1, 1, 12, 0, 0),
                datetime(2022, 1, 2, 12, 0, 0),
                datetime(2022, 1, 3, 12, 0, 0),
            ],
            "links": [None, None, None],
            "is_retweet": [False, False, False],
            "original_author": [None, None, None],
            "mentioned_usernames": [None, None, None],
            "mentioned_hashtags": [None, None, None],
        }
    )
    # patch the to_parquet method of the DataFrame class
    mock_pd.DataFrame.to_parquet = MagicMock()
    # make sure that calling to_parquet from data will call the mock
    data.to_parquet = mock_pd.DataFrame.to_parquet

    # call the function
    file_path = "tests/data/test_data.parquet"
    save_to_parquet(data, file_path)

    data.to_parquet.assert_called(), "to_parquet was not called"
    data.to_parquet.called_with(
        file_path, index=False
    ), "to_parquet was not called with the expected arguments"
