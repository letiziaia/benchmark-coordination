from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def tf_idf(
    df: pd.DataFrame,
    column: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate the TF-IDF values for the content in a column.
    :param df: pd.DataFrame, the dataframe containing the content.
    :param column: str, the name of the column containing the content.
    :param kwargs: additional keyword arguments to be passed to TfidfVectorizer.
    :return: pd.DataFrame, the dataframe with the TF-IDF values added
        as a new column of type list ('tf_idf').
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "text": ["Hello world", "Hi there", "Hey you"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> tf_idf(df, "text")
          text                                             tf_idf
    0  Hello world  [0.7071067811865476, 0.0, 0.0, 0.0, 0.70710678...
    1     Hi there  [0.0, 0.0, 0.7071067811865476, 0.7071067811865...
    2      Hey you  [0.0, 0.7071067811865476, 0.0, 0.0, 0.0, 0.707...
    >>> tf_idf(df, "text", stop_words='english')
            text                                             tf_idf
    0  Hello world  [0.7071067811865476, 0.0, 0.0, 0.7071067811865...
    1     Hi there                               [0.0, 0.0, 1.0, 0.0]
    2      Hey you                               [0.0, 1.0, 0.0, 0.0]
    """
    vectorizer = TfidfVectorizer(**kwargs)
    X = vectorizer.fit_transform(df[column])
    df["tf_idf"] = [list(x) for x in list(X.toarray())]
    return df
