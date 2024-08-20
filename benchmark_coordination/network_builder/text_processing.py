import pandas as pd
import nltk  # type: ignore
from nltk.util import ngrams  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore


nltk.download("punkt_tab")
nltk.download("wordnet")


def lower_case_column_content(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert the content of a column to lower case.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be converted to lower case.
    :return: pd.DataFrame, the dataframe with lower case content in the specified column.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "author": ["Alice", "Bob", "Charlie"],
    ...     "tweet": ["Hello", "Hi", "Hey"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> lower_case_column_content(df, "tweet")
       author tweet
    0  Alice  hello
    1    Bob    hi
    2 Charlie   hey
    """
    df[column] = df[column].str.lower()
    return df


def remove_leading_symbol(df: pd.DataFrame, column: str, symbol: str) -> pd.DataFrame:
    """
    Remove leading symbol from the content of a column.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be modified.
    :param symbol: str, the leading symbol to be removed (e.g. "#" or "@").
    :return: pd.DataFrame, the dataframe with leading symbol removed from the specified column.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "text": ["#Hello", "#Hi", "#Hey"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> remove_leading_symbol(df, "text", "#")
       text
    0  Hello
    1    Hi
    2   Hey
    """
    assert len(symbol) == 1, "Symbol must be a single character."
    df[column] = df[column].str.replace(rf"^{symbol}", "", regex=True)
    return df


def clean_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Clean the text content of a column by removing special characters
    and leading/trailing spaces.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be cleaned.
    :return: pd.DataFrame, the dataframe with cleaned text content in the specified column.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "text": ["Hello!", " Hi ", "Hey! "]
    ... }
    >>> df = pd.DataFrame(data)
    >>> clean_text_column(df, "text")
       text
    0  Hello
    1    Hi
    2   Hey
    """
    df[column] = df[column].str.replace(r"[^a-zA-Z0-9\s]", "")
    df[column] = df[column].str.strip()
    return df


def split_text_column_into_ngrams(
    df: pd.DataFrame, column: str, n: int
) -> pd.DataFrame:
    """
    Split the text content of a column into n-grams.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be split.
    :param n: int, the size of the n-grams.
    :return: pd.DataFrame, the dataframe with the text content
        in the specified column split into n-grams.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...    "text": ["Hello world", "Hi there", "Hey you"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> split_text_column_into_ngrams(df, "text", 2)
       text
    0  [(Hello, world), (world, </s>)]
    1    [(Hi, there), (there, </s>)]
    2      [(Hey, you), (you, </s>)]
    """
    df[column] = (
        df[column]
        .str.split()
        .apply(
            lambda x: list(
                ngrams(sequence=x, n=n, pad_right=True, right_pad_symbol="</s>")
            )
        )
    )
    return df


def remove_stopwords(df: pd.DataFrame, column: str, stopwords: list) -> pd.DataFrame:
    """
    Remove stopwords from the text content of a column.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be modified.
    :param stopwords: list, the list of stopwords to be removed.
    :return: pd.DataFrame, the dataframe with stopwords removed from the specified column.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "text": ["Hello world", "Hi there", "Hey you"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> remove_stopwords(df, "text", ["world", "there"])
       text
    0  Hello
    1     Hi
    2  Hey you
    """
    df[column] = df[column].apply(
        lambda x: " ".join([word for word in x.split() if word not in stopwords])
    )
    return df


def text_lemmatize_and_tokenize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Lemmatize and tokenize the text content of a column.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be modified.
    :return: pd.DataFrame, the dataframe with lemmatized and tokenized text content
        in the specified column.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "text": ["I am running", "He is walking", "They are jumping"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> text_lemmatize_and_tokenize(df, "text")
                   text
    0      [I, am, running]
    1     [He, is, walking]
    2  [They, are, jumping]
    """
    df[column] = [word_tokenize(x) for x in df[column].values]
    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return df


def text_stemming(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Stem the text content of a column.
    :param df: pd.DataFrame, the dataframe to be modified.
    :param column: str, the name of the column to be stemmed.
    :return: pd.DataFrame, the dataframe with stemmed text content in the specified column.
    ----------------
    Example:
    ----------------
    >>> import pandas as pd
    >>> data = {
    ...     "text": ["I am running", "He is walking", "They are jumping"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> text_stemming(df, "text")
                text
    0       i am run
    1     he is walk
    2  they are jump
    """
    stemmer = nltk.PorterStemmer()
    df[column] = df[column].apply(lambda x: stemmer.stem(x))
    return df
