from datetime import datetime


def timestamp_str_to_datetime(timestamp: str) -> datetime:
    """
    Convert a timestamp string to a datetime object.
    :param timestamp: str, the timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.
    :return: datetime, the datetime object.
    """
    return datetime.strptime(timestamp, r"%Y-%m-%d %H:%M:%S")


def datetime_to_timestamp_str(dt: datetime) -> str:
    """
    Convert a datetime object to a timestamp string.
    :param dt: datetime, the datetime object.
    :return: str, the timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return dt.strftime(r"%Y-%m-%d %H:%M:%S")
