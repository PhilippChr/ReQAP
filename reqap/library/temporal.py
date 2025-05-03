from datetime import datetime, time, timedelta, date
from loguru import logger


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATETIME_FORMAT2 = "%Y-%m-%d %H:%M:%S"
DATETIME_FORMAT3 = "%Y-%m-%d %H:%M:%S+00:00"
WEEKTIME_FORMAT = "%w, %H:%M:%SZ"
WEEKYEAR_FORMAT = "Week %W in %Y"
TIME_FORMAT = "%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


"""
RELATED TO TEMPORAL OPERATIONS
"""
def timestamp_to_datetime(date_str: str, all_formats: bool=False) -> datetime:
    def _timestamp_to_datetime(date_str: str, date_format: str, show_warning: bool=True) -> datetime | str:
        try:
            dt = datetime.strptime(date_str, date_format)
        except ValueError as e:
            if show_warning:
                logger.warning(f"Date {date_str} does not match format {date_format}. Error: {e}")
                raise e
            return None
        return dt
    if all_formats:
        res = _timestamp_to_datetime(date_str, date_format=DATETIME_FORMAT, show_warning=False)
        if not res is None:
            return res
        res = _timestamp_to_datetime(date_str, date_format=DATETIME_FORMAT2, show_warning=False)
        if not res is None:
            return res
        return _timestamp_to_datetime(date_str, date_format=DATETIME_FORMAT3, show_warning=True)
    else:
        return _timestamp_to_datetime(date_str, date_format=DATETIME_FORMAT, show_warning=True)


def datetime_to_timestamp(dt: datetime) -> str:
    date_str = dt.strftime(DATETIME_FORMAT)
    return date_str


def weektime_to_datetime(weektime_str: str) -> datetime:
    dt = datetime.strptime(weektime_str, WEEKTIME_FORMAT)
    return dt


def datetime_to_weektime(dt: datetime) -> str:
    weektime_str = dt.strftime(WEEKTIME_FORMAT)
    return weektime_str


def datetime_to_time(dt: datetime) -> str:
    time_str = dt.strftime(TIME_FORMAT)
    return time_str


def datetime_to_date(dt: datetime) -> str:
    time_str = dt.strftime(DATE_FORMAT)
    return time_str


def date_to_datetime(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, DATE_FORMAT)
    return dt


def date_to_timestamp(d: date) -> str:
    dt = datetime.combine(d, datetime.min.time())
    return datetime_to_timestamp(dt)


def str_to_time(time_str: str) -> time:
    t = time.fromisoformat(time_str)
    return t


def time_to_str(t: time) -> str:
    time_str = t.isoformat()
    return time_str


def combine_dt_t(dt: datetime, t: time) -> datetime:
    dt_combined = datetime.combine(dt, t)
    return dt_combined


def drop_time_from_date(date_str: str) -> str:
    return date_str.split("T")[0]


def timestamp_to_weekyear(date_str: str) -> str:
    dt = timestamp_to_datetime(date_str)
    return datetime_to_weekyear(dt)


def datetime_to_weekyear(dt: datetime) -> str:
    weekyear_str = dt.strftime(WEEKYEAR_FORMAT)
    return weekyear_str


def set_weektime(weektime_str: str, week_start_dt: datetime, week_end_dt: datetime) -> datetime:
    """
    Computes a specific date when given the weektime (<weekday>, <weektime>),
    and the start date and end date of the week.
    """
    weektime_dt = weektime_to_datetime(weektime_str)
    weektime_weekday = int(weektime_str.split(", ")[0])
    days_offset = (weektime_weekday - week_start_dt.isocalendar().weekday) % 7
    specific_date_dt = week_start_dt + timedelta(days=days_offset)
    specific_date_dt = specific_date_dt.replace(
        hour=weektime_dt.hour,
        minute=weektime_dt.minute,
        second=weektime_dt.second
    )
    assertion_error_text = (
        f"Critical issue with setting the weektime: computed weektime not in desired week."
        "weektime_str: {weektime_str}, week_start_dt: {week_start_dt}, specific_date_dt: {specific_date_dt}"
    )
    assert specific_date_dt > week_start_dt, assertion_error_text
    assert specific_date_dt < week_end_dt, assertion_error_text
    return specific_date_dt


def num_timestamp_to_datetime(timestamp: str|int|float) -> datetime:
    timestamp = int(timestamp)
    return datetime.fromtimestamp(timestamp)


def current_year() -> int:
    """
    Returns the current year as int.
    """
    return datetime.now().year


def current_datetime() -> datetime:
    """
    Returns the current date as datetime.
    """
    return datetime.now()


def is_temporal_expression(string: str) -> bool:
    return is_date(string) or is_timestamp(string)


def is_timestamp(date_str: str) -> bool:
    def _is_timestamp(date_str: str, date_format: str):
        try:
            datetime.strptime(date_str, date_format)
            return True
        except:
            return False
    return _is_timestamp(date_str, DATETIME_FORMAT) or _is_timestamp(date_str, DATETIME_FORMAT2) or _is_timestamp(date_str, DATETIME_FORMAT3)
        

def is_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, DATE_FORMAT)
        return True
    except:
        return False
    

def temporal_equivalent(str1: str, str2: str) -> bool:
    if is_date(str1) and is_date(str2):
        return date_to_datetime(str1) == date_to_datetime(str2)
    elif is_date(str1) and is_timestamp(str2):
        return date_to_datetime(str1) == timestamp_to_datetime(str2, all_formats=True)
    elif is_date(str2) and is_timestamp(str1):
        return date_to_datetime(str2) == timestamp_to_datetime(str1, all_formats=True)
    elif is_timestamp(str1) and is_timestamp(str2):
        return timestamp_to_datetime(str1, all_formats=True) == timestamp_to_datetime(str2, all_formats=True)
    else:
        return False
    

def numbers_equivalent(n1: int|float, n2: int|float, relax_factor: float=0.0):
    def _compare(n1: float, n2: float, relax_factor: float=0.0):
        n1 = round(n1, 2)
        n2 = round(n2, 2)
        n2_range = [n2 * (1.0 - relax_factor), n2 * (1.0 + relax_factor)]
        if n1 >= n2_range[0] and n1 <= n2_range[1]:
            return True
        return False
    if not type(n1) in [float, int] or not type(n2) in [float, int]:
        return False
    
    FACTORS = [1, 3600, 60, 1000]  # factors to convert units
    return any(_compare(n1*f, n2, relax_factor) or _compare(n1, n2*f, relax_factor) for f in FACTORS)


def datetime_overlap(e1_start_dt: datetime, e1_end_dt: datetime, e2_start_dt: datetime, e2_end_dt: datetime) -> bool:
    """
    Return if there is an overlap of two datetime pairs.
    """
    if e1_start_dt <= e2_end_dt and e2_start_dt <= e1_end_dt:
        return True
    return False
