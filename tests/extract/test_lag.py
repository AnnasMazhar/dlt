import pytest
from datetime import datetime, date, timedelta, timezone
from typing import Any, Callable, Union

from dlt.common.incremental.typing import LastValueFunc
from dlt.common.pendulum import pendulum
from dlt.common.time import ensure_pendulum_date, ensure_pendulum_datetime_non_utc
from dlt.extract.incremental.lag import _apply_lag_to_value


@pytest.mark.parametrize(
    "lag,value,last_value_func,expected",
    [
        # Python datetime - naive
        (3600, datetime(2023, 1, 1, 12, 0, 0), max, datetime(2023, 1, 1, 11, 0, 0)),
        (3600, datetime(2023, 1, 1, 12, 0, 0), min, datetime(2023, 1, 1, 13, 0, 0)),
        (7200, datetime(2023, 1, 1, 12, 0, 0), max, datetime(2023, 1, 1, 10, 0, 0)),
        (7200, datetime(2023, 1, 1, 12, 0, 0), min, datetime(2023, 1, 1, 14, 0, 0)),
        # Fractional seconds
        (1.5, datetime(2023, 1, 1, 12, 0, 0), max, datetime(2023, 1, 1, 11, 59, 58, 500000)),
        (1.5, datetime(2023, 1, 1, 12, 0, 0), min, datetime(2023, 1, 1, 12, 0, 1, 500000)),
        # Zero lag
        (0, datetime(2023, 1, 1, 12, 0, 0), max, datetime(2023, 1, 1, 12, 0, 0)),
        (0, datetime(2023, 1, 1, 12, 0, 0), min, datetime(2023, 1, 1, 12, 0, 0)),
    ],
    ids=[
        "python_datetime_naive_max_1hour",
        "python_datetime_naive_min_1hour",
        "python_datetime_naive_max_2hour",
        "python_datetime_naive_min_2hour",
        "python_datetime_naive_max_1.5sec",
        "python_datetime_naive_min_1.5sec",
        "python_datetime_naive_max_zero_lag",
        "python_datetime_naive_min_zero_lag",
    ],
)
def test_apply_lag_to_value_python_datetime_naive(
    lag: Union[int, float], value: datetime, last_value_func: LastValueFunc[Any], expected: datetime
) -> None:
    result = _apply_lag_to_value(lag, value, last_value_func)
    assert result == expected
    # assert type(result) == type(expected)
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "lag,value,last_value_func,expected_offset_hours",
    [
        # Python datetime - timezone aware
        (3600, 0, max, -1),  # 1 hour back
        (3600, 5, min, 1),  # 1 hour forward (UTC+5)
        (7200, -3, max, -2),  # 2 hours back (UTC-3)
        (1800, 8, min, 0.5),  # 30 minutes forward (UTC+8)
    ],
    ids=[
        "python_datetime_utc_max_1hour",
        "python_datetime_plus5_min_1hour",
        "python_datetime_minus3_max_2hour",
        "python_datetime_plus8_min_30min",
    ],
)
def test_apply_lag_to_value_python_datetime_aware(
    lag: Union[int, float],
    value: int,
    last_value_func: LastValueFunc[Any],
    expected_offset_hours: Union[int, float],
) -> None:
    base_dt = datetime(2023, 1, 1, 12, 0, 0)

    if value == 0:
        tz_aware_dt = base_dt.replace(tzinfo=timezone.utc)
    else:
        tz = timezone(timedelta(hours=value))
        tz_aware_dt = base_dt.replace(tzinfo=tz)

    result = _apply_lag_to_value(lag, tz_aware_dt, last_value_func)

    expected_dt = tz_aware_dt + timedelta(hours=expected_offset_hours)
    if last_value_func == max:
        expected_dt = tz_aware_dt - timedelta(seconds=lag)
    else:
        expected_dt = tz_aware_dt + timedelta(seconds=lag)

    assert result == expected_dt
    assert result.tzinfo.utcoffset(result) == tz_aware_dt.tzinfo.utcoffset(result)


@pytest.mark.parametrize(
    "lag,value,last_value_func,expected",
    [
        # Python date
        (1, date(2023, 1, 15), max, date(2023, 1, 14)),
        (1, date(2023, 1, 15), min, date(2023, 1, 16)),
        (7, date(2023, 1, 15), max, date(2023, 1, 8)),
        (7, date(2023, 1, 15), min, date(2023, 1, 22)),
        (0, date(2023, 1, 15), max, date(2023, 1, 15)),
        (0, date(2023, 1, 15), min, date(2023, 1, 15)),
        # Edge cases with month boundaries
        (1, date(2023, 2, 1), max, date(2023, 1, 31)),
        (1, date(2023, 1, 31), min, date(2023, 2, 1)),
    ],
    ids=[
        "python_date_max_1day",
        "python_date_min_1day",
        "python_date_max_7days",
        "python_date_min_7days",
        "python_date_max_zero_lag",
        "python_date_min_zero_lag",
        "python_date_max_month_boundary",
        "python_date_min_month_boundary",
    ],
)
def test_apply_lag_to_value_python_date(
    lag: Union[int, float],
    value: date,
    last_value_func: LastValueFunc[Any],
    expected: date,
) -> None:
    result = _apply_lag_to_value(lag, value, last_value_func)
    assert result == expected
    assert type(result) is pendulum.Date


@pytest.mark.parametrize(
    "lag,value_str,last_value_func,expected_str",
    [
        # Pendulum datetime - naive (treated as local time)
        (3600, "2023-01-01T12:00:00", max, "2023-01-01T11:00:00"),
        (3600, "2023-01-01T12:00:00", min, "2023-01-01T13:00:00"),
        (7200, "2023-01-01T12:00:00", max, "2023-01-01T10:00:00"),
        (7200, "2023-01-01T12:00:00", min, "2023-01-01T14:00:00"),
        # Pendulum datetime - UTC
        (3600, "2023-01-01T12:00:00Z", max, "2023-01-01T11:00:00Z"),
        (3600, "2023-01-01T12:00:00Z", min, "2023-01-01T13:00:00Z"),
        # Pendulum datetime - with timezone
        (3600, "2023-01-01T12:00:00+05:00", max, "2023-01-01T11:00:00+05:00"),
        (3600, "2023-01-01T12:00:00-03:00", min, "2023-01-01T13:00:00-03:00"),
    ],
    ids=[
        "pendulum_datetime_naive_max_1hour",
        "pendulum_datetime_naive_min_1hour",
        "pendulum_datetime_naive_max_2hour",
        "pendulum_datetime_naive_min_2hour",
        "pendulum_datetime_utc_max_1hour",
        "pendulum_datetime_utc_min_1hour",
        "pendulum_datetime_plus5_max_1hour",
        "pendulum_datetime_minus3_min_1hour",
    ],
)
def test_apply_lag_to_value_pendulum_datetime(
    lag: Union[int, float], value_str: str, last_value_func: LastValueFunc[Any], expected_str: str
) -> None:
    value = ensure_pendulum_datetime_non_utc(value_str)
    expected = ensure_pendulum_datetime_non_utc(expected_str)

    result = _apply_lag_to_value(lag, value, last_value_func)

    assert result == expected
    assert isinstance(result, pendulum.DateTime)
    assert result.timezone == value.timezone


@pytest.mark.parametrize(
    "lag,value_str,last_value_func,expected_str",
    [
        # Pendulum date
        (1, "2023-01-15", max, "2023-01-14"),
        (1, "2023-01-15", min, "2023-01-16"),
        (7, "2023-01-15", max, "2023-01-08"),
        (7, "2023-01-15", min, "2023-01-22"),
        (0, "2023-01-15", max, "2023-01-15"),
        (0, "2023-01-15", min, "2023-01-15"),
    ],
    ids=[
        "pendulum_date_max_1day",
        "pendulum_date_min_1day",
        "pendulum_date_max_7days",
        "pendulum_date_min_7days",
        "pendulum_date_max_zero_lag",
        "pendulum_date_min_zero_lag",
    ],
)
def test_apply_lag_to_pendulum_date(
    lag: Union[int, float], value_str: str, last_value_func: LastValueFunc[Any], expected_str: str
) -> None:
    value = ensure_pendulum_date(value_str)
    expected = ensure_pendulum_date(expected_str)

    result = _apply_lag_to_value(lag, value, last_value_func)

    assert result == expected
    assert isinstance(result, pendulum.Date)


@pytest.mark.parametrize(
    "lag,value_str,last_value_func,expected_str",
    [
        # String datetime - ISO format
        (3600, "2023-01-01T12:00:00Z", max, "2023-01-01T11:00:00Z"),
        (3600, "2023-01-01T12:00:00Z", min, "2023-01-01T13:00:00Z"),
        (7200, "2023-01-01T12:00:00+05:00", max, "2023-01-01T10:00:00+05:00"),
        (1800, "2023-01-01T12:00:00-03:00", min, "2023-01-01T12:30:00-03:00"),
        # String datetime - without timezone (naive)
        (3600, "2023-01-01T12:00:00", max, "2023-01-01T11:00:00"),
        (3600, "2023-01-01T12:00:00", min, "2023-01-01T13:00:00"),
        # String date - YYYY-MM-DD format
        (1, "2023-01-15", max, "2023-01-14"),
        (1, "2023-01-15", min, "2023-01-16"),
        (7, "2023-01-15", max, "2023-01-08"),
        (7, "2023-01-15", min, "2023-01-22"),
        # String date - YYYYMMDD format
        (1, "20230115", max, "20230114"),
        (1, "20230115", min, "20230116"),
        # ISO week dates crossing a year boundary where the week-numbering year
        # differs from the calendar year (regression guard for %V vs %W detection)
        (604800, "2026-W01", max, "2025-W52"),
        (604800, "2026-W01", min, "2026-W02"),
        (86400, "2026-W01-1", min, "2026-W01-2"),
        (604800, "2020-W53", min, "2021-W01"),
        (604800, "2021-W01", max, "2020-W53"),
    ],
    ids=[
        "string_datetime_utc_max_1hour",
        "string_datetime_utc_min_1hour",
        "string_datetime_plus5_max_2hour",
        "string_datetime_minus3_min_30min",
        "string_datetime_naive_max_1hour",
        "string_datetime_naive_min_1hour",
        "string_date_iso_max_1day",
        "string_date_iso_min_1day",
        "string_date_iso_max_7days",
        "string_date_iso_min_7days",
        "string_date_compact_max_1day",
        "string_date_compact_min_1day",
        "string_week_iso_max_boundary",
        "string_week_iso_min_boundary",
        "string_week_iso_day_min",
        "string_week_iso_53_min",
        "string_week_iso_max_to_53",
    ],
)
def test_apply_lag_to_str_value(
    lag: Union[int, float], value_str: str, last_value_func: LastValueFunc[Any], expected_str: str
) -> None:
    result = _apply_lag_to_value(lag, value_str, last_value_func)

    assert result == expected_str
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "lag,value,last_value_func",
    [
        # Numeric values
        (10, 100, max),
        (10, 100, min),
        (5.5, 50.5, max),
        (5.5, 50.5, min),
        (0, 42, max),
        (0, 42, min),
    ],
    ids=[
        "int_max_10",
        "int_min_10",
        "float_max_5.5",
        "float_min_5.5",
        "int_max_zero_lag",
        "int_min_zero_lag",
    ],
)
def test_apply_lag_to_value_numeric(
    lag: Union[int, float], value: Union[int, float], last_value_func: LastValueFunc[Any]
):
    result = _apply_lag_to_value(lag, value, last_value_func)

    if last_value_func == max:
        expected = value - lag
    else:
        expected = value + lag

    assert result == expected
    assert type(result) is type(value)


@pytest.mark.parametrize(
    "lag,value,last_value_func",
    [
        # Unsupported types
        (10, "invalid_date_string", max),
        (10, ["list"], max),
        (10, {"dict": "value"}, max),
        (10, None, max),
    ],
    ids=[
        "invalid_string",
        "list_type",
        "dict_type",
        "none_type",
    ],
)
def test_apply_lag_to_value_unsupported_types(
    lag: Union[int, float], value: str, last_value_func: LastValueFunc[Any]
):
    with pytest.raises(ValueError):
        _apply_lag_to_value(lag, value, last_value_func)
    # assert val_ex.value.args[0] == value


@pytest.mark.parametrize(
    "lag,value,last_value_func,expected_tz_preserved",
    [
        # Test timezone preservation
        (3600, "2023-01-01T12:00:00+05:00", max, True),
        (3600, "2023-01-01T12:00:00Z", max, True),
        (3600, "2023-01-01T12:00:00", max, False),  # Naive datetime
    ],
    ids=[
        "preserve_plus5_timezone",
        "preserve_utc_timezone",
        "naive_no_timezone",
    ],
)
def test_apply_lag_to_value_timezone_preservation(
    lag: Union[int, float], value: str, last_value_func, expected_tz_preserved: bool
):
    result = _apply_lag_to_value(lag, value, last_value_func)
    assert isinstance(result, str)

    # Parse both original and result to check timezone info
    parsed_original = ensure_pendulum_datetime_non_utc(value)
    parsed_result = ensure_pendulum_datetime_non_utc(result)

    if expected_tz_preserved:
        assert parsed_result.timezone == parsed_original.timezone
    else:
        # For naive datetimes, both should be naive
        assert parsed_result.timezone is None or str(parsed_result.timezone) == "UTC"


def test_apply_lag_to_value_edge_cases():
    """Test edge cases like leap years, DST transitions, etc."""

    # Leap year - February 29th
    leap_date = date(2024, 3, 1)
    result = _apply_lag_to_value(365, leap_date, max)  # Go back 365 days
    expected = date(2023, 3, 2)  # one day more
    assert result == expected

    # Month boundary crossing
    month_boundary = datetime(2023, 3, 1, 0, 0, 0)
    result = _apply_lag_to_value(3600, month_boundary, max)  # Go back 1 hour
    expected = datetime(2023, 2, 28, 23, 0, 0)
    assert result == expected

    # Year boundary crossing
    year_boundary = datetime(2023, 1, 1, 0, 0, 0)
    result = _apply_lag_to_value(3600, year_boundary, max)  # Go back 1 hour
    expected = datetime(2022, 12, 31, 23, 0, 0)
    assert result == expected


@pytest.mark.parametrize(
    "cursor_unit,value,lag,last_value_func,expected",
    [
        # cursor_unit="date" forces lag in DAYS, even for datetime-shaped strings
        ("date", "2026-08-27T00:00:00Z", 28, max, "2026-07-30T00:00:00Z"),
        ("date", "2026-08-27", 1, max, "2026-08-26"),
        # cursor_unit="datetime" forces lag in SECONDS, even for date-shaped strings
        (
            "datetime",
            "2026-08-27",
            1,
            max,
            "2026-08-26",
        ),  # midnight minus 1s → 2026-08-26T23:59:59 → re-emitted as date
        ("datetime", "2026-08-27T00:00:00Z", 3600, max, "2026-08-26T23:00:00Z"),
        # None + str: current behavior preserved (inferred from format)
        (None, "2026-08-27", 1, max, "2026-08-26"),  # date format → days
        (None, "2026-08-27T00:00:00Z", 1, max, "2026-08-26T23:59:59Z"),  # datetime format → seconds
        # Typed objects with explicit cursor_unit
        ("date", date(2023, 1, 15), 1, max, date(2023, 1, 14)),
        ("datetime", datetime(2023, 1, 1, 12, 0, 0), 3600, max, datetime(2023, 1, 1, 11, 0, 0)),
    ],
    ids=[
        "date_unit_datetime_string_28days",
        "date_unit_date_string_1day",
        "datetime_unit_date_string_1sec",
        "datetime_unit_datetime_string_1hour",
        "none_unit_date_string_inferred_days",
        "none_unit_datetime_string_inferred_seconds",
        "date_unit_date_object_1day",
        "datetime_unit_datetime_object_1hour",
    ],
)
def test_apply_lag_to_value_cursor_unit(
    cursor_unit: str, value: Any, lag: float, last_value_func: Callable, expected: Any
) -> None:
    """Test that cursor_unit parameter correctly determines lag unit."""
    result = _apply_lag_to_value(lag, value, last_value_func, cursor_unit)
    assert result == expected


def test_apply_lag_to_value_warning_on_inferred_unit(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a warning is emitted when cursor_unit is None and value is a string."""
    import logging

    # Enable propagation so caplog can capture dlt logger messages
    dlt_logger = logging.getLogger("dlt")
    original_propagate = dlt_logger.propagate
    dlt_logger.propagate = True

    try:
        with caplog.at_level(logging.WARNING, logger="dlt"):
            # No cursor_unit, string value → should warn
            _apply_lag_to_value(1, "2026-08-27", max, cursor_unit=None)

        assert any("inferred" in record.message.lower() for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)
    finally:
        dlt_logger.propagate = original_propagate


def test_apply_lag_to_value_no_warning_with_cursor_unit(caplog: pytest.LogCaptureFixture) -> None:
    """Test that NO warning is emitted when cursor_unit is explicitly set."""
    import logging

    # Enable propagation so caplog can capture dlt logger messages
    dlt_logger = logging.getLogger("dlt")
    original_propagate = dlt_logger.propagate
    dlt_logger.propagate = True

    try:
        with caplog.at_level(logging.WARNING, logger="dlt"):
            # Explicit cursor_unit="date" → should NOT warn
            _apply_lag_to_value(1, "2026-08-27", max, cursor_unit="date")

        assert not any("inferred" in record.message.lower() for record in caplog.records)
    finally:
        dlt_logger.propagate = original_propagate
