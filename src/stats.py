import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict

from src.config import DATE_FORMAT, DATETIME_FORMAT


def generate_train_stats(data_file) -> Dict[str, Any]:
    """
    Generate comprehensive train statistics from detection data.

    :return: A dictionary containing various statistical metrics about train detections
    :rtype: Dict[str, Any]
    """
    # Read current data
    with open(data_file, "r") as f:
        data = json.load(f)

    # Current date and time
    now = datetime.now()
    today_str = now.strftime(DATE_FORMAT)
    today_date = now.date()

    # Time period definitions
    time_periods = {
        "Morning": (6, 12),
        "Afternoon": (12, 18),
        "Evening": (18, 24),
        "Night": (0, 6),
    }
    return {
        "daily_counts": _compute_daily_counts(data),
        "today_count": _compute_today_count(data, today_str),
        "period_counts": _compute_period_counts(data, today_str, time_periods),
        "last_7_days_counts": _compute_last_7_days_counts(data, today_date),
        "weekday_averages": _compute_weekday_averages(data),
        "monthly_stats": _compute_monthly_stats(data, now),
        "today_str": today_str,
    }


def _compute_daily_counts(data: Dict) -> Dict[str, int]:
    """
    Compute the number of train detections per day.

    :param data: Detection data dictionary
    :type data: Dict
    :return: Dictionary with dates as keys and detection counts as values
    :rtype: Dict[str, int]
    """
    daily_counts = defaultdict(int)
    for detection in data.get("detections", []):
        dt = datetime.strptime(detection["timestamp"], DATETIME_FORMAT)
        daily_counts[dt.strftime(DATE_FORMAT)] += 1
    return dict(daily_counts)


def _compute_today_count(data: Dict, today_str: str) -> int:
    """
    Calculate the number of train detections for today.

    :param data: Detection data dictionary
    :type data: Dict
    :param today_str: Today's date string in YYYY-MM-DD format
    :type today_str: str
    :return: Number of train detections today
    :rtype: int
    """
    return sum(
        1 for detection in data.get("detections", [])
        if datetime.strptime(detection["timestamp"], DATETIME_FORMAT).strftime(DATE_FORMAT) == today_str
    )


def _compute_period_counts(data: Dict, today_str: str, time_periods: Dict) -> Dict[str, int]:
    """
    Compute train detection counts for different time periods of today.

    :param data: Detection data dictionary
    :type data: Dict
    :param today_str: Today's date string in YYYY-MM-DD format
    :type today_str: str
    :param time_periods: Dictionary defining time periods and their hour ranges
    :type time_periods: Dict
    :return: Dictionary with time periods and their detection counts
    :rtype: Dict[str, int]
    """
    period_counts = defaultdict(int)
    for detection in data.get("detections", []):
        dt = datetime.strptime(detection["timestamp"], DATETIME_FORMAT)
        if dt.strftime(DATE_FORMAT) == today_str:
            for period, (start, end) in time_periods.items():
                if start <= dt.hour < end:
                    period_counts[period] += 1
    return dict(period_counts)


def _compute_last_7_days_counts(data: Dict, today_date: date) -> Dict[str, int]:
    """
    Compute train detection counts for the last 7 days.

    :param data: Detection data dictionary
    :type data: Dict
    :param today_date: Current date
    :type today_date: date
    :return: Dictionary with dates and their detection counts for the last 7 days
    :rtype: Dict[str, int]
    """
    daily_counts = _compute_daily_counts(data)
    return {
        (today_date - timedelta(days=i)).strftime(DATE_FORMAT):
            daily_counts.get((today_date - timedelta(days=i)).strftime(DATE_FORMAT), 0)
        for i in range(7)
    }


def _compute_weekday_averages(data: Dict) -> Dict[str, int]:
    """
    Calculate average train detections by weekday.

    :param data: Detection data dictionary
    :type data: Dict
    :return: Dictionary with weekdays and their average detection counts
    :rtype: Dict[str, int]
    """
    # Collect counts for each weekday
    weekday_counts = defaultdict(list)
    for detection in data.get("detections", []):
        dt = datetime.strptime(detection["timestamp"], DATETIME_FORMAT)
        weekday_counts[dt.strftime("%A")].append(1)

    # Compute averages
    return {
        day: int(sum(counts) / len(counts))
        for day, counts in weekday_counts.items() if counts
    }


def _compute_monthly_stats(data: Dict, now: datetime) -> Dict[str, int]:
    """
    Compute monthly train detection statistics.

    :param data: Detection data dictionary
    :type data: Dict
    :param now: Current datetime
    :type now: datetime
    :return: Dictionary with total and average monthly train detections
    :rtype: Dict[str, int]
    """
    daily_counts = _compute_daily_counts(data)
    current_month = now.strftime("%Y-%m")

    monthly_counts = [
        count for date, count in daily_counts.items()
        if date.startswith(current_month)
    ]

    month_total = sum(monthly_counts)
    month_average = month_total // len(monthly_counts) if monthly_counts else 0

    return {
        "total": month_total,
        "average": month_average
    }


def _format_stats_message(stats: Dict[str, Any]) -> str:
    """
    Format train detection statistics into a human-readable message.

    :param stats: Dictionary containing train detection statistics
    :type stats: Dict[str, Any]
    :return: Formatted message string with statistical information
    :rtype: str
    """
    message = "📊 Train Statistics Report 📊\n\n"

    # Today's stats
    message += f"Today ({stats['today_str']}): {stats['today_count']} trains\n"
    message += f"Today's Average: {stats['today_count'] / 24:.2f} trains per hour\n\n"

    # Time period breakdown
    message += "Today's Breakdown:\n"
    for period, count in stats['period_counts'].items():
        message += f"• {period}: {count} trains\n"
    message += "\n"

    # Weekly totals
    message += "Last 7 Days:\n"
    for d, count in stats['last_7_days_counts'].items():
        message += f"{d}: {count} trains\n"
    message += "\n"

    # Weekly average by weekday
    message += "Weekday Averages:\n"
    for day, avg in stats['weekday_averages'].items():
        message += f"• {day}: {avg} trains\n"
    message += "\n"

    # Monthly stats
    message += f"Month Total: {stats['monthly_stats']['total']} trains\n"
    message += f"Month Average: {stats['monthly_stats']['average']} trains per day\n"

    return message
