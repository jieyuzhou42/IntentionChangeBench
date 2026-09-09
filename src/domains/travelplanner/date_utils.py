from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Any, Dict, List, Optional


def inclusive_date_count(start_date: Any, end_date: Any) -> Optional[int]:
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    if start is None or end is None or end < start:
        return None
    return (end - start).days + 1


def sync_query_dates(
    query_data: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply intention-level date fields to TravelPlanner's date-list format."""
    if constraints.get("date") is not None:
        query_data["date"] = copy.deepcopy(constraints["date"])
        return query_data

    start = _parse_iso_date(constraints.get("start_date"))
    end = _parse_iso_date(constraints.get("end_date"))
    days = _positive_int(constraints.get("days"))

    if start is not None and end is not None and end >= start:
        query_data["date"] = _date_range(start, end)
    elif start is not None and days is not None:
        query_data["date"] = [
            (start + timedelta(days=offset)).isoformat()
            for offset in range(days)
        ]
    elif end is not None and days is not None:
        first = end - timedelta(days=days - 1)
        query_data["date"] = _date_range(first, end)
    return query_data


def _parse_iso_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _date_range(start: date, end: date) -> List[str]:
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]
