import re
from datetime import datetime, timedelta, timezone

REMINDER_KEYWORDS = [
    "remind", "reminder", "todo", "to do", "don't forget",
    "dont forget", "remember to", "need to", "must",
    "follow up", "follow-up", "task", "action",
]

# Ordered by specificity — first match wins in simple cases
DATE_PATTERNS = [
    # Explicit dates: "2026-09-25", "09/25/2026", "25 September 2026"
    (r'\b(\d{4})-(\d{2})-(\d{2})\b', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    (r'\b(\d{2})/(\d{2})/(\d{4})\b', lambda m: f"{m.group(3)}-{m.group(1)}-{m.group(2)}"),
    (r'\b(\d{2})\.(\d{2})\.(\d{4})\b', lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
    # "25 September", "September 25" (uses current year)
    (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b',
     lambda m: _month_day(m.group(2), int(m.group(1)))),
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b',
     lambda m: _month_day(m.group(1), int(m.group(2)))),
    # Relative: "tomorrow", "today", "day after tomorrow"
    (r'\b(tomorrow)\b', lambda m: _relative("tomorrow")),
    (r'\b(today)\b', lambda m: _relative("today")),
    (r'\b(day after tomorrow)\b', lambda m: _relative("day_after_tomorrow")),
    (r'\b(next week)\b', lambda m: _relative("next_week")),
    (r'\b(next monday|next tuesday|next wednesday|next thursday|next friday|next saturday|next sunday)\b',
     lambda m: _next_weekday(m.group(1))),
    # "in 2 days", "in 3 days", "in 1 week"
    (r'\b(in\s+(\d+)\s+days?)\b', lambda m: _relative_days(int(m.group(2)))),
    (r'\b(in\s+(\d+)\s+weeks?)\b', lambda m: _relative_weeks(int(m.group(2)))),
    (r'\b(in\s+(\d+)\s+months?)\b', lambda m: _relative_months(int(m.group(2)))),
    # Time: "at 3pm", "at 15:00", "by 5 o'clock"
    (r'\b(at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)\b',
     lambda m: _extract_time(m.group(2), m.group(3), m.group(4))),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

def _month_day(month_name: str, day: int) -> str:
    now = datetime.now(timezone.utc)
    month = MONTH_MAP.get(month_name.lower())
    if month is None:
        return None
    try:
        dt = datetime(now.year, month, min(day, 28), tzinfo=timezone.utc)
        if dt < now.replace(hour=0, minute=0, second=0, microsecond=0):
            dt = dt.replace(year=now.year + 1)
        return dt.isoformat()
    except ValueError:
        return None

def _relative(rel: str) -> str:
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if rel == "today":
        return today.isoformat()
    if rel == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if rel == "day_after_tomorrow":
        return (today + timedelta(days=2)).isoformat()
    if rel == "next_week":
        return (today + timedelta(days=7)).isoformat()
    return None

def _next_weekday(text: str) -> str:
    now = datetime.now(timezone.utc)
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }
    target = weekday_map.get(text.lower().split()[-1])
    if target is None:
        return None
    days_ahead = target - now.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)).isoformat()

def _relative_days(n: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=n)).isoformat()

def _relative_weeks(n: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(weeks=n)).isoformat()

def _relative_months(n: int) -> str:
    now = datetime.now(timezone.utc)
    month = now.month + n
    year = now.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return datetime(year, month, now.day, tzinfo=timezone.utc).isoformat()

def _extract_time(hour: str, minute: str | None, ampm: str | None) -> str:
    h = int(hour)
    m = int(minute) if minute else 0
    if ampm and ampm.lower() == "pm" and h != 12:
        h += 12
    if ampm and ampm.lower() == "am" and h == 12:
        h = 0
    return f"{h:02d}:{m:02d}:00"

def extract_reminder_info(text: str) -> dict:
    """Scan note text for reminder indicators and a date.

    Returns:
        {
            "is_reminder": bool,
            "reminder_date": str | None,   # ISO datetime or date
            "time": str | None,           # HH:MM:SS if extracted
            "matched_keyword": str | None,
        }
    """
    text_lower = text.lower()
    matched_keyword = None
    for kw in REMINDER_KEYWORDS:
        if kw in text_lower:
            matched_keyword = kw
            break

    if not matched_keyword:
        return {"is_reminder": False, "reminder_date": None, "time": None, "matched_keyword": None}

    # Try to find a date in the text
    reminder_date = None
    for pattern, extractor in DATE_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            result = extractor(m)
            if result:
                reminder_date = result
                break

    # Try to find a time
    time_match = re.search(r'\b(at\s+\d{1,2}(?::\d{2})?\s*(am|pm)?)\b', text_lower)
    time_str = None
    if time_match:
        tm = time_match.group(0)
        parts = re.match(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', tm)
        if parts:
            time_str = _extract_time(parts.group(1), parts.group(2), parts.group(3))

    return {
        "is_reminder": True,
        "reminder_date": reminder_date,
        "time": time_str,
        "matched_keyword": matched_keyword,
    }
