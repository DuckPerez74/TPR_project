from datetime import datetime, timedelta, timezone
from typing import Tuple


def get_time_range(current_time: datetime, window_minutes: int) -> Tuple[datetime, datetime]:
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    end_time = current_time
    start_time = current_time - timedelta(minutes=window_minutes)

    return start_time, end_time
