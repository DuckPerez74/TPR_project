from datetime import datetime
from config import EXECUTION_SCHEDULE

def should_execute(current_time: datetime, window_minutes: int) -> bool:
    minute = current_time.minute

    if window_minutes not in EXECUTION_SCHEDULE:
        return False

    return minute in EXECUTION_SCHEDULE[window_minutes]

def get_execution_windows(current_time: datetime) -> list:
    windows = []
    for window_minutes, schedule in EXECUTION_SCHEDULE.items():
        if current_time.minute in schedule:
            windows.append(window_minutes)
    return windows
