from datetime import datetime
from typing import List


class Scheduler:
    def __init__(self, config: dict):
        self.execution_schedule = config.get('execution_schedule', {})
        self.windows_schedule = self.execution_schedule.get('windows', {})
        self.metrics_config = config.get('metrics', {})

    def should_execute(self, current_time: datetime) -> bool:
        minute = current_time.minute

        for window_minutes in self.windows_schedule.values():
            if minute in window_minutes:
                return True

        return False

    def get_active_windows(self, current_time: datetime) -> List[int]:
        minute = current_time.minute
        active_windows = []

        for window_str, schedule in self.windows_schedule.items():
            if minute in schedule:
                active_windows.append(int(window_str))

        return sorted(active_windows, reverse=True)

    def should_save_metrics(self, layer: str, window: int, minute: int) -> bool:
        layer_config = self.metrics_config.get('layers', {}).get(layer, {})
        save_schedule = layer_config.get('save_schedule', {})

        window_schedule = save_schedule.get(str(window))

        if window_schedule == "always":
            return True

        if isinstance(window_schedule, list):
            return minute in window_schedule

        return False
