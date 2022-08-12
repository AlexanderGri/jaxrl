from typing import List

import yaml


def load_config(path):
    with open(path, 'r') as f:
        raw_text = f.read()
    config_text = raw_text.split('\n\n')[0][9:]
    config = yaml.unsafe_load(config_text)
    return config


class StepCounter:
    def __init__(self, keys: List[str], intervals: List[int]):
        self.keys = keys
        self.intervals = intervals
        self.next_moments = [1] * len(self.keys)
        self.interval_ends = [False] * len(self.keys)
        self.total_steps = 0

    def update(self, new_steps: int):
        self.total_steps += new_steps
        for i, (next_moment, interval) in enumerate(zip(self.next_moments,
                                                        self.intervals)):
            interval_end = (self.total_steps >= next_moment * interval)
            self.interval_ends[i] = interval_end
            if interval_end:
                self.next_moments[i] = self.total_steps // interval + 1

    def check_key(self, key: str) -> bool:
        return self.interval_ends[self.keys.index(key)]
