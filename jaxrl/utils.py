import os
import re
from typing import List, Optional

import gtimer as gt
from tensorboardX import SummaryWriter
import yaml

from jaxrl.evaluation import collect_trajectories


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


class Logger:
    def __init__(self, flags):
        config = flags.config
        if config.use_recurrent_policy:
            config.learner_kwargs.update(config.recurrent_policy_kwargs)
            config.learner_kwargs.use_recurrent_policy = True
        else:
            config.learner_kwargs.update(config.policy_kwargs)
            config.learner_kwargs.use_recurrent_policy = False
        config.learner_kwargs.update(config.meta_kwargs)
        self.config = config

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        flags.append_flags_into_file(os.path.join(self.config.save_dir, 'flags'))

        self.summary_writer = SummaryWriter(os.path.join(self.config.save_dir, 'tb'))

        if self.config.save_replay:
            replay_dir = self.get_replay_dir()
            if not os.path.exists(replay_dir):
                os.makedirs(replay_dir)

        interval_keys = ['eval', 'log', 'replay', 'save']
        intervals = [getattr(config, f'{key}_interval') for key in interval_keys]
        self.step_counter = StepCounter(interval_keys, intervals)
        self.it = 0

    def get_replay_dir(self) -> Optional[str]:
        if self.config.save_replay:
            replay_dir = os.path.join(self.config.save_dir, 'replay')
        else:
            replay_dir = None
        return replay_dir

    def if_not_exausted(self) -> bool:
        return self.step_counter.total_steps < self.config.max_steps

    def if_save_replay(self) -> bool:
        return self.step_counter.check_key('replay')

    def get_replay_prefix(self) -> str:
        return f'step_{self.step_counter.total_steps}_'

    def update_counts(self, counts: int):
        self.step_counter.update(counts)
        self.it += 1

    def log_periodically(self, update_info: dict, rollout_info: dict):
        if self.step_counter.check_key('log'):
            time_report_raw = gt.report(include_stats=False,
                                    delim_mode=True)
            times = dict(re.findall('\n(\S+)\t(\d+\.\d+)', time_report_raw))
            for k, v in times.items():
                self.summary_writer.add_scalar(f'time_{k}', float(v), self.it)
            self.summary_writer.add_scalar(f'training/total_steps', self.step_counter.total_steps, self.it)
            for k, v in update_info.items():
                self.summary_writer.add_scalar(f'training/{k}', v, self.it)
            for k, v in rollout_info.items():
                self.summary_writer.add_scalar(f'training/{k}', v, self.it)
            self.summary_writer.flush()
            gt.stamp('log')

    def eval_periodically(self, envs, agent):
        if self.step_counter.check_key('eval'):
            num_trajectories_per_env = self.config.eval_episodes // self.config.num_envs + 1
            _, eval_info = collect_trajectories(envs, agent,
                                                num_trajectories_per_env=num_trajectories_per_env,
                                                distribution='det')
            for k, v in eval_info.items():
                self.summary_writer.add_scalar(f'eval/{k}', v, self.it)
            gt.stamp('eval')

    def save_periodically(self, agent):
        if self.step_counter.check_key('save'):
            dump_dir = os.path.join(self.config.save_dir, 'models')
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            model_path_prefix = os.path.join(dump_dir, f'step_{self.step_counter.total_steps}')
            agent.save(model_path_prefix)
            gt.stamp('save')
