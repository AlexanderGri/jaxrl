import json
import logging
import os
import re
import random

import gtimer as gt
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import PGLearner, MetaPGLearner, MetaNoRewardPGLearner
from jaxrl.evaluation import collect_trajectories
from jaxrl.utils import StepCounter
from smac.env import StarCraft2Env

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    os.path.join(os.path.dirname(__file__), 'config.py'),
    lock_config=False)


def main(_):
    # disable check_types warnings
    logger = logging.getLogger("root")
    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()
    logger.addFilter(CheckTypesFilter())

    config = FLAGS.config
    if config.use_recurrent_policy:
        config.learner_kwargs.update(config.recurrent_policy_kwargs)
        config.learner_kwargs.use_recurrent_policy = True
    else:
        config.learner_kwargs.update(config.policy_kwargs)
        config.learner_kwargs.use_recurrent_policy = False
    if config.use_meta_rewards:
        if not config.meta_kwargs.no_rewards_in_meta:
            config.learner_kwargs.update(config.meta_kwargs)
            delattr(config.learner_kwargs, 'no_rewards_in_meta')
    FLAGS.append_flags_into_file(os.path.join(config.save_dir, 'flags'))

    summary_writer = SummaryWriter(os.path.join(config.save_dir, 'tb'))

    if config.save_replay:
        replay_dir = os.path.join(config.save_dir, 'replay')
        if not os.path.exists(replay_dir):
            os.makedirs(replay_dir)
    else:
        replay_dir = None

    env = StarCraft2Env(map_name=config.map_name, replay_dir=replay_dir,
                        reward_only_positive=FLAGS.config.reward_only_positive)

    env_info = env.get_env_info()
    dummy_states_batch = np.ones((1, 1, env_info['state_shape']))
    if FLAGS.config.one_hot_to_observations:
        obs_shape = env_info['obs_shape'] + env_info['n_agents']
    else:
        obs_shape = env_info['obs_shape']
    dummy_observations_batch = np.ones((1, 1, env_info['n_agents'], obs_shape))
    dummy_available_actions_batch = np.zeros((1, 1, env_info['n_agents'], env_info["n_actions"],), dtype=bool)

    np.random.seed(config.seed)
    random.seed(config.seed)


    learner_kwargs = dict(config.learner_kwargs)
    if config.use_meta_rewards:
        if config.meta_kwargs.no_rewards_in_meta:
            Learner = MetaNoRewardPGLearner
        else:
            Learner = MetaPGLearner
            learner_kwargs["n_agents"] = env_info["n_agents"]
    else:
        Learner = PGLearner

    agent = Learner(config.seed,
                    dummy_states_batch,
                    dummy_observations_batch,
                    dummy_available_actions_batch,
                    env_info["n_actions"],
                    env_info["episode_limit"],
                    **learner_kwargs)

    if config.model_load_path != '':
        agent.load(config.model_load_path)

    interval_keys = ['eval', 'log', 'replay', 'save']
    intervals = [getattr(config, f'{key}_interval') for key in interval_keys ]
    step_counter = StepCounter(interval_keys, intervals)
    it = 0
    gt.reset_root()
    gt.rename_root('RL_algorithm')
    gt.set_def_unique(False)
    prev_data = None
    prev_actor = None
    while step_counter.total_steps < config.max_steps:
        data, rollout_info = collect_trajectories(env, agent,
                                                  n_trajectories=config.trajectories_per_update,
                                                  use_recurrent_policy=config.use_recurrent_policy,
                                                  save_replay=step_counter.check_key('replay'),
                                                  replay_prefix=f'step_{step_counter.total_steps}_',
                                                  one_hot_to_observations=FLAGS.config.one_hot_to_observations)
        if config.penalty_per_step is not None:
            data = data._replace(rewards=(data.rewards - config.penalty_per_step))
        gt.stamp('collect_data')
        step_counter.update(rollout_info['iter_steps'])
        it += 1
        if config.use_meta_rewards:
            if config.meta_kwargs.no_rewards_in_meta:
                update_info = agent.update(prev_data, data)
                prev_data = data
            else:
                update_only_intrinsic = (FLAGS.config.stop_agent_training_at != -1) and \
                                        (FLAGS.config.stop_agent_training_at < step_counter.total_steps)
                update_info = agent.update_except_actor(prev_data, data, prev_actor,
                                                        update_only_intrinsic)
                if update_only_intrinsic:
                    agent.actor = prev_actor
                    prev_data, _ = collect_trajectories(env, agent,
                                                        n_trajectories=config.trajectories_per_update,
                                                        use_recurrent_policy=config.use_recurrent_policy,
                                                        one_hot_to_observations=FLAGS.config.one_hot_to_observations)
                    update_info_actor = agent.update_actor(prev_data)
                else:
                    prev_actor = agent.actor
                    update_info_actor = agent.update_actor(data)
                    prev_data = data
                update_info.update(update_info_actor)
        else:
            update_info = agent.update(data)
        gt.stamp('train')

        if step_counter.check_key('log'):
            time_report_raw = gt.report(include_stats=False,
                                    delim_mode=True)
            times = dict(re.findall('\n(\S+)\t(\d+\.\d+)', time_report_raw))
            for k, v in times.items():
                summary_writer.add_scalar(f'time_{k}', float(v), it)
            summary_writer.add_scalar(f'training/total_steps', step_counter.total_steps, it)
            for k, v in update_info.items():
                summary_writer.add_scalar(f'training/{k}', v, it)
            for k, v in rollout_info.items():
                summary_writer.add_scalar(f'training/{k}', v, it)
            summary_writer.flush()
            gt.stamp('log')

        if step_counter.check_key('eval'):
            _, eval_info = collect_trajectories(env, agent,
                                                n_trajectories=config.eval_episodes,
                                                use_recurrent_policy=config.use_recurrent_policy,
                                                one_hot_to_observations=FLAGS.config.one_hot_to_observations,
                                                distribution='det')
            for k, v in eval_info.items():
                summary_writer.add_scalar(f'eval/{k}', v, it)
            gt.stamp('eval')
        if step_counter.check_key('save'):
            dump_dir = os.path.join(config.save_dir, 'models')
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            model_path_prefix = os.path.join(dump_dir, f'step_{step_counter.total_steps}')
            agent.save(model_path_prefix)
            gt.stamp('save')


if __name__ == '__main__':
    app.run(main)
