import os
import re
import random

import gtimer as gt
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl.algorithm import MetaPGLearner
from jaxrl.evaluation import collect_trajectories
from jaxrl.utils import Logger
from jaxrl.vec_env import SubprocVecStarcraft


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    os.path.join(os.path.dirname(__file__), 'config.py'),
    lock_config=False)


def init_dummy_arrays(env_info):
    states = np.ones((1, 1, env_info['state_dim']))
    observations = np.ones((1, 1, env_info['n_agents'], env_info['obs_dim']))
    available_actions = np.zeros((1, 1, env_info['n_agents'], env_info["n_actions"],), dtype=bool)
    return states, observations, available_actions


def main(_):
    logger = Logger(FLAGS)
    config = logger.config

    env_kwargs = {'map_name': config.map_name,
                  'replay_dir': logger.get_replay_dir(),
                  'reward_only_positive': config.reward_only_positive,
                  'continuing_episode': config.continuing_episode}

    envs = SubprocVecStarcraft(num_envs=config.num_envs,
                               one_hot_to_observations=config.one_hot_to_observations,
                               **env_kwargs)

    env_info = envs.get_info()
    states, observations, available_actions = init_dummy_arrays(env_info)

    np.random.seed(config.seed)
    random.seed(config.seed)

    agent = MetaPGLearner(seed=config.seed,
                          states=states,
                          observations=observations,
                          available_actions=available_actions,
                          n_actions=env_info["n_actions"],
                          time_limit=env_info["time_limit"],
                          n_agents=env_info["n_agents"],
                          **config.learner_kwargs)

    if config.model_load_path != '':
        agent.load(config.model_load_path)

    gt.reset_root()
    gt.rename_root('RL_algorithm')
    gt.set_def_unique(False)

    assert (not config.update_only_reward) or config.meta_kwargs.sampling_scheme == 'double'
    if config.meta_kwargs.sampling_scheme == 'reuse' or config.meta_kwargs.sampling_scheme == 'importance_sampling':
        train_one_batch_per_update(envs, agent, logger)
    elif config.meta_kwargs.sampling_scheme == 'double':
        train_two_batches_per_update(envs, agent, logger)
    envs.close()


def train_one_batch_per_update(envs, agent, logger):
    prev_data = None
    prev_actor = None
    while logger.if_not_exausted():
        data, rollout_info = collect_trajectories(envs, agent,
                                                  num_trajectories_per_env=logger.config.num_trajectories_per_env_per_update,
                                                  save_replay=logger.if_save_replay(),
                                                  replay_prefix=logger.get_replay_prefix())
        logger.update_counts(rollout_info['iter_steps'])
        gt.stamp('collect_data')

        if prev_data is None:
            update_info = {}
        else:
            update_info = agent.update_except_actor(prev_data=prev_data, data=data, prev_actor=prev_actor)
        gt.stamp('train_except_actor')

        prev_actor = agent.actor
        prev_data = data
        update_info_actor = agent.update_actor(data)
        update_info.update(update_info_actor)
        gt.stamp('train_actor')

        logger.log_periodically(update_info, rollout_info)
        logger.eval_periodically(envs, agent)
        logger.save_periodically(agent)


def train_two_batches_per_update(envs, agent, logger):
    freezed_actor = agent.actor
    while logger.if_not_exausted():
        if logger.config.update_only_reward:
            agent.actor = freezed_actor
        prev_actor = agent.actor
        prev_data, prev_rollout_info = collect_trajectories(envs, agent,
                                                            num_trajectories_per_env=logger.config.num_trajectories_per_env_per_update,
                                                            save_replay=logger.if_save_replay(),
                                                            replay_prefix=logger.get_replay_prefix())
        gt.stamp('collect_data')

        update_info_actor = agent.update_actor(prev_data)
        gt.stamp('train_actor')

        data, rollout_info = collect_trajectories(envs, agent,
                                                  num_trajectories_per_env=logger.config.num_trajectories_per_env_per_update)
        logger.update_counts(prev_rollout_info['iter_steps'] + rollout_info['iter_steps'])
        gt.stamp('collect_data')

        if logger.config.update_only_reward:
            update_info = agent.update_only_reward(prev_data=prev_data, data=data, prev_actor=prev_actor)
        else:
            update_info = agent.update_except_actor(prev_data=prev_data, data=data, prev_actor=prev_actor)
        update_info.update(update_info_actor)
        gt.stamp('train_except_actor')

        logger.log_periodically(update_info, rollout_info)
        logger.eval_periodically(envs, agent)
        logger.save_periodically(agent)


if __name__ == '__main__':
    app.run(main)
