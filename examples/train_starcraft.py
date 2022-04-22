import json
import logging
import os
import re
from typing import Tuple, Union
import random

import gtimer as gt
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from jax import numpy as jnp
from tensorboardX import SummaryWriter

from jaxrl.agents import PGLearner, MetaPGLearner
from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.utils import StepCounter
from smac.env import StarCraft2Env

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    os.path.join(os.path.dirname(__file__), 'config.py'),
    lock_config=False)


InfoDict = dict


def collect_trajectories(env: StarCraft2Env, agent: Union[PGLearner, MetaPGLearner],
                         n_trajectories: int = 1, save_replay: bool = False,
                         replay_prefix: str = None, use_recurrent_policy: bool = False) \
        -> Tuple[PaddedTrajectoryData, InfoDict]:
    env.replay_prefix = replay_prefix
    env_info = env.get_env_info()
    state_dim = env_info['state_shape']
    obs_dim = env_info['obs_shape']
    time_limit = env_info['episode_limit']
    n_actions = env_info['n_actions']
    n_agents = env_info['n_agents']

    # per trajectory
    dones = np.zeros((n_trajectories,), dtype=bool)
    # per trajectory per step
    states = np.zeros((n_trajectories, time_limit, state_dim))
    next_states = np.zeros((n_trajectories, time_limit, state_dim))
    rewards = np.zeros((n_trajectories, time_limit))
    all_agents_alive = np.zeros((n_trajectories, time_limit), dtype=bool)
    # per trajectory per step per agent
    observations = np.zeros((n_trajectories, time_limit, n_agents, obs_dim))
    next_observations = np.zeros((n_trajectories, time_limit, n_agents, obs_dim))
    actions = np.zeros((n_trajectories, time_limit, n_agents), dtype=int)
    available_actions = np.zeros((n_trajectories, time_limit, n_agents, n_actions), dtype=bool)
    agent_alive = np.zeros((n_trajectories, time_limit, n_agents), dtype=bool)

    returns = []
    total_steps = 0
    end_info = {"dead_allies": [],
                "dead_enemies": [],
                "battle_won": []}
    for traj_ind in range(n_trajectories):
        step = 0
        ret = 0
        done = False
        env.reset()
        if save_replay:
            env.save_replay()
        if use_recurrent_policy:
            carry = agent.initialize_carry(1, n_agents)
        while not done:
            ii = (traj_ind, step, ...)
            states[ii] = env.get_state()
            observations[ii] = np.stack(env.get_obs())
            available_actions[ii] = env.get_avail_actions()
            if use_recurrent_policy:
                # adding two leading dimensions accounting for trajectories and steps
                carry, cur_actions = agent.sample_actions(
                    observations[ii][np.newaxis, np.newaxis],
                    available_actions[ii][np.newaxis, np.newaxis],
                    carry)
                actions[ii] = cur_actions[0, 0]
            else:
                actions[ii] = agent.sample_actions(observations[ii],
                                                   available_actions[ii])
            all_agents_alive[ii] = True
            agent_alive[ii] = [env.get_unit_by_id(i).health > 0 for i in range(n_agents)]
            rewards[ii], done, step_info = env.step(actions[ii])
            next_states[ii] = env.get_state()
            next_observations[ii] = np.stack(env.get_obs())
            step += 1
            ret += rewards[ii]
        if step_info.get("episode_limit", False):
            done = False
        for k in end_info:
            # sometimes step() return empty info
            if k in step_info:
                end_info[k].append(step_info[k])
        dones[traj_ind] = done
        returns.append(ret)
        total_steps += step
    data_to_jax = dict(
        states=states,
        observations=observations,
        next_states=next_states,
        next_observations=next_observations,
        actions=actions,
        available_actions=available_actions,
        rewards=rewards,
        all_agents_alive=all_agents_alive,
        agent_alive=agent_alive,
        dones=dones,)
    data_jax = {k: jnp.array(v) for k, v in data_to_jax.items()}
    data = PaddedTrajectoryData(
        length=time_limit,
        **data_jax
    )
    info = {'returns': np.mean(returns),
            'iter_steps': total_steps}
    for k, arr in end_info.items():
        info[k] = np.mean(arr)
    return data, info


def evaluate(env: StarCraft2Env, agent: PGLearner, n_trajectories: int = 1,
             use_recurrent_policy: bool = False) -> InfoDict:
    returns = []
    total_steps = 0
    end_info = {"dead_allies": [],
                "dead_enemies": [],
                "battle_won": []}
    for traj_ind in range(n_trajectories):
        step = 0
        ret = 0
        done = False
        env.reset()
        if use_recurrent_policy:
            carry = agent.initialize_carry(1, env.get_env_info()['n_agents'])
        while not done:
            # adding two leading dimensions accounting for trajectories and steps
            observations = np.stack(env.get_obs())[np.newaxis, np.newaxis]
            available_actions = np.stack(env.get_avail_actions())[np.newaxis, np.newaxis]
            if use_recurrent_policy:
                carry, actions = agent.sample_actions(
                    observations,
                    available_actions,
                    carry,
                    distribution='det')
            else:
                actions = agent.sample_actions(observations,
                                               available_actions,
                                               distribution='det')
            actions = actions[0, 0]
            reward, done, step_info = env.step(actions)
            step += 1
            ret += reward
        for k in end_info:
            # sometimes step() return empty info
            if k in step_info:
                end_info[k].append(step_info[k])
        returns.append(ret)
        total_steps += step
    info = {'returns': np.mean(returns),
            'iter_steps': total_steps}
    for k, arr in end_info.items():
        info[k] = np.mean(arr)
    return info


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
        config.learner_kwargs.update(config.meta_kwargs)
    FLAGS.append_flags_into_file(os.path.join(config.save_dir, 'flags'))

    summary_writer = SummaryWriter(os.path.join(config.save_dir, 'tb'))

    if config.save_replay:
        replay_dir = os.path.join(config.save_dir, 'replay')
        if not os.path.exists(replay_dir):
            os.makedirs(replay_dir)
    else:
        replay_dir = None

    env = StarCraft2Env(map_name=config.map_name, replay_dir=replay_dir)

    env_info = env.get_env_info()
    dummy_states_batch = np.ones((1, 1, env_info['state_shape']))
    dummy_observations_batch = np.ones((1, 1, 1, env_info['obs_shape']))
    dummy_available_actions_batch = np.zeros((1, 1, 1, env_info["n_actions"],), dtype=bool)

    np.random.seed(config.seed)
    random.seed(config.seed)


    learner_kwargs = dict(config.learner_kwargs)
    if config.use_meta_rewards:
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

    interval_keys = ['eval', 'log', 'replay', 'save']
    intervals = [getattr(config, f'{key}_interval') for key in interval_keys ]
    step_counter = StepCounter(interval_keys, intervals)
    it = 0
    gt.reset_root()
    gt.rename_root('RL_algorithm')
    gt.set_def_unique(False)
    while step_counter.total_steps < config.max_steps:
        data, rollout_info = collect_trajectories(env, agent,
                                                  n_trajectories=config.trajectories_per_update,
                                                  use_recurrent_policy=config.use_recurrent_policy,
                                                  save_replay=step_counter.check_key('replay'),
                                                  replay_prefix=f'step_{step_counter.total_steps}_')
        gt.stamp('collect_data')
        step_counter.update(rollout_info['iter_steps'])
        it += 1
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
            eval_info = evaluate(env, agent,
                                 n_trajectories=config.eval_episodes,
                                 use_recurrent_policy=config.use_recurrent_policy)
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
