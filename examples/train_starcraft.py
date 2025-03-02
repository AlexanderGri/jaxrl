import logging
import os
from typing import Tuple
import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from jax import numpy as jnp
from tensorboardX import SummaryWriter

from jaxrl.agents import PGLearner
from jaxrl.datasets import PaddedTrajectoryData
from smac.env import StarCraft2Env

FLAGS = flags.FLAGS

flags.DEFINE_string('map_name', '2s3z', 'Map name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 45, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1, 'Logging interval.')
flags.DEFINE_integer('replay_interval', 50, 'Replay interval.')
flags.DEFINE_integer('eval_interval', 100, 'Eval interval.')
flags.DEFINE_integer('trajectories_per_update', 5, 'Number of trajectories collected per each step')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e5), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_replay', True, 'Save videos during evaluation.')
flags.DEFINE_boolean('use_recurrent_policy', True, 'Use recurrent policy')
config_flags.DEFINE_config_file(
    'config',
    'configs/pg_recurrent_default.py' if FLAGS.use_recurrent_policy else 'configs/pg_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


InfoDict = dict


def collect_trajectories(env: StarCraft2Env, agent: PGLearner,
                         n_trajectories: int = 1, save_replay: bool = False,
                         use_recurrent_policy: bool = False) \
        -> Tuple[PaddedTrajectoryData, InfoDict]:
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
            observations = np.stack(env.get_obs())[np.newaxis, np.newaxis]
            available_actions = np.stack(env.get_avail_actions())[np.newaxis, np.newaxis]
            if use_recurrent_policy:
                # adding two leading dimensions accounting for trajectories and steps
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

    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    if FLAGS.save_replay:
        replay_dir = os.path.join(FLAGS.save_dir, 'replay')
    else:
        replay_dir = None

    env = StarCraft2Env(map_name=FLAGS.map_name, replay_dir=replay_dir)

    env_info = env.get_env_info()
    dummy_states_batch = np.ones((1, 1, env_info['state_shape']))
    dummy_observations_batch = np.ones((1, 1, 1, env_info['obs_shape']))
    dummy_available_actions_batch = np.zeros((1, 1, 1, env_info["n_actions"],), dtype=bool)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)

    agent = PGLearner(FLAGS.seed,
                      dummy_states_batch,
                      dummy_observations_batch,
                      dummy_available_actions_batch,
                      env_info["n_actions"],
                      env_info["episode_limit"],
                      **kwargs)

    total_steps = 0
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        data, info = collect_trajectories(env, agent,
                                          n_trajectories=FLAGS.trajectories_per_update,
                                          save_replay=i % FLAGS.replay_interval == 0,
                                          use_recurrent_policy=FLAGS.use_recurrent_policy)
        total_steps += info['iter_steps']
        for _ in range(FLAGS.updates_per_step):
            update_info = agent.update(data)

        if i % FLAGS.log_interval == 0:
            summary_writer.add_scalar(f'training/total_steps', total_steps, i)
            for k, v in update_info.items():
                summary_writer.add_scalar(f'training/{k}', v, i)
            for k, v in info.items():
                summary_writer.add_scalar(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(env, agent, n_trajectories=FLAGS.eval_episodes,
                                 use_recurrent_policy=FLAGS.use_recurrent_policy)
            for k, v in eval_info.items():
                summary_writer.add_scalar(f'eval/{k}', v, i)

if __name__ == '__main__':
    app.run(main)
