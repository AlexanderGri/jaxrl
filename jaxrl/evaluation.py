from typing import Tuple, Union

from jax import numpy as jnp
import numpy as np
from smac.env import StarCraft2Env

from jaxrl.agents import PGLearner, MetaPGLearner
from jaxrl.datasets import PaddedTrajectoryData


def collect_trajectories(env: StarCraft2Env, agent: Union[PGLearner, MetaPGLearner],
                         n_trajectories: int = 1, save_replay: bool = False,
                         replay_prefix: str = None, use_recurrent_policy: bool = False,
                         one_hot_to_observations: bool = False, distribution='log_prob') \
        -> Tuple[PaddedTrajectoryData, dict]:
    env.replay_prefix = replay_prefix
    env_info = env.get_env_info()
    state_dim = env_info['state_shape']
    obs_dim = env_info['obs_shape']
    time_limit = env_info['episode_limit']
    n_actions = env_info['n_actions']
    n_agents = env_info['n_agents']
    if one_hot_to_observations:
        obs_dim += n_agents

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
    log_prob = np.zeros((n_trajectories, time_limit, n_agents))
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
            env_obs = np.stack(env.get_obs())
            if one_hot_to_observations:
                observations[ii] = np.concatenate((env_obs, np.eye(n_agents)), axis=1)
            else:
                observations[ii] = env_obs
            available_actions[ii] = env.get_avail_actions()
            if use_recurrent_policy:
                # adding two leading dimensions accounting for trajectories and steps
                carry, cur_actions, cur_log_prob = agent.sample_actions(
                    observations[ii][np.newaxis, np.newaxis],
                    available_actions[ii][np.newaxis, np.newaxis],
                    carry, distribution=distribution)
            else:
                cur_actions, cur_log_prob = agent.sample_actions(observations[ii][np.newaxis, np.newaxis],
                                                                 available_actions[ii][np.newaxis, np.newaxis],
                                                                 distribution=distribution)
            actions[ii] = cur_actions[0, 0]
            log_prob[ii] = cur_log_prob[0, 0]
            all_agents_alive[ii] = True
            agent_alive[ii] = [env.get_unit_by_id(i).health > 0 for i in range(n_agents)]
            rewards[ii], done, step_info = env.step(actions[ii])
            next_states[ii] = env.get_state()
            env_obs = np.stack(env.get_obs())
            if one_hot_to_observations:
                next_observations[ii] = np.concatenate((env_obs, np.eye(n_agents)), axis=1)
            else:
                next_observations[ii] = env_obs
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
        log_prob=log_prob,
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
