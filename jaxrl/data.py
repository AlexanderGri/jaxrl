import collections
from typing import List

import numpy as np

import jax.numpy as jnp


PaddedTrajectoryData = collections.namedtuple(
    'PaddedTrajectoryData',
    ['is_ended', 'states', 'next_states', 'rewards', 'any_agents_alive', 'observations',
     'actions', 'log_prob', 'available_actions', 'agents_alive',])


def init_data(n_trajectories, time_limit, n_agents, state_dim, obs_dim, n_actions) -> collections.namedtuple:
    return PaddedTrajectoryData(
        # per trajectory
        is_ended=np.zeros((n_trajectories,), dtype=bool),
        # per trajectory per step
        states=np.zeros((n_trajectories, time_limit, state_dim)),
        next_states=np.zeros((n_trajectories, time_limit, state_dim)),
        rewards=np.zeros((n_trajectories, time_limit)),
        any_agents_alive=np.zeros((n_trajectories, time_limit), dtype=bool),
        # per trajectory per step per agent
        observations=np.zeros((n_trajectories, time_limit, n_agents, obs_dim)),
        actions=np.zeros((n_trajectories, time_limit, n_agents), dtype=int),
        log_prob=np.zeros((n_trajectories, time_limit, n_agents)),
        available_actions=np.zeros((n_trajectories, time_limit, n_agents, n_actions), dtype=bool),
        agents_alive=np.zeros((n_trajectories, time_limit, n_agents), dtype=bool),
    )


def concatenate(datas: List[PaddedTrajectoryData]) -> PaddedTrajectoryData:
    keys = datas[0]._asdict().keys()
    combined_data = PaddedTrajectoryData(**{k: jnp.concatenate([getattr(data, k) for data in datas]) for k in keys})
    return combined_data
