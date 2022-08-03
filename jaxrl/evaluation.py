from typing import Tuple

import numpy as np

import jax.numpy as jnp

from jaxrl.agents import MetaPGLearner
from jaxrl.datasets.dataset import PaddedTrajectoryData, concatenate, init_data
from jaxrl.vec_env import SubprocVecStarcraft


def collect_trajectories(envs: SubprocVecStarcraft, agent: MetaPGLearner,
                         num_trajectories_per_env,
                         save_replay: bool = False, replay_prefix: str = None,
                         distribution='log_prob') -> Tuple[PaddedTrajectoryData, dict]:
    datas = []
    all_end_info = {}
    total_steps = 0
    for _ in range(num_trajectories_per_env):
        try:
            jax_data, end_info, num_steps = collect_one_trajectory_per_env(envs, agent,
                                                                           save_replay=save_replay,
                                                                           replay_prefix=replay_prefix,
                                                                           distribution=distribution)
        except Exception:
            envs.close()
            raise

        datas.append(jax_data)
        for k, v in end_info.items():
            all_end_info[k] = all_end_info.get(k, []) + v
        total_steps += num_steps
    combined_data = concatenate(datas)

    info = {'returns': (combined_data.rewards * combined_data.any_agents_alive).sum(axis=1).mean(),
            'iter_steps': total_steps}
    for k, arr in all_end_info.items():
        info[k] = np.mean(arr)
    return combined_data, info


def collect_one_trajectory_per_env(envs: SubprocVecStarcraft, agent: MetaPGLearner,
                                   save_replay: bool = False, replay_prefix: str = None,
                                   distribution='log_prob') -> Tuple[PaddedTrajectoryData, dict, int]:
    if save_replay:
        envs.env_method('save_replay')
    envs.set_attr('replay_prefix', replay_prefix)

    data = init_data(n_trajectories=envs.num_envs,
                     time_limit=envs.time_limit,
                     n_agents=envs.n_agents,
                     state_dim=envs.state_dim,
                     obs_dim=envs.obs_dim,
                     n_actions=envs.n_actions)

    total_steps = 0
    end_info = {"dead_allies": [],
                "dead_enemies": [],
                "battle_won": []}

    carry = agent.initialize_carry(envs.num_envs)
    step = 0
    envs.reset()
    while not_done_indices := envs.get_not_done_indices():
        ii_unsqueezed = (not_done_indices, slice(step, step + 1), ...)
        ii = (not_done_indices, step, ...)

        data.any_agents_alive[ii] = True
        data.states[ii], data.observations[ii], data.available_actions[ii], data.agents_alive[ii] = envs.get_data(not_done_indices)

        policy_kwargs = {'observations': data.observations[ii_unsqueezed],
                         'available_actions': data.available_actions[ii_unsqueezed],
                         'distribution': distribution}
        if agent.use_recurrent_policy:
            carry, data.actions[ii_unsqueezed], data.log_prob[ii_unsqueezed] = agent.sample_actions(**policy_kwargs, carry=carry)
        else:
            data.actions[ii_unsqueezed], data.log_prob[ii_unsqueezed] = agent.sample_actions(**policy_kwargs)

        total_steps += len(not_done_indices)
        data.rewards[ii], envs_done, envs_step_info = envs.step(data.actions[ii])
        carry = carry[np.logical_not(envs_done)]
        data.next_states[ii] = envs.get_states(not_done_indices)

        for i, d, step_info in zip(not_done_indices, envs_done, envs_step_info):
            if d:
                episode_limit = step_info.get("episode_limit", False)
                data.is_ended[i] = not episode_limit
                for k in end_info:
                    # sometimes step() return empty info
                    if k in step_info:
                        end_info[k].append(step_info[k])
        step += 1
    jax_data = PaddedTrajectoryData(*map(jnp.array, data))
    return jax_data, end_info, total_steps
