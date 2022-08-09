from functools import partial
import multiprocessing
from typing import Tuple, List, Dict, Iterable as It
from time import sleep

import numpy as np

from smac.env import StarCraft2Env

# based on stable_baselines.common.vec_env.subproc_vec_env


class SubprocVecStarcraft:
    def __init__(self, num_envs: int, start_method='forkserver', one_hot_to_observations=True, **env_kwargs) -> None:
        self.waiting = False
        self.closed = False
        self.num_envs = num_envs

        env_fns = [partial(StarCraft2Env, **env_kwargs) for _ in range(num_envs)]

        ctx = multiprocessing.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()

            self.processes.append(process)
            work_remote.close()

        env_info = self.env_method('get_env_info')[0]

        if one_hot_to_observations:
            self.agents_obs_fun = lambda agents_obs: np.concatenate((agents_obs, np.eye(agents_obs.shape[0])), axis=1)
            self.obs_dim = env_info['obs_shape'] + env_info['n_agents']
        else:
            self.agents_obs_fun = lambda x: x
            self.obs_dim = env_info['obs_shape']

        self.state_dim = env_info['state_shape']
        self.n_actions = env_info['n_actions']
        self.n_agents = env_info['n_agents']
        self.time_limit = env_info['episode_limit']
        self.envs_done = None

    def get_info(self):
        names = ['num_envs', 'obs_dim', 'state_dim', 'n_actions', 'n_agents', 'time_limit']
        return {name: getattr(self, name) for name in names}

    def step_async(self, envs_agents_action: It[It[int]]) -> None:
        for remote, agents_action in zip(self._get_not_done_remotes(), envs_agents_action):
            remote.send(('step', agents_action))
        self.waiting = True

    def get_not_done_indices(self):
        return [i for i, d in enumerate(self.envs_done) if not d]

    def _update_dones(self, envs_done: It[bool]) -> None:
        self.envs_done[self.get_not_done_indices()] = envs_done

    def _get_not_done_remotes(self):
        return self._get_target_remotes(self.get_not_done_indices())

    def get_states(self, indices=None) -> It[It[float]]:
        envs_state = process_array(self.env_method('get_state', indices=indices))
        return envs_state

    def get_observations(self, indices=None) -> It[It[It[float]]]:
        raw_envs_agents_observation = process_array(self.env_method('get_obs', indices=indices))
        envs_agents_observation = process_array(map(self.agents_obs_fun, raw_envs_agents_observation))
        return envs_agents_observation

    def get_available_actions(self, indices=None) -> It[It[It[bool]]]:
        envs_agents_available_actions = process_array(self.env_method('get_avail_actions', indices=indices))
        return envs_agents_available_actions

    def get_agents_alive(self, indices=None) -> It[It[bool]]:
        envs_agents_unit = zip(*[self.env_method('get_unit_by_id', indices=indices, a_id=i) for i in range(self.n_agents)])
        envs_agents_alive = np.array([[unit.health > 0 for unit in agents_unit] for agents_unit in envs_agents_unit])
        return envs_agents_alive

    def get_data(self, indices=None) -> Tuple[It[It[float]], It[It[It[float]]], It[It[It[int]]],  It[It[bool]]]:
        envs_state = self.get_states(indices=indices)
        envs_agents_observation = self.get_observations(indices=indices)
        envs_agents_available_actions = self.get_available_actions(indices=indices)
        envs_agents_alive = self.get_agents_alive(indices=indices)
        return envs_state, envs_agents_observation, envs_agents_available_actions, envs_agents_alive

    def step_wait(self) -> Tuple[It[float], It[bool], It[Dict]]:
        results = [remote.recv() for remote in self._get_not_done_remotes()]
        self.waiting = False
        envs_reward, envs_done, envs_info = zip(*results)
        envs_reward = process_array(envs_reward)
        envs_done = process_array(envs_done)
        self._update_dones(envs_done)
        return envs_reward, envs_done, envs_info

    def step(self, envs_agents_action: It[It[int]]):
        """Step for envs with ongoing trajectories."""
        self.step_async(envs_agents_action)
        return self.step_wait()

    def reset(self) -> Tuple[It[It[float]], It[It[It[float]]], It[It[It[int]]],  It[It[bool]]]:
        for remote in self.remotes:
            remote.send(('reset', None))
        self.envs_done = np.zeros(self.num_envs, dtype=bool)
        for remote in self.remotes:
            remote.recv()
        return self.get_data()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices=None) -> List:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.
        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                reward, done, info = env.step(data)
                remote.send((reward, done, info))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'render':
                remote.send(env.render(data))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break


def process_array(arg):
    return np.array(list(arg))
