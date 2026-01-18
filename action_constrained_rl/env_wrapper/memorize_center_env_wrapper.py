# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import gym
import numpy as np
from gym import spaces
import math
import time
from stable_baselines3.common.monitor import ResultsWriter


class MemorizeCenterEnvWrapper(gym.Wrapper):
    """
    env wrapper to calculate centers of action spaces and to concatenate it with observations
    """
    def __init__(self, constraint, env, n=1, dual_learning_rate=0.0):
        super().__init__(env)
        self.env = env
        self.constraint = constraint
        self.n=n
        self.dual_learning_rate = dual_learning_rate
        act_space = self.env.action_space
        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(low = np.concatenate((obs_space.low, act_space.low)),
                                                high  = np.concatenate((obs_space.high, act_space.high)),
                                                shape = (obs_space.shape[0] + act_space.shape[0],),
                                                dtype = obs_space.dtype)


    """
    def observation(self, obs):
        return np.concatenate((obs, self.constraint.get_center(obs)))
    """

    def observation(self, obs):
    # original behavior: append constraint center to observation
        try:
            center = self.constraint.get_center(obs)
        except Exception as e:
            # Fallback when Gurobi (or any center computation) fails
            if not getattr(self, "_warned_center", False):
                print("Warning: could not compute constraint center, using zeros instead:", e)
                self._warned_center = True
            # use a simple zero center with action dimension
            action_dim = self.env.action_space.shape[-1]
            center = np.zeros(action_dim, dtype=obs.dtype)

        return np.concatenate((obs, center))

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        self.prev_obs = self.env.reset(**kwargs)
        return self.observation(self.prev_obs)

    def step(self, action):
        state = self.prev_obs.squeeze()
        
        assert self.constraint.isConstraintSatisfied(state, action), f"constraint violated state={state} action{action} "
        
        next_state, reward, done, info = self.env.step(action)

        self.prev_obs = next_state

        return self.observation(next_state), reward, done, info
