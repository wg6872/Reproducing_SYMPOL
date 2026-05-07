# env_wrappers.py
# Author(s): Evan Soper
# Environment wrappers for benchmarking PPO with SDT
# Taken from: https://github.com/s-marton/SYMPOL/blob/master/utils.py#L242

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper

class FlatCurrentReducedWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.select_indices = [0, 1, 2, 8, 9]

        h, w, _ = env.observation_space.spaces["image"].shape
        obs_dim = h * w * len(self.select_indices)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        image = obs["image"].astype(np.float32)
        x = image[:, :, self.select_indices]
        x = x.reshape(-1)
        x = x * 2.0 - 1.0
        return x


class NormalizeWrapperLunarLander(gym.ObservationWrapper):
    def observation(self, obs):
        obs = obs.astype(np.float32).copy()

        obs[0] /= 1.5
        obs[1] /= 1.5
        obs[2] /= 5.0
        obs[3] /= 5.0
        obs[4] /= 3.14
        obs[5] /= 5.0
        obs[6] = (obs[6] - 1.0) / 0.5
        obs[7] = (obs[7] - 1.0) / 0.5

        return obs
