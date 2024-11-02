import gym
from gym.wrappers import FrameStack
import numpy as np

# Workaround for an incompatibility of Ray RLlib with the FrameStack wrapper
# The problem is reported in https://github.com/ray-project/ray/issues/22075
class FrameStackFixed(FrameStack):
	def astype(self, dtype):
		return self._array_(dtype=dtype)

class PartialObservability(gym.Wrapper):
	def step(self, action):
		obs, reward, done, truncated, info = self.env.step(action)
		obs_new = np.zeros(len(obs), dtype=np.int32)
		obs_new[action] = obs[action]
		return obs_new, reward, done, truncated, info

class FreeChannelReward(gym.Wrapper):
	def step(self, action):
		obs, _, done, truncated, info = self.env.step(action)
		reward_new = 1.0 if obs[action] == 1 else 0.0
		return obs, reward_new, done, truncated, info
