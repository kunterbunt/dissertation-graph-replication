import gym
from gym.spaces import Box, Discrete
import numpy as np
import random

from environment.wrappers import FrameStackFixed, PartialObservability, FreeChannelReward
from environment.entities import Aircraft, GroundStation
from environment.renderer import Renderer

class SingleFreeChannelEnvironmentBase(gym.Env):
	BUSY = -1 # symbolic value to indicate a busy channel
	IDLE = 1 # symbolic value to indicate an idle channel

	def __init__(self, config):
		# parse from config
		self.num_freq_channels = config['num_freq_channels']
		self.switching_probability = config['switching_probability']
		# set action and observation spaces
		self.action_space = Discrete(self.num_freq_channels)
		self.observation_space = Box(low=self.BUSY, high=self.IDLE, shape=(self.num_freq_channels, ), dtype=np.int8)
		self.time_step = None
		self.free_channel = None
		self.aircraft = None
		self.ground_stations = None

	def reset(self, seed=0, options=None):
		self.time_step = -1.0
		self.free_channel = -1
		self.aircraft = [Aircraft(random.random(), random.random(), random.uniform(0, 360), random.uniform(0.0009, 0.0011)) for _ in range(50)]
		self.ground_stations = [
			GroundStation(0.8, 0.15),
			GroundStation(0.7, 0.4),
			GroundStation(0.66, 0.61),
			GroundStation(0.65, 0.85),
			GroundStation(0.88, 0.65),
			GroundStation(0.2, 0.48),
		]
		return np.zeros(self.num_freq_channels, dtype=np.int8)

	def step(self, action):
		assert self.time_step is not None and self.free_channel is not None and self.aircraft is not None and self.ground_stations is not None, 'Cannot call env.step() before calling reset()'
		assert self.action_space.contains(action), f'Invalid action: {action!r} ({type(action)})'
		# increase time step
		self.time_step += 1.0
		# switch free channel based on switching probability
		if random.random() < self.switching_probability or self.free_channel == -1:
			self.free_channel = (self.free_channel + 1) % self.num_freq_channels
		# observe the frequency channels
		observation = np.full(self.num_freq_channels, self.BUSY, dtype=np.int8)
		observation[self.free_channel] = self.IDLE
		# create info dictionary
		for aircraft in self.aircraft:
			pos_x, pos_y = aircraft.get_position(time=self.time_step)
			pos_x = pos_x % 1.0
			pos_y = pos_y % 1.0
			aircraft.set_position(pos_x, pos_y, time=self.time_step)
		info = {
			"aircraft": [{
				"pos_x": aircraft.get_position()[0],
				"pos_y": aircraft.get_position()[1],
				"rot": aircraft.get_rotation(),
			} for aircraft in self.aircraft],
			"ground_stations": [{
				"pos_x": ground_station.get_position()[0],
				"pos_y": ground_station.get_position()[1]
			} for ground_station in self.ground_stations]
		}
		return observation, 0.0, False, info

	def render(self, mode='human'):
		pass

	def close(self):
		pass


def SingleFreeChannelEnvironment(env_config):
	env = SingleFreeChannelEnvironmentBase(env_config)
	env = FreeChannelReward(env)
	if env_config['partially_observable']:
		env = PartialObservability(env)
	env = Renderer(env)
	if env_config['observation_history_length'] > 0:
		env = FrameStackFixed(env, env_config['observation_history_length'])
	return env


if __name__ == '__main__':
	config = {
		'num_freq_channels': 16,
		'partially_observable': False,
		'switching_probability': 1,
		'observation_history_length': 0
	}
	env = SingleFreeChannelEnvironment(config)
	is_done = False
	observation = env.reset()
	print('_', observation.__array__(np.int8), '_')
	for i in range(1000):
		env.render()
		action = env.action_space.sample()
		observation, reward, is_done, info = env.step(action)
		if is_done:
			break
		print(action, observation.__array__(np.int8), reward)
	env.close()
