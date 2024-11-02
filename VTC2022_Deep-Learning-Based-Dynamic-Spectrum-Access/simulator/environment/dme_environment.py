import simpy
import gym
from gym.spaces import Box, Discrete
import numpy as np
import random

from environment.wrappers import FrameStackFixed, PartialObservability, FreeChannelReward
from environment.entities import Aircraft, GroundStation
from environment.radio_medium import RadioMedium
from environment.renderer import Renderer

class DMEAirborneStation(Aircraft):
	def __init__(self, env, radio_medium, pos_x, pos_y, rot, vel, request_freq_channel_index, request_periodicity, start_offset=0):
		super().__init__(pos_x, pos_y, rot, vel)
		"""
		request_periodicity: Time in-between request transmissions.
		start_offset: Time the user spends idle at the start of the simulation.
		"""
		self.env = env
		self.simpy_env = env.simpy_env
		self.radio_medium = radio_medium
		self.request_freq_channel_index = request_freq_channel_index
		self.request_periodicity = request_periodicity
		self.start_offset = start_offset		
		self.pulse_pair_tx_time = 0.0000155 # transmission time, see https://ieeexplore.ieee.org/abstract/document/6218404

		self.simpy_env.process(self.run())

	def run(self):
		if self.request_freq_channel_index == -1:
			return

		# initially wait some time before starting transmission
		yield self.simpy_env.timeout(self.start_offset)
		while True:
			# wait until next request transmission
			yield self.simpy_env.timeout(self.request_periodicity)
			# send request			
			self.radio_medium.transmit(self, self.request_freq_channel_index, ("request", self.pulse_pair_tx_time))


class DMEGroundStation(GroundStation):
	def __init__(self, env, radio_medium, pos_x, pos_y, request_freq_channel_index, response_freq_channel_index, response_delay):
		super().__init__(pos_x, pos_y)
		self.env = env
		self.simpy_env = env.simpy_env
		self.radio_medium = radio_medium
		self.request_freq_channel_index = request_freq_channel_index
		self.response_freq_channel_index = response_freq_channel_index
		self.response_delay = response_delay
		self.pulse_pair_tx_time = 0.0000155 # transmission time, see https://ieeexplore.ieee.org/abstract/document/6218404

		self.simpy_env.process(self.run())

	def run(self):
		if self.request_freq_channel_index == -1:
			return

		self.radio_medium.add_receiver(self, self.request_freq_channel_index)
		while True:
			# wait for a request transmission
			signal = yield self.radio_medium.receive(self, self.request_freq_channel_index)
			# transmit the reply								
			yield self.simpy_env.timeout(signal[1] + self.response_delay)
			if self.response_freq_channel_index != -1:
				self.radio_medium.transmit(self, self.response_freq_channel_index, ("response", self.pulse_pair_tx_time))


class DMETaggedAircraft(DMEAirborneStation):
	def __init__(self, env, radio_medium, pos_x, pos_y, rot, vel, num_freq_channels, request_freq_channel_index, request_periodicity, start_offset=0):
		super().__init__(env, radio_medium, pos_x, pos_y, rot, vel, request_freq_channel_index, request_periodicity, start_offset)
		self.num_freq_channels = num_freq_channels

		self.transmissions = np.zeros(self.num_freq_channels)
		for freq_channel in range(self.num_freq_channels):
			self.simpy_env.process(self.observe(freq_channel))

	def handle_transmission(self, freq_channel, transmission_delay):
		self.env.freq_channels[freq_channel] = DMEEnvironmentBase.BUSY
		self.transmissions[freq_channel] += 1
		yield self.simpy_env.timeout(transmission_delay)
		self.transmissions[freq_channel] -= 1

	def observe(self, freq_channel):
		self.radio_medium.add_receiver(self, freq_channel)
		while True:
			# wait for a transmission
			signal = yield self.radio_medium.receive(self, freq_channel)
			# handle the transmission
			self.simpy_env.process(self.handle_transmission(freq_channel, signal[1]))


class DMEEnvironmentBase(gym.Env):

	BUSY = -1  # symbolic value to indicate a busy channel
	IDLE = 1  # symbolic value to indicate an idle channel
	NOT_SENSED = 0  # symbolic value to indicate an unsensed channel in case of partial observability

	def __init__(self, config):
		# parse from config
		self.airborne_users_config = config['airborne_users']
		self.ground_stations_config = config['ground_stations']
		self.num_freq_channels = config['num_freq_channels']
		self.ldacs_time_slot_duration = config['ldacs_time_slot_duration']
		self.max_simulation_time = config['max_simulation_time']
		self.debug = config['debug'] if 'debug' in config else False
		# set action and observation spaces
		self.action_space = Discrete(self.num_freq_channels)
		self.observation_space = Box(low=DMEEnvironmentBase.BUSY, high=DMEEnvironmentBase.IDLE, shape=(self.num_freq_channels, ), dtype=np.int8)
		# set up global variables
		self.simpy_env = None
		self.radio_medium = None
		self.freq_channels = np.full(self.num_freq_channels, DMEEnvironmentBase.IDLE, dtype=np.int8)
		self.ground_stations = []
		self.airborne_users = []

	def reset(self, seed=0, options=None):
		self.simpy_env = simpy.Environment()		
		self.radio_medium = RadioMedium(self.simpy_env, 1.0, 0.1)
		for ground_station in self.ground_stations_config:
			self.ground_stations.append(DMEGroundStation(self, self.radio_medium, ground_station['pos_x'], ground_station['pos_y'], ground_station['request_freq_channel_index'], ground_station['response_freq_channel_index'], ground_station['response_delay']))
		for i, airborne_user in enumerate(self.airborne_users_config):
			if i == 0:
				self.airborne_users.append(DMETaggedAircraft(self, self.radio_medium, airborne_user['pos_x'], airborne_user['pos_y'], airborne_user['rot'], airborne_user['vel'], self.num_freq_channels, airborne_user['request_freq_channel_index'], airborne_user['request_periodicity'], airborne_user['start_offset']))
			else:
				self.airborne_users.append(DMEAirborneStation(self, self.radio_medium, airborne_user['pos_x'], airborne_user['pos_y'], airborne_user['rot'], airborne_user['vel'], airborne_user['request_freq_channel_index'], airborne_user['request_periodicity'], airborne_user['start_offset']))
		return np.zeros(self.num_freq_channels, dtype=np.int8), {}
		
	def step(self, action):
		assert self.simpy_env is not None, 'Cannot call env.step() before calling reset()'
		assert self.action_space.contains(action), f'Invalid action: {action!r} ({type(action)})'
		if self.debug:
			print('LDACS slot starts at %fs' % self.simpy_env.now)
		# continue simulation for another LDACS time slot duration
		self.simpy_env.run(self.simpy_env.now + self.ldacs_time_slot_duration)
		is_done = self.simpy_env.now >= self.max_simulation_time
		# observe the frequency channels
		observation = np.copy(self.freq_channels)
		# reset frequency channels
		self.freq_channels = np.full(self.num_freq_channels, DMEEnvironmentBase.IDLE, dtype=np.int8)
		for frequency_channel, tansmissions in enumerate(self.airborne_users[0].transmissions): # unless there's an ongoing transmission
			if tansmissions > 0:
				self.freq_channels[frequency_channel] = DMEEnvironmentBase.BUSY
		if self.debug:
			print('LDACS slot ends at %fs' % self.simpy_env.now, end=' o=')
			print(observation)
		for airborne_user in self.airborne_users:
			airborne_user.update_position(time=self.simpy_env.now)
		info = {
			"aircraft": [{
				"pos_x": aircraft.get_position()[0],
				"pos_y": aircraft.get_position()[1],
				"rot": aircraft.get_rotation(),
			} for aircraft in self.airborne_users],
			"ground_stations": [{
				"pos_x": ground_station.get_position()[0],
				"pos_y": ground_station.get_position()[1]
			} for ground_station in self.ground_stations]
		}
		return observation, 0.0, is_done, False, info

	def render(self, mode='human'):
		pass

	def close(self):
		pass
		


def DMEEnvironment(env_config):
	env = DMEEnvironmentBase(env_config)
	env = FreeChannelReward(env)
	if env_config['partially_observable']:
		env = PartialObservability(env)
	env = Renderer(env)
	if env_config['observation_history_length'] > 0:
		env = FrameStackFixed(env, env_config['observation_history_length'])
	return env


if __name__ == "__main__":	
	num_freq_channels = 4
	request_periodicity = 0.066
	max_sim_time = 500*request_periodicity
	config = {
		'airborne_users': [
			{
				'pos_x': random.uniform(0.45, 0.55),
				'pos_y': random.uniform(0.45, 0.55),
				'rot': random.uniform(170, 190),
				'vel': 0.01,
				'request_freq_channel_index': i,
				'request_periodicity': request_periodicity,
				'start_offset': 0
			} for i in range(0, num_freq_channels, 2)
		],
		'ground_stations': [
			{
				'pos_x': random.uniform(0.2, 0.55),
				'pos_y': random.uniform(0.45, 0.55),
				'request_freq_channel_index': i,
				'response_freq_channel_index': i+1,
				'response_delay': 0.000050
			} for i in range(0, num_freq_channels, 2)
		],
		'num_freq_channels': num_freq_channels,
		'partially_observable': False,
		'ldacs_time_slot_duration': 0.024,
		'max_simulation_time': max_sim_time,
		'observation_history_length': 0,
		'debug': True
	}
	env = DMEEnvironment(config)
	is_done = False
	observation = env.reset()
	print('_', observation.__array__(np.int8), '_')
	while True:
		env.render()
		action = env.action_space.sample()
		observation, reward, is_done, info = env.step(action)
		if is_done:
			break
		print(action, observation.__array__(np.int8), reward)
	env.close()
