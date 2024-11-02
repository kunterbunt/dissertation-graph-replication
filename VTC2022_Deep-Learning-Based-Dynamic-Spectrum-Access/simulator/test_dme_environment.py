import pytest
from gym.utils.env_checker import check_env

import random
import numpy as np

from environment.dme_environment import DMEEnvironment

def assert_gym(env, actions, expected_observations, expected_rewards):
	observation = env.reset()
	np.testing.assert_array_equal(observation, expected_observations[0], "Error in observation after reset")

	for i, (action, expected_observation, expected_reward) in enumerate(zip(actions[1:], expected_observations[1:], expected_rewards[1:]), 1):
		observation, reward, _, _ = env.step(action)
		np.testing.assert_array_equal(observation, expected_observation, f"Error in observation after step {i} with action {action}")
		np.testing.assert_equal(reward, expected_reward, f"Error in reward after step {i} with action {action}")


num_freq_channels = 4
request_periodicity = 0.066
max_sim_time = 5*request_periodicity
config_baseline = {
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
	'observation_history_length': 0
}


def test_interface():
	config = config_baseline.copy()

	env = DMEEnvironment(config)
	check_env(env)


def test_interface_renderer():
	config = config_baseline.copy()

	env = DMEEnvironment(config)
	check_env(env, skip_render_check=False)


def test_invalid_action():
	config = config_baseline.copy()

	env = DMEEnvironment(config)
	env.reset()

	with pytest.raises(AssertionError, match='Invalid action:'):
		env.step(config['num_freq_channels'])
