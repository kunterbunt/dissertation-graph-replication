import pytest
from gym.utils.env_checker import check_env

import numpy as np

from environment.single_free_channel_environment import SingleFreeChannelEnvironment

def assert_gym(env, actions, expected_observations, expected_rewards):
	observation = env.reset()
	np.testing.assert_array_equal(observation, expected_observations[0], "Error in observation after reset")

	for i, (action, expected_observation, expected_reward) in enumerate(zip(actions[1:], expected_observations[1:], expected_rewards[1:]), 1):
		observation, reward, _, _ = env.step(action)
		np.testing.assert_array_equal(observation, expected_observation, f"Error in observation after step {i} with action {action}")
		np.testing.assert_equal(reward, expected_reward, f"Error in reward after step {i} with action {action}")


config_baseline = {
	'num_freq_channels': 4,
	'partially_observable': False,
	'switching_probability': 1,
	'observation_history_length': 0
}


def test_interface():
	config = config_baseline.copy()

	env = SingleFreeChannelEnvironment(config)
	check_env(env)


def test_interface_renderer():
	config = config_baseline.copy()

	env = SingleFreeChannelEnvironment(config)
	check_env(env, skip_render_check=False)


def test_invalid_action():
	config = config_baseline.copy()

	env = SingleFreeChannelEnvironment(config)
	env.reset()
	with pytest.raises(AssertionError, match='Invalid action:'):
		env.step(config['num_freq_channels'])


def test_baseline():
	config = config_baseline.copy()

	actions = [None, 0, 0, 0, 0, 0]
	expected_observations = [
		[  0,  0,  0,  0],
		[  1, -1, -1, -1],
		[ -1,  1, -1, -1],
		[ -1, -1,  1, -1],
		[ -1, -1, -1,  1],
		[  1, -1, -1, -1]
	]
	expected_rewards = [None, 1.0, 0.0, 0.0, 0.0, 1.0]

	env = SingleFreeChannelEnvironment(config)
	assert_gym(env, actions, expected_observations, expected_rewards)


def test_three_freq_channels():
	config = config_baseline.copy()
	config['num_freq_channels'] = 3

	actions = [None, 0, 0, 0, 0, 0]
	expected_observations = [
		[  0,  0,  0],
		[  1, -1, -1],
		[ -1,  1, -1],
		[ -1, -1,  1],
		[  1, -1, -1],
		[ -1,  1, -1]
	]
	expected_rewards = [None, 1.0, 0.0, 0.0, 1.0, 0.0]

	env = SingleFreeChannelEnvironment(config)
	assert_gym(env, actions, expected_observations, expected_rewards)


def test_switching_probability_reset():
	config = config_baseline.copy()
	config['switching_probability'] = 0.5

	actions = [None, 0]
	expected_observations = [
		[  0,  0,  0,  0],
		[  1, -1, -1, -1],
	]
	expected_rewards = [None, 1.0]

	for _ in range(100):
		env = SingleFreeChannelEnvironment(config)
		assert_gym(env, actions, expected_observations, expected_rewards)


def test_partially_observable():
	config = config_baseline.copy()
	config['partially_observable'] = True

	actions = [None, 0, 0, 1, 3, 2]
	expected_observations = [
		[  0,  0,  0,  0],
		[  1,  0,  0,  0],
		[ -1,  0,  0,  0],
		[  0, -1,  0,  0],
		[  0,  0,  0,  1],
		[  0,  0, -1,  0]
	]
	expected_rewards = [None, 1.0, 0.0, 0.0, 1.0, 0.0]

	env = SingleFreeChannelEnvironment(config)
	assert_gym(env, actions, expected_observations, expected_rewards)


def test_observation_history_length_one():
	config = config_baseline.copy()
	config['observation_history_length'] = 1

	actions = [None, 0, 0, 0, 0, 0]
	expected_observations = [
		[[  0,  0,  0,  0]],
		[[  1, -1, -1, -1]],
		[[ -1,  1, -1, -1]],
		[[ -1, -1,  1, -1]],
		[[ -1, -1, -1,  1]],
		[[  1, -1, -1, -1]]
	]
	expected_rewards = [None, 1.0, 0.0, 0.0, 0.0, 1.0]

	env = SingleFreeChannelEnvironment(config)
	assert_gym(env, actions, expected_observations, expected_rewards)


def test_observation_history_length_two():
	config = config_baseline.copy()
	config['observation_history_length'] = 2

	actions = [None, 0, 0, 0, 0, 0]
	expected_observations = [
		[[  0,  0,  0,  0], [  0,  0,  0,  0]],
		[[  0,  0,  0,  0], [  1, -1, -1, -1]],
		[[  1, -1, -1, -1], [ -1,  1, -1, -1]],
		[[ -1,  1, -1, -1], [ -1, -1,  1, -1]],
		[[ -1, -1,  1, -1], [ -1, -1, -1,  1]],
		[[ -1, -1, -1,  1], [  1, -1, -1, -1]]
	]
	expected_rewards = [None, 1.0, 0.0, 0.0, 0.0, 1.0]

	env = SingleFreeChannelEnvironment(config)
	assert_gym(env, actions, expected_observations, expected_rewards)
