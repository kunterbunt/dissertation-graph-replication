import pytest
import numpy as np
from environment.single_free_channel_environment import SingleFreeChannelEnvironment
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import dqn
from ray.tune.stopper import TrialPlateauStopper

def test_reward_first_channel_selected():	
	config = {
		'num_freq_channels': 2,
		'partially_observable': False,
		'switching_probability': 1,
		'observation_history_length': 0
	}
	env = SingleFreeChannelEnvironment(config)
	env.reset()
	channel_selection = 0
	observation, reward, is_done, info = env.step(action=channel_selection)	
	assert reward == 1

def test_reward_second_channel_selected():	
	config = {
		'num_freq_channels': 8,
		'partially_observable': False,
		'switching_probability': 1,
		'observation_history_length': 0
	}
	env = SingleFreeChannelEnvironment(config)
	env.reset()
	channel_selection = 1
	observation, reward, is_done, info = env.step(action=channel_selection)	
	assert reward == 0

def test_train_agent():
	config = {
		'num_freq_channels': 8,
		'partially_observable': False,
		'switching_probability': 1,
		'observation_history_length': 0
	}	
	ray.init(num_cpus=1, num_gpus=0)
	register_env('single_free_channel', SingleFreeChannelEnvironment)
	
	# stopper = TrialPlateauStopper(metric='episode_reward_mean', metric_threshold=85, mode='max')
	tune.run(
		'DQN',
		num_samples=1,
		stop={'agent_timesteps_total': 5000, 'episode_reward_mean': 90},
		local_dir='results',
		verbose=2,		
		config={
			# Environment			
			'env': 'single_free_channel',
			'env_config': config,
			'gamma': 0.9,

			# Training
			'num_workers': 0,
			'rollout_fragment_length': 1,
			'num_steps_sampled_before_learning_starts': 0,

			# Metric collection
			'horizon': 100,  # virtual episode length since our environment has an infinite episode
			'soft_horizon': True,
			'min_sample_timesteps_per_iteration': 100,
			'metrics_num_episodes_for_smoothing': 1,

			# Algorithm
			'preprocessor_pref': None,
			'num_atoms': 1,
			'noisy': False,
			'dueling': True,#tune.grid_search([False, True]),
			'double_q': True,#tune.grid_search([False, True]),
			'target_network_update_freq': 250,
			'n_step': 1,

			# Model
			# Options and default values:
			# https://github.com/ray-project/ray/blob/ray-1.11.0/rllib/models/catalog.py#L37
			'model': {
				'fcnet_hiddens': [200, 200],
				'fcnet_activation': 'relu',
			},

			# Experience replay
			#'prioritized_replay': False,
			'replay_buffer_config': {
				'capacity': 1000000
			},
			'train_batch_size': 32,

			# Exploration
			'exploration_config': {
				'type': 'EpsilonGreedy',
				'initial_epsilon': 0.5,
				'final_epsilon': 0.05,
				'epsilon_timesteps': 500,
			},

			# Neural network training
			'lr': 1e-4,
			'adam_epsilon': 1e-8,
			'grad_clip': None,

			# ML framework
			'framework': 'tf2',
			'eager_tracing': True,

			# GPU
			'num_gpus': 0,

			# Render
			'render_env': False,
			#'record_env': True
		}
	)