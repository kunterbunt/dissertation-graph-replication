import random

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import dqn

from environment.dme_environment import DMEEnvironment
from environment.single_free_channel_environment import SingleFreeChannelEnvironment

# Initialize Ray
ray.init(num_cpus=4, num_gpus=0)

# Register environments
register_env('dme', DMEEnvironment)
register_env('single_free_channel', SingleFreeChannelEnvironment)

# Environment configs
num_freq_channels = 4
request_periodicity = 0.066
max_sim_time = 10000*request_periodicity
dme_env_config = {
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

single_free_channel_env_config = {
	'num_freq_channels': 16,
	'partially_observable': True,
	'switching_probability': 0.9,
	'observation_history_length': 16
}

# Tune config
# Options and default values:
# https://docs.ray.io/en/releases-1.11.0/tune/api_docs/execution.html#tune-run
tune.run(
	'DQN',
	num_samples=1,
	stop={'agent_timesteps_total': 2e6},
	local_dir='results',
	verbose=2,
	# Trainer config
	# Options and default values:
	# (upper configs inherent options and overwrite values from lower configs)
	# https://github.com/ray-project/ray/blob/ray-1.11.0/rllib/agents/dqn/dqn.py#L38
	# https://github.com/ray-project/ray/blob/ray-1.11.0/rllib/agents/dqn/simple_q.py#L33
	# https://github.com/ray-project/ray/blob/ray-1.11.0/rllib/agents/trainer.py#L79
	config={
		# Environment
		'env': 'dme',
		'env_config': dme_env_config,
		#'env': 'single_free_channel',
		#'env_config': single_free_channel_env_config,
		'gamma': 0.9,

		# Training
		'num_workers': 0,
		'rollout_fragment_length': 1,
		'num_steps_sampled_before_learning_starts': 0,

		# Metric collection
		'horizon': 1000,
		'soft_horizon': True,
		'min_sample_timesteps_per_iteration': 1000,
		'metrics_num_episodes_for_smoothing': 1,

		# Algorithm
		'preprocessor_pref': None,
		'num_atoms': 1,
		'noisy': False,
		'dueling': tune.grid_search([False, True]),
		'double_q': tune.grid_search([False, True]),
		'target_network_update_freq': 500,
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
			'initial_epsilon': 0.1,
			'final_epsilon': 0.1,
			'epsilon_timesteps': 10000,
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
