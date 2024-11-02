import ray
import sys

import tensorflow as tf

import algorithm.baseline as baseline
import algorithm.rnn as rnn
from environment.dme_environment import DMEEnvironment
from environment.simple_environment import get_env
from plot.prediction import plot_roc, plot_timeseries
from serialize.json import save, load


def train_rnn(config_base, num_users):
	config = config_base.copy()
	config["num_users"] = num_users

	training_model = rnn.create_model(config, batch_size=config["batch_size"])
	for _ in range(config["epochs"]):
		rnn.train(training_model, config, batch_size=config["batch_size"])

	return training_model


@ray.remote
def evaluate_rnn(model, config_base, num_users):
	config = config_base.copy()
	config["num_users"] = num_users

	if config["evaluation_train"] == False:
		evaluation_model = rnn.create_model(config, batch_size=config["evaluation_batch_size"])
		evaluation_model.set_weights(model.get_weights())

		labels, predictions = rnn.predict(evaluation_model, config, steps=config["evaluation_steps"], warmup_steps=config["evaluation_warmup_steps"], batch_size=config["evaluation_batch_size"], train=False)
	else:
		labels = []
		predictions = []
		for _ in range(config["evaluation_batch_size"]):
			evaluation_model = rnn.create_model(config, batch_size=1)
			evaluation_model.set_weights(model.get_weights())

			run_labels, run_predictions = rnn.predict(evaluation_model, config, steps=config["evaluation_steps"], warmup_steps=config["evaluation_warmup_steps"], batch_size=1, train=True)
			labels.append(run_labels[0])
			predictions.append(run_predictions[0])

	return {"algorithm": "rnn", "num_users": config["num_users"], "integer": config["integer"]}, (labels, predictions)


@ray.remote
def evaluate_baseline(config, integer, num_users):
	labels = []
	predictions = []
	for _ in range(config["evaluation_batch_size"]):
		env = DMEEnvironment(get_env(num_users=num_users, num_obs=1, integer=integer))
		env.reset()

		baseline_tuples = baseline.train(env, steps=config["evaluation_warmup_steps"])
		run_labels, run_predictions = baseline.predict(env, baseline_tuples, current_step=config["evaluation_warmup_steps"], steps=config["evaluation_steps"])

		labels.append(run_labels)
		predictions.append(run_predictions)

	return {"algorithm": "baseline", "num_users": num_users, "integer": integer}, (labels, predictions)


if __name__ == "__main__":
	config = {
		# Environment
		"integer": False,

		# Neural Network
		"num_obs": 128,
		"pre_lstm_layers": 1,
		"pre_lstm_neurons_per_layer": 32,
		"post_lstm_layers": 1,
		"post_lstm_neurons_per_layer": 64,
		"lstm_activation": "tanh",
		"lstm_recurrent_activation": "sigmoid",
		"conv_layers": 1,
		"conv_kernels": 64,
		"conv_kernel_size": 32,
		"dense_layers": 2,
		"dense_neurons_per_layer": 32,

		# Training
		"batch_size": 128,
		"epochs": 25,
		"steps_per_epoch": 2500,
		"lr": 1e-3,

		# Evaluation
		"evaluation_batch_size": 32,
		"evaluation_steps": 2500,
		"evaluation_warmup_steps": 2500,
		"evaluation_train": False
	}

	if "nosimulate" not in sys.argv:
		ray.init(num_cpus=5)
		tf.config.threading.set_intra_op_parallelism_threads(1)
		tf.config.threading.set_inter_op_parallelism_threads(1)

		save("rnn_roc_config", config)

		# Train
		model = train_rnn(config, num_users=[1, 15])
		model.save("output/model")

		# Evaluation
		runs = []
		for num_users in range(1, 16):
			runs.append(evaluate_rnn.remote(model, config, num_users))
		for integer in [True, False]:
			runs.append(evaluate_baseline.remote(config, integer, num_users=[1, 15]))
		data = ray.get(runs)
		save("rnn_roc_data", data)

	if "noplot" not in sys.argv:
		data = load("rnn_roc_data")

		for metadata, (labels, predictions) in data:
			algorithm = metadata["algorithm"]
			environment = "integer" if metadata["integer"] else "uniform"
			num_users = metadata["num_users"]

			plot_timeseries(f"timeseries_{algorithm}_{environment}_{num_users}users", labels, predictions)

		plot_roc(f"roc", data)
