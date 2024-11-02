from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
import tensorflow as tf

import algorithm.rnn as rnn


def trial(config):
	tf.config.threading.set_intra_op_parallelism_threads(config["threads"])
	tf.config.threading.set_inter_op_parallelism_threads(config["threads"])

	training_model = rnn.create_model(config, batch_size=config["batch_size"])
	evaluation_model = rnn.create_model(config, batch_size=32)

	for _ in range(config["epochs"]):
		#Source: https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
		train_history = rnn.train(training_model, config, batch_size=config["batch_size"])

		evaluation_model.set_weights(training_model.get_weights())

		evaluation_loss, evaluation_acc, evaluation_precision, evaluation_recall, evaluation_auc, evaluation_f1 = rnn.evaluate(evaluation_model, config, steps=1000, batch_size=32)

		tune.report(
			training_loss=train_history.history['loss'][-1],
			evaluation_loss=evaluation_loss,
			evaluation_acc=evaluation_acc,
			evaluation_precision=evaluation_precision,
			evaluation_recall=evaluation_recall,
			evaluation_auc=evaluation_auc,
			evaluation_f1=evaluation_f1[0]
		)


if __name__ == "__main__":
	config = {
		# Environment
		"num_users": [1, 15],
		"integer": False,

		# Neural Network
		"num_obs": 128, #tune.randint(32, 128),
		"pre_lstm_layers": tune.randint(0, 3),
		"pre_lstm_neurons_per_layer": tune.choice([32, 64]),
		"post_lstm_layers": tune.randint(0, 3),
		"post_lstm_neurons_per_layer": tune.choice([32, 64, 128]),
		"lstm_activation": "tanh", #tune.choice(["relu", "tanh", "sigmoid"]),
		"lstm_recurrent_activation": "sigmoid", #tune.choice(["relu", "tanh", "sigmoid"]),
		#"conv_layers": 0,
		#"conv_kernels": tune.choice([32, 64, 128]),
		#"conv_kernel_size": 32,
		"dense_layers": tune.randint(0, 3),
		"dense_neurons_per_layer": tune.choice([32, 64, 128]),

		# Training
		"threads": 1,
		"batch_size": 128, #tune.choice([32, 64, 128]),
		"epochs": 10,
		"steps_per_epoch": 2500,
		"lr": 1e-3 #tune.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
	}

	bayesopt = HyperOptSearch()
	algo = ConcurrencyLimiter(bayesopt, max_concurrent=8)

	analysis = tune.run(
		trial,
		num_samples=1000,
		resources_per_trial={
			"cpu": config["threads"]
		},
		search_alg=algo,
		metric="evaluation_auc",
		mode="max",
		local_dir="results_hyperopt",
		config=config
	)

	print("Best hyperparameters: ", analysis.best_config)
