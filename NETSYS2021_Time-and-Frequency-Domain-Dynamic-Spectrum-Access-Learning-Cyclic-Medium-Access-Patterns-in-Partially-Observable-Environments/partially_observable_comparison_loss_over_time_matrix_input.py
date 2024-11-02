#   Code to replicate results of the publication 
#   'Time- and Frequency-Domain Dynamic Spectrum Access: Learning Cyclic Medium Access Patterns in Partially Observable Environments' 
#   published at the Conference on Networked Systems (NetSys) 2021, Lübeck, Germany.
#   https://github.com/ComNetsHH/netsys2021-malene-code-release
#
#     Copyright (C) 2021  Institute of Communication Networks, 
#                         Hamburg University of Technology,
#                         https://www.tuhh.de/comnets
#               (C) 2021  Sebastian Lindner, 
#                         sebastian.lindner@tuhh.de
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
import settings
import matplotlib.pyplot as plt
import progressbar
import sys
import tensorflow as tf
import numpy as np
from _collections import deque

from env import *
from neural_network import LSTMRNN
import os
import argparse
from confidence_intervals import columnwise_batch_means, calculate_confidence_interval
from running_mean import running_mean

dict_label_loss = 'loss_mat'
dict_label_acc = 'acc_mat'
dict_label_ci_loss = 'loss_ci_loss'
dict_label_ci_acc = 'loss_ci_acc'
LINE_WIDTH = 1
RUNNING_MEAN_AGGREGATE = 500

def plot(filename, n_sample_length):
	fig = plt.figure()
	settings.init()
	settings.set_params()

	colors=['xkcd:teal', 'xkcd:coral']
	with open("_data/" + filename + "__matrix.json") as json_file:
		json_data = json.load(json_file)

		# Accuracy
		yAcc, yAcc_m, yAcc_p = json_data[dict_label_ci_acc]
		yAcc = running_mean(yAcc, RUNNING_MEAN_AGGREGATE)
		yAcc_m = running_mean(yAcc_m, RUNNING_MEAN_AGGREGATE)
		yAcc_p = running_mean(yAcc_p, RUNNING_MEAN_AGGREGATE)
		x = np.array(range(1, len(yAcc) + 1))
		plt.plot(x, yAcc, label='$l=' + str(n_sample_length) + '$: accuracy', color=colors[0], linestyle=':', linewidth=LINE_WIDTH)
		plt.fill_between(x, yAcc_m, yAcc_p, alpha=0.5, color=colors[0], linewidth=LINE_WIDTH)
		# Loss
		yLoss, yLoss_m, yLoss_p = json_data[dict_label_ci_loss]
		yLoss = running_mean(yLoss, RUNNING_MEAN_AGGREGATE)
		yLoss_m = running_mean(yLoss_m, RUNNING_MEAN_AGGREGATE)
		yLoss_p = running_mean(yLoss_p, RUNNING_MEAN_AGGREGATE)		
		plt.plot(x, yLoss, label='$l=' + str(n_sample_length) + '$: loss', color=colors[0], linewidth=LINE_WIDTH)
		plt.fill_between(x, yLoss_m, yLoss_p, alpha=0.5, color=colors[0], linewidth=LINE_WIDTH)		

	with open("_data/" + filename + "__vector.json") as json_file:
		json_data = json.load(json_file)

		# Accuracy
		yAcc, yAcc_m, yAcc_p = json_data[dict_label_ci_acc]
		yAcc = running_mean(yAcc, RUNNING_MEAN_AGGREGATE)
		yAcc_m = running_mean(yAcc_m, RUNNING_MEAN_AGGREGATE)
		yAcc_p = running_mean(yAcc_p, RUNNING_MEAN_AGGREGATE)
		x = np.array(range(1, len(yAcc) + 1))
		plt.plot(x, yAcc, label='$l=1$: accuracy', color=colors[1], linestyle=':', linewidth=LINE_WIDTH)
		plt.fill_between(x, yAcc_m, yAcc_p, alpha=0.5, color=colors[1], linewidth=LINE_WIDTH)
		# Loss
		yLoss, yLoss_m, yLoss_p = json_data[dict_label_ci_loss]
		yLoss = running_mean(yLoss, RUNNING_MEAN_AGGREGATE)
		yLoss_m = running_mean(yLoss_m, RUNNING_MEAN_AGGREGATE)
		yLoss_p = running_mean(yLoss_p, RUNNING_MEAN_AGGREGATE)		
		plt.plot(x, yLoss, label='$l=1$: loss', color=colors[1], linewidth=LINE_WIDTH)
		plt.fill_between(x, yLoss_m, yLoss_p, alpha=0.5, color=colors[1], linewidth=LINE_WIDTH)		

	fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
	fig.tight_layout()
	plt.legend(loc='center left', bbox_to_anchor=(.325, .58))
	plt.xlabel('Time step $t$')
	ax = plt.gca()
	ax.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.225), ncol=2)
	filename = "_imgs/" + filename + ".pdf"
	fig.savefig(filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)
	plt.close()
	print("Graph saved to " + filename)


def simulate_matrix_input(filename, n_channels=8, n_neurons=128, n_sample_length=16, n_reps=1, split=1, max_t=10000):
	filename = "_data/" + filename + ".json"

	# Result containers.
	loss_mat = np.zeros((n_reps, max_t))
	acc_mat = np.zeros((n_reps, max_t))

	# For each repetition...
	for rep in range(n_reps):
		print("Repetition " + str(rep+1) + " / " + str(n_reps))
		bar = progressbar.ProgressBar(max_value=max_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		loss_vec = np.zeros(max_t)
		acc_vec = np.zeros(max_t)
		env = get_env_partially_observable(n_channels)
		neural_network = LSTMRNN(n_channels, n_neurons, sample_length=n_sample_length)

		# Make an initial observation.
		observation_matrix = deque() #np.zeros((n_sample_length, 1, n_channels))
		action = np.random.randint(0, n_channels)
		observation_matrix.append(np.reshape(env.step(action), (1, n_channels)))

		# ... for each time slot ...
		for t in range(max_t):
			# Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example
			with tf.GradientTape() as tape:
				# Make new observation...
				action = np.random.randint(0, n_channels)
				observation = np.reshape(env.step(action), (1, n_channels))

				# ... if we've accumulated enough observations to start training
				if len(observation_matrix) == n_sample_length:
					# get prediction
					input_matrix = np.reshape(observation_matrix, (1, n_sample_length, n_channels))  # batch x samples x observation vectors
					prediction = neural_network.model(input_matrix, training=False)

					# compute binary accuracy
					state = env.observe_fully_observable()
					acc_vec[t] = sum([1 if (prediction[0][i] > 0 and state[i] == 1) or (prediction[0][i] < 0 and state[i] == -1) else 0 for i in range(n_channels)]) / n_channels

					# compute loss
					loss_value = neural_network.loss(np.reshape(observation, (1, 1, n_channels)), prediction)[-1, -1]
					loss_vec[t] = float(loss_value)

					# get gradients
					gradients = tape.gradient(loss_value, neural_network.model.trainable_weights)

					# train network
					neural_network.optimizer.apply_gradients(zip(gradients, neural_network.model.trainable_weights))

				# ... in any case, add the new observation to the matrix
				observation_matrix.append(observation)
				# ... and if we've accumulated too many observations, ...
				if len(observation_matrix) > n_sample_length:
					# ... drop the oldest observation
					observation_matrix.popleft()

			bar.update(t)
		# ... save results of this repetition
		acc_mat[rep] = acc_vec
		loss_mat[rep] = loss_vec
		bar.finish()

	json_data = {}
	json_data[dict_label_loss] = loss_mat.tolist()
	json_data[dict_label_acc] = acc_mat.tolist()

	# Calculate confidence intervals.
	batch_means_loss = columnwise_batch_means(loss_mat, split)
	yLoss = np.zeros(max_t)
	yLoss_m = np.zeros(max_t)
	yLoss_p = np.zeros(max_t)

	batch_means_acc = columnwise_batch_means(acc_mat, split)
	yAcc = np.zeros(max_t)
	yAcc_m = np.zeros(max_t)
	yAcc_p = np.zeros(max_t)
	for d in range(max_t):
		yLoss[d], yLoss_m[d], yLoss_p[d] = calculate_confidence_interval(batch_means_loss[:, d], confidence=.95)
		yAcc[d], yAcc_m[d], yAcc_p[d] = calculate_confidence_interval(batch_means_acc[:, d], confidence=.95)
	json_data[dict_label_ci_loss] = [yLoss.tolist(), yLoss_m.tolist(), yLoss_p.tolist()]
	json_data[dict_label_ci_acc] = [yAcc.tolist(), yAcc_m.tolist(), yAcc_p.tolist()]

	with open(filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Raw data saved to " + filename)


def simulate_vector_input(filename, n_channels=8, n_neurons=128, n_reps=1, split=1, max_t=10000):
	filename = "_data/" + filename + ".json"

	# Result containers.
	loss_mat = np.zeros((n_reps, max_t))
	acc_mat = np.zeros((n_reps, max_t))

	# For each repetition...
	for rep in range(n_reps):
		print("Repetition " + str(rep+1) + " / " + str(n_reps))
		bar = progressbar.ProgressBar(max_value=max_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		loss_vec = np.zeros(max_t)
		acc_vec = np.zeros(max_t)
		env = get_env_partially_observable(n_channels)
		neural_network = LSTMRNN(n_channels, n_neurons)

		# Make an initial observation.
		action = np.random.randint(0, n_channels)
		observation = np.reshape(env.step(action), (1, 1, n_channels))
		# ... for each time slot ...
		for t in range(max_t):
			# Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example
			with tf.GradientTape() as tape:
				# get prediction
				prediction = neural_network.model(observation, training=False)

				# get observation, which serves as the label (this should've been predicted)
				action = np.random.randint(0, n_channels)
				observation = np.reshape(env.step(action), (1, 1, n_channels))

				# compute binary accuracy
				state = env.observe_fully_observable()
				acc_vec[t] = sum([1 if (prediction[0][i] > 0 and state[i] == 1) or (prediction[0][i] < 0 and state[i] == -1) else 0 for i in range(n_channels)]) / n_channels

				# compute loss
				loss_value = neural_network.loss(observation, prediction)
				loss_vec[t] = float(loss_value)

				# get gradients
				gradients = tape.gradient(loss_value, neural_network.model.trainable_weights)

				# train network
				neural_network.optimizer.apply_gradients(zip(gradients, neural_network.model.trainable_weights))

			bar.update(t)
		# ... save results of this repetition
		acc_mat[rep] = acc_vec
		loss_mat[rep] = loss_vec
		bar.finish()

	json_data = {}
	json_data[dict_label_loss] = loss_mat.tolist()
	json_data[dict_label_acc] = acc_mat.tolist()

	# Calculate confidence intervals.
	batch_means_loss = columnwise_batch_means(loss_mat, split)
	yLoss = np.zeros(max_t)
	yLoss_m = np.zeros(max_t)
	yLoss_p = np.zeros(max_t)

	batch_means_acc = columnwise_batch_means(acc_mat, split)
	yAcc = np.zeros(max_t)
	yAcc_m = np.zeros(max_t)
	yAcc_p = np.zeros(max_t)
	for d in range(max_t):
		yLoss[d], yLoss_m[d], yLoss_p[d] = calculate_confidence_interval(batch_means_loss[:, d], confidence=.95)
		yAcc[d], yAcc_m[d], yAcc_p[d] = calculate_confidence_interval(batch_means_acc[:, d], confidence=.95)
	json_data[dict_label_ci_loss] = [yLoss.tolist(), yLoss_m.tolist(), yLoss_p.tolist()]
	json_data[dict_label_ci_acc] = [yAcc.tolist(), yAcc_m.tolist(), yAcc_p.tolist()]

	with open(filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Raw data saved to " + filename)


if __name__ == '__main__':
	filename = "netsys_partially_observable_comparison_loss_over_time_matrix_input"

	parser = argparse.ArgumentParser(description='Simulate data and plot the graph in _imgs/' + filename + ".pdf.")
	parser.add_argument('--no_sim', action='store_true', help='Whether *not* to simulate and produce data in _data/' + filename + ".json.")
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot a graph from the data in _data/' + filename + ".json to output _imgs/" + filename + ".pdf.")
	parser.add_argument('--use_gpu', action='store_true', help='Whether to disable GPU utilization in TensorFlow (default: False).')
	parser.add_argument('--gpu_mem', type=int, help='Limit GPU memory utilization to this value in MB (default: 1024)', default=1024)
	parser.add_argument('--n_channels', type=int, help='Number of frequency channels (default: 5).', default=5)
	parser.add_argument('--n_neurons', type=int, help='Number of neurons (default: 128).', default=128)
	parser.add_argument('--n_reps', type=int, help='Number of repetitions (default: 1).', default=1)
	parser.add_argument('--n_sample_length', type=int, help='Number of observations to aggregate into the input matrix (default: 6).', default=6)
	parser.add_argument('--split', type=int, help='How many repetitions to group into one mean value for the calculation of confidence intervals (default: 1). Note: n_reps must be divisible by split.', default=1)
	parser.add_argument('--max_t', type=int, help='Number of time slots (default: 25000).', default=25000)

	args = parser.parse_args()

	if not args.no_sim:
		if not args.use_gpu:
			print("Disabling GPU.")
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
		else:
			# Limit TensorFlow to 1GB of VRAM to run multiple instances in parallel
			# Source: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
			gpus = tf.config.list_physical_devices('GPU')
			for gpu in gpus:
				try:
					tf.config.experimental.set_virtual_device_configuration(
						gpu,
						[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_mem)]
					)
				except RuntimeError as e:
					print(e)

		if not args.n_reps % args.split == 0:
			parser.error("n_reps and split must be divisible")

		print("Simulation: " + str(args.n_sample_length) + " observation samples")
		simulate_matrix_input(filename + "__matrix", args.n_channels, args.n_neurons, args.n_sample_length, args.n_reps, args.split, args.max_t)
		print("Simulation: single observation sample")
		simulate_vector_input(filename + "__vector", args.n_channels, args.n_neurons, args.n_reps, args.split, args.max_t)
	# print("Simulation: partially observable")
	# simulate_partially_observable(filename + "__partial", args.n_channels, args.n_neurons, args.n_sample_length, args.n_reps, args.split, args.max_t)
	if not args.no_plot:
		plot(filename, args.n_sample_length)

