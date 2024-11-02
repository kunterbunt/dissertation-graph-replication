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

import matplotlib.pyplot as plt
import progressbar
import sys
import tensorflow as tf
import numpy as np
from env import *
from neural_network import LSTMRNN
import os
import argparse
from confidence_intervals import columnwise_batch_means, calculate_confidence_interval
from running_mean import running_mean
import settings

dict_label_x = 'x'
dict_label_conv = 'conv_mat'
dict_label_ci_conv = 'loss_ci_conv'
LINE_WIDTH = 1


def simulate_partially_observable(filename, n_channels_vec=[2, 3, 4, 5, 6, 7], n_neurons=128, n_reps=1, split=1, max_t=25000, convergence_criterion=1000):
	filename = "_data/" + filename + ".json"

	# Result containers.
	total_num_channels = len(n_channels_vec)
	conv_mat = np.zeros((n_reps, total_num_channels, 1))

	# For each repetition...
	for rep in range(n_reps):
		print("Repetition " + str(rep+1) + " / " + str(n_reps))
		# ... for each number of channels...
		i = 0
		for n_channels in n_channels_vec:
			print("\t#Channels: " + str(n_channels) + " / " + str(n_channels_vec[-1]))
			bar = progressbar.ProgressBar(max_value=max_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
			bar.start()

			acc_vec = np.zeros(max_t)
			env = get_env_partially_observable(n_channels)
			neural_network = LSTMRNN(n_channels, n_neurons)

			has_converged = False
			t = 0
			t_convergence = 0

			# Make an initial observation.
			action = np.random.randint(0, n_channels)
			observation = np.reshape(env.step(action), (1, 1, n_channels))
			# ... until convergence ...
			while not has_converged and t < max_t:
				# Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example
				with tf.GradientTape() as tape:
					# get prediction
					prediction = neural_network.model(observation, training=False)

					# get observation, which serves as the label (this should've been predicted)
					action = np.random.randint(0, n_channels)
					observation = np.reshape(env.step(action), (1, 1, n_channels))

					# compute loss
					loss_value = neural_network.loss(observation, prediction)

					# get gradients
					gradients = tape.gradient(loss_value, neural_network.model.trainable_weights)

					# train network
					neural_network.optimizer.apply_gradients(zip(gradients, neural_network.model.trainable_weights))

					# check for convergence
					state = env.observe_fully_observable()
					acc_vec[t] = sum([1 if (prediction[0][i] > 0 and state[i] == 1) or (prediction[0][i] < 0 and state[i] == -1) else 0 for i in range(n_channels)]) / n_channels
					correct_predictions = sum(acc_vec[t-convergence_criterion-1:t-1])
					if t > convergence_criterion and correct_predictions == convergence_criterion:
						print("Convergence reached at t=" + str(t))
						has_converged = True
						t_convergence = t - convergence_criterion

				bar.update(t)
				t += 1
			# ... save results of this repetition
			conv_mat[rep, i] = t_convergence
			bar.finish()
			i += 1

	json_data = {}
	json_data[dict_label_conv] = conv_mat.tolist()
	json_data[dict_label_x] = n_channels_vec

	# Calculate confidence intervals.
	batch_means_loss = columnwise_batch_means(conv_mat, split)
	yConv = np.zeros(total_num_channels)
	yConv_m = np.zeros(total_num_channels)
	yConv_p = np.zeros(total_num_channels)

	for c in range(total_num_channels):
		yConv[c], yConv_m[c], yConv_p[c] = calculate_confidence_interval(batch_means_loss[:, c], confidence=.95)
	json_data[dict_label_ci_conv] = [yConv.tolist(), yConv_m.tolist(), yConv_p.tolist()]

	with open(filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Raw data saved to " + filename)


def simulate_fully_observable(filename, n_channels_vec=[2, 3, 4, 5, 6, 7], n_neurons=128, n_reps=1, split=1, max_t=25000, convergence_criterion=1000):
	filename = "_data/" + filename + ".json"

	# Result containers.
	total_num_channels = len(n_channels_vec)
	conv_mat = np.zeros((n_reps, total_num_channels, 1))

	# For each repetition...
	for rep in range(n_reps):
		print("Repetition " + str(rep+1) + " / " + str(n_reps))
		# ... for each number of channels...
		i = 0
		for n_channels in n_channels_vec:
			print("\t#Channels: " + str(n_channels) + " / " + str(n_channels_vec[-1]))
			bar = progressbar.ProgressBar(max_value=max_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
			bar.start()

			acc_vec = np.zeros(max_t)
			env = get_env_fully_observable(n_channels)
			neural_network = LSTMRNN(n_channels, n_neurons)

			has_converged = False
			t = 0
			t_convergence = 0

			# Make an initial observation.
			observation = np.reshape(env.step(), (1, 1, n_channels))
			# ... until convergence ...
			while not has_converged and t < max_t:
				# Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example
				with tf.GradientTape() as tape:
					# get prediction
					prediction = neural_network.model(observation, training=False)

					# get observation, which serves as the label (this should've been predicted)
					observation = np.reshape(env.step(), (1, 1, n_channels))

					# compute loss
					loss_value = tf.keras.losses.mse(observation, prediction)

					# get gradients
					gradients = tape.gradient(loss_value, neural_network.model.trainable_weights)

					# train network
					neural_network.optimizer.apply_gradients(zip(gradients, neural_network.model.trainable_weights))

					# check for convergence
					state = env.observe_fully_observable()
					acc_vec[t] = sum([1 if (prediction[0][i] > 0 and state[i] == 1) or (prediction[0][i] < 0 and state[i] == -1) else 0 for i in range(n_channels)]) / n_channels
					correct_predictions = sum(acc_vec[t-convergence_criterion-1:t-1])
					if t > convergence_criterion and correct_predictions == convergence_criterion:
						print("Convergence reached at t=" + str(t))
						has_converged = True
						t_convergence = t - convergence_criterion

				bar.update(t)
				t += 1
			# ... save results of this repetition
			conv_mat[rep, i] = t_convergence
			bar.finish()
			i += 1

	json_data = {}
	json_data[dict_label_conv] = conv_mat.tolist()
	json_data[dict_label_x] = n_channels_vec

	# Calculate confidence intervals.
	batch_means_loss = columnwise_batch_means(conv_mat, split)
	yConv = np.zeros(total_num_channels)
	yConv_m = np.zeros(total_num_channels)
	yConv_p = np.zeros(total_num_channels)

	for c in range(total_num_channels):
		yConv[c], yConv_m[c], yConv_p[c] = calculate_confidence_interval(batch_means_loss[:, c], confidence=.95)
	json_data[dict_label_ci_conv] = [yConv.tolist(), yConv_m.tolist(), yConv_p.tolist()]

	with open(filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Raw data saved to " + filename)


def plot(filename):
	yConv_full = None
	yConv_m_full = None
	yConv_p_full = None
	x = None
	with open("_data/" + filename + "__full.json") as json_file:
		json_data = json.load(json_file)
		x = json_data[dict_label_x]		
		yConv_full, yConv_m_full, yConv_p_full = json_data[dict_label_ci_conv]		

	yConv_partial = None
	yConv_m_partial = None
	yConv_p_partial = None
	with open("_data/" + filename + "__partial.json") as json_file:
		json_data = json.load(json_file)		
		yConv_partial, yConv_m_partial, yConv_p_partial = json_data[dict_label_ci_conv]		
  
	# Plot
	colors = ['xkcd:teal', 'xkcd:coral']
	fig = plt.figure()
	settings.init()
	settings.set_params()
	plt.xlabel('no. of frequency channels $c$')
	plt.ylabel('convergence time')
	# plt.scatter(x, yConv_full, color=colors[0], linewidth=LINE_WIDTH)
	_, caps, bars = plt.errorbar(x, yConv_full, label='fully observable', yerr=np.array(yConv_p_full) - np.array(yConv_full), alpha=0.5, color=colors[0], fmt='o')
	[bar.set_alpha(1) for bar in bars]
	[cap.set_alpha(1) for cap in caps]
	# plt.scatter(x, yConv_partial, color=colors[1], linewidth=LINE_WIDTH, marker='x')	
	_, caps, bars = plt.errorbar(x, yConv_partial, yerr=np.array(yConv_p_partial) - np.array(yConv_partial), alpha=0.5, color=colors[1], fmt='x', label='partially observable')
	[bar.set_alpha(1) for bar in bars]
	[cap.set_alpha(1) for cap in caps]
	ax = plt.gca()
	ax.set_yscale('log')	
	plt.legend()
	ax.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.4, 1.2), ncol=2, columnspacing=0.5)
	plt.xticks(x)	
	fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
	fig.tight_layout()
	pdf_filename = "_imgs/" + filename + ".pdf"
	fig.savefig(pdf_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)
	plt.close()
	print("Graph saved to " + pdf_filename)


if __name__ == '__main__':
	filename = "netsys_comparison_convergence_over_channels"

	parser = argparse.ArgumentParser(description='Simulate data and plot the graph in _imgs/' + filename + ".pdf.")
	parser.add_argument('--no_sim', action='store_true', help='Whether *not* to simulate and produce data in _data/' + filename + ".json.")
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot a graph from the data in _data/' + filename + ".json to output _imgs/" + filename + ".pdf.")
	parser.add_argument('--use_gpu', action='store_true', help='Whether to disable GPU utilization in TensorFlow (default: False).')
	parser.add_argument('--gpu_mem', type=int, help='Limit GPU memory utilization to this value in MB (default: 1024)', default=1024)
	parser.add_argument('--n_channels', nargs='+', type=int, help='Number of frequency channels to evaluate (default: [2, 3, 4, 5, 6, 7]).', default=[2, 3, 4, 5, 6, 7])
	parser.add_argument('--n_neurons', type=int, help='Number of neurons (default: 128).', default=128)
	parser.add_argument('--n_reps', type=int, help='Number of repetitions (default: 1).', default=1)
	parser.add_argument('--split', type=int, help='How many repetitions to group into one mean value for the calculation of confidence intervals (default: 1). Note: n_reps must be divisible by split.', default=1)
	parser.add_argument('--max_t', type=int, help='Maximum number of time slots (default: 250000).', default=250000)
	parser.add_argument('--convergence_criterion', type=int, help='Number of consecutive time slots that must have been correctly predicted for a model to be considered converged (default: 1000).', default=1000)

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
		print("Simulation: fully observable")
		simulate_fully_observable(filename + "__full", args.n_channels, args.n_neurons, args.n_reps, args.split, args.max_t, args.convergence_criterion)
		print("Simulation: partially observable")
		simulate_partially_observable(filename + "__partial", args.n_channels, args.n_neurons, args.n_reps, args.split, args.max_t, args.convergence_criterion)
	if not args.no_plot:
		plot(filename)

