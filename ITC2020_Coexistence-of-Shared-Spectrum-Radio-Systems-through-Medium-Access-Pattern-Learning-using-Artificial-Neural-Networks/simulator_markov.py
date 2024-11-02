import os, sys
import util.settings
from unicodedata import numeric
from predictor.neural_network import *
from environment.data_generator import *
import numpy as np
from util.callback_loss_history import *
import matplotlib.pyplot as plt
from util.confidence_intervals import *
from pathlib import Path
import argparse
import json


json_label_predictions_idle = 'predictions_idle'
json_label_predictions_busy = 'predictions_busy'


def generate_data(channel, num_timeslots, sample_length):
	"""
	:param channel: The channel model.
	:param num_timeslots: Number of timeslots to generate.
	:param sample_length: Length of one input sample.
    :return: (num_timeslots/sample_length, sample_vec)-matrix.
	"""
	num_samples = int(num_timeslots / sample_length)
	data_mat = np.zeros((num_samples, sample_length))
	label_vec = np.zeros(num_samples)
	for sample in range(num_samples):
		for i in range(sample_length):
			channel.update()
			data_mat[sample][i] = channel.get_state_vector()[0]
			if i == 0 and sample > 0:
				label_vec[sample - 1] = channel.get_state_vector()[0]
	channel.update()
	label_vec[num_samples - 1] = channel.get_state_vector()[0]
	return data_mat, label_vec


def simulate(num_timeslots, num_repetitions, p, q, json_results_filename):
	"""
	:return: writes simulation results into a JSON-formatted file.
	"""	
	channel = TransitionProbPoissonProcessChannelModel(p, q)

	sample_length = 1  # We input a single observation into the neural network.
	neural_network = TumuluruMLPAdam(lookback_length=sample_length)	
	num_samples = int(num_timeslots / sample_length)

	# Prepare idle channel input vector.
	input_vec_idle = np.zeros((1, sample_length))

	# And the same for a busy channel.
	input_vec_busy = np.zeros((1, sample_length))
	for i in range(len(input_vec_busy)):
		input_vec_busy[0, i] = 1

	predictions_idle = np.zeros((num_repetitions, num_samples))
	predictions_busy = np.zeros((num_repetitions, num_samples))
	for rep in range(num_repetitions):
		print("Repetition " + str(rep+1) + " / " + str(num_repetitions))
		# This keeps track of what the neural network predicts for an idle channel after every training batch.
		prediction_history_idle = PredictionHistory(neural_network, input_vec_idle, num_timeslots, verbose=True)
		# And the prediction on a now-busy channel.
		prediction_history_busy = PredictionHistory(neural_network, input_vec_busy, num_timeslots)
		
		training_data, training_labels = generate_data(channel, num_timeslots, sample_length)		
		neural_network.get_keras_model().fit(x=training_data, y=training_labels, batch_size=1, callbacks=[prediction_history_idle, prediction_history_busy], verbose=0)
		predictions_idle[rep] = np.reshape(prediction_history_idle.predictions, len(prediction_history_idle.predictions))
		predictions_busy[rep] = np.reshape(prediction_history_busy.predictions, len(prediction_history_idle.predictions))

	# Write simulation results to file
	json_data = {}
	json_data[json_label_predictions_idle] = predictions_idle.tolist()
	json_data[json_label_predictions_busy] = predictions_busy.tolist()
	with open(json_results_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved simulation results to '" + json_results_filename + "'.")    	


def plot(json_results_filename, batch_means_split, num_timeslots, p, q, graph_filename):
	"""
	:return: writes graphical results into a PDF-formatted file.
	"""	
	with open(json_results_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)		
		predictions_idle = np.array(json_data[json_label_predictions_idle])
		predictions_busy = np.array(json_data[json_label_predictions_busy])

		# Collect batch means.
		idle_batch_means = columnwise_batch_means(predictions_idle, batch_means_split)
		busy_batch_means = columnwise_batch_means(predictions_busy, batch_means_split)

		# Calculate confidence intervals on the batch means.
		idle_ci_means = np.zeros(num_timeslots)
		idle_ci_minus = np.zeros(num_timeslots)
		idle_ci_plus = np.zeros(num_timeslots)
		busy_ci_means = np.zeros(num_timeslots)
		busy_ci_minus = np.zeros(num_timeslots)
		busy_ci_plus = np.zeros(num_timeslots)
		confidence = 0.95
		for timeslot in range(num_timeslots):
			idle_ci_means[timeslot], idle_ci_minus[timeslot], idle_ci_plus[timeslot] = calculate_confidence_interval(idle_batch_means[:,timeslot], confidence)
			busy_ci_means[timeslot], busy_ci_minus[timeslot], busy_ci_plus[timeslot] = calculate_confidence_interval(busy_batch_means[:,timeslot], confidence)
		x = range(len(idle_ci_means))

		util.settings.init()
		util.settings.set_params()
		fig = plt.figure()		
		plt.xlabel('Time step $t$')

		plt.axhline(y=1-p, color='black', linestyle='--', linewidth=1, alpha=.75)
		plt.axhline(y=q, color='black', linestyle='--', linewidth=1, alpha=.75)

		colors = ['xkcd:teal', 'xkcd:coral']
		plt.plot(x, idle_ci_means, color=colors[0], linewidth=1, label="$h_\\Theta{}($idle $\Leftrightarrow x=1)$")
		plt.fill_between(x, idle_ci_minus, idle_ci_plus, facecolor=colors[0], alpha=0.25)

		plt.plot(x, busy_ci_means, color=colors[1], linewidth=1, label="$h_\\Theta{}($busy $\Leftrightarrow x=0)$")
		plt.fill_between(x, busy_ci_minus, busy_ci_plus, facecolor=colors[1], alpha=0.25)

		plt.ylim(0, 1)
		plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)

		plt.yticks([0, q, 0.5, 1-p, 1.0])		
		locs, labels = plt.yticks()		
		labels[0] = '$0$'
		labels[1] = '$q$'
		labels[2] = '$0.5$'
		labels[3] = '$1-p$'
		labels[4] = '$1$'
		plt.yticks(locs, labels)
		
		fig.tight_layout()
		fig.set_size_inches((util.settings.fig_width, util.settings.fig_height), forward=False)
		fig.savefig(graph_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("File saved to " + graph_filename)
		plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate graphs on channel access predictions of Markovian behavior.')	
	parser.add_argument('--imgs_dir', type=str, help='Directory path that contains the graph files.', default='_imgs')
	parser.add_argument('--data_dir', type=str, help='Directory path that contains the result files.', default='_data')
	parser.add_argument('--t', type=int, help='Number of timeslots.', default=2500)
	parser.add_argument('--p', type=float, help='Transition probability p.', default=0.1)
	parser.add_argument('--q', type=float, help='Transition probability q.', default=0.25)
	parser.add_argument('--no_sim_single', action='store_true', help='Whether not to generate markovian_predictions_over_time.json.')		
	parser.add_argument('--no_plot_single', action='store_true', help='Whether not to generate markovian_predictions_over_time.pdf.')		
	parser.add_argument('--no_sim_avg', action='store_true', help='Whether not to generate markovian_predictions_over_time-averages.json.')		
	parser.add_argument('--no_plot_avg', action='store_true', help='Whether not to generate markovian_predictions_over_time-averages.pdf.')		

	args = parser.parse_args()	

	# create dirs if they don't exist
	Path(args.imgs_dir).mkdir(parents=True, exist_ok=True)		
	Path(args.data_dir).mkdir(parents=True, exist_ok=True)		

	filename_base__single = "markovian_predictions_over_time_t-" + str(args.t) + "_p-" + str(args.p) + "_q-" + str(args.q)
	json_results_filename__single = args.data_dir + "/" + filename_base__single + ".json"
	graph_filename__single = args.imgs_dir + "/" + filename_base__single + ".pdf"
	if not args.no_sim_single:		
		simulate(num_timeslots=args.t, num_repetitions=1, p=args.p, q=args.q, json_results_filename=json_results_filename__single)	
	if not args.no_plot_single:
		plot(json_results_filename=json_results_filename__single, batch_means_split=1, num_timeslots=args.t, p=args.p, q=args.q, graph_filename=graph_filename__single)

	filename_base__avg = "markovian_predictions_over_time_t-" + str(args.t) + "_p-" + str(args.p) + "_q-" + str(args.q) + "_averages"
	json_results_filename__avg = args.data_dir + "/" + filename_base__avg + ".json"
	graph_filename__avg = args.imgs_dir + "/" + filename_base__avg + ".pdf"
	if not args.no_sim_avg:			
		simulate(num_timeslots=args.t, num_repetitions=12, p=args.p, q=args.q, json_results_filename=json_results_filename__avg)	
	if not args.no_plot_avg:
		plot(json_results_filename=json_results_filename__avg, batch_means_split=3, num_timeslots=args.t, p=args.p, q=args.q, graph_filename=graph_filename__avg)	
