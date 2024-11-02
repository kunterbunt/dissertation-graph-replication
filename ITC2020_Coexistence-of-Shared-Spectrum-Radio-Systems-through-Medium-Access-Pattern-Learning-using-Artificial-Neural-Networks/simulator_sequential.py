import os, sys
import util.settings
from environment.data_generator import *
from predictor.neural_network import *
from util.callback_loss_history import LossHistory, AccuracyHistory
import matplotlib.pyplot as plt
import numpy as np
from util.confidence_intervals import *
import progressbar
import json
import argparse
from pathlib import Path


json_label_loss = 'loss_mat'
json_label_prediction = 'prediction_vec'
json_label_actually_idle = 'first_channel_actually_idle'
json_label_validation_timeslots = 'validation_timeslots'
json_label_switching_prob = 'switching_prob'
json_label_num_channels = 'num_channels'


def run_with_validation(num_neurons, num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length, learning_rate, num_training_samples, num_validation_samples, num_reps, confidence, batch_size):
	mean_accuracy_vec = np.zeros(num_reps)
	neural_network = None

	for rep in range(num_reps):
		print("- repetition " + str(rep+1) + "/" + str(num_reps) + " -")
		data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
		neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers)

		# Generate training data.
		observation_mat, labels_mat = data_generator.read_next(num_training_samples)  # num_samples x num_channel matrices
		# reshape to batch x time_step x data
		observation_mat = np.reshape(observation_mat, (num_training_samples, sample_length, num_channels))
		labels_mat = np.reshape(labels_mat, (num_training_samples, sample_length, num_channels))
		# Train.
		neural_network.get_keras_model().fit(observation_mat, labels_mat)

		# Generate validation data.
		validation_mat, validation_labels = data_generator.read_next(num_validation_samples)
		validation_mat = np.reshape(validation_mat, (num_validation_samples, sample_length, num_channels))
		validation_labels = np.reshape(validation_labels, (num_validation_samples, sample_length, num_channels))
		# Validate.
		_, mean_accuracy_vec[rep] = neural_network.get_keras_model().evaluate(x=validation_mat, y=validation_labels)

	sample_mean, sample_mean_minus, sample_mean_plus = calculate_confidence_interval(mean_accuracy_vec, confidence)
	return sample_mean, sample_mean_minus, sample_mean_plus, neural_network


def shift(arr, shift, fill_value=np.nan):
	"""
	Shifts array 'arr' contents by 'shift'.
	:param arr:
	:param shift:
	:param fill_value:
	:return:
	"""
	result = np.empty_like(arr)
	if shift > 0:
		result[:shift] = fill_value
		result[shift:] = arr[:-shift]
	elif shift < 0:
		result[shift:] = fill_value
		result[:shift] = arr[-shift:]
	else:
		result[:] = arr
	return result


def simulate_loss(num_channels, switching_prob, time_limit, num_reps, json_results_filename):
	"""
	:return: _data/lstm_loss_per_timeslot.json
	"""	
	activation_pattern = list(range(0, num_channels))	
	sample_length = 1
	learning_rate = 0.005	
	num_hidden_layers = 1
	num_neurons = [200]		

	loss_mat = np.zeros((num_reps, time_limit))
	for rep in range(num_reps):
		print('Repetition ' + str(rep+1) + " / " + str(num_reps))
		data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
		neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers, use_softmax=False)
		loss_vec = np.zeros(time_limit)

		observation_mat = np.zeros((sample_length, num_channels))
		label_mat = np.zeros((sample_length, num_channels))

		bar = progressbar.ProgressBar(max_value=time_limit, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])		
		bar.start()
		for timeslot in range(time_limit):
			# Read next observation.
			observation_mat[timeslot % sample_length], label_mat[timeslot % sample_length] = data_generator.read_next(1)			
			if timeslot % sample_length == 0:  # when history_size > 1 this ensures that training only happens when enough observations have been aggregated
				reshaped_input_matrix = np.reshape(observation_mat, (1, sample_length, num_channels))
				reshaped_label_matrix = np.reshape(label_mat, (1, sample_length, num_channels))
				# .fit is called not for the entire data, but for each input, so that the loss for every timeslot can be obtained
				history = LossHistory()
				neural_network.get_keras_model().fit(reshaped_input_matrix, reshaped_label_matrix, callbacks=[history], verbose=False)
				loss_vec[timeslot] = history.losses[0]
			bar.update(timeslot)
		bar.finish()
		loss_mat[rep] = loss_vec

	# Write simulation results to file
	json_data = {}
	json_data[json_label_loss] = loss_mat.tolist()	
	with open(json_results_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved simulation results to '" + json_results_filename + "'.")


def plot_loss(json_results_filename, batch_means_split, time_limit, num_channels, graph_filename):
	with open(json_results_filename) as json_file:
		json_data = json.load(json_file)		
		loss_mat = np.array(json_data[json_label_loss])
		# Compute batch-means for every data point.
		batch_means = columnwise_batch_means(loss_mat, batch_means_split)
		# Compute range for each data point using confidence intervals.
		sample_means = np.zeros(time_limit)
		sample_means_minus = np.zeros(time_limit)
		sample_means_plus = np.zeros(time_limit)
		for data_point in range(time_limit):
			sample_means[data_point], sample_means_minus[data_point], sample_means_plus[data_point] = calculate_confidence_interval(batch_means[:,data_point], confidence=.95)

		fig = plt.figure()
		util.settings.set_params()
		x = range(1, time_limit+1)		
		plt.plot(x, sample_means, label="LSTM Loss", color='xkcd:teal')
		plt.fill_between(x, sample_means_minus, sample_means_plus, color='xkcd:teal', alpha=0.5)						
		added_label = False
		x_ticks = []
		i = 0
		for x in range(num_channels, time_limit+1, num_channels):
			if not added_label:
				plt.axvline(x=x, color='gray', label='end of full pattern', linestyle='--', linewidth=0.5, )
				added_label = True
			else:
				plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, )			
			# add every second iteration to x_ticks
			# if i % 2 == 0:
			x_ticks.append(x)
			# i += 1
		plt.xticks(x_ticks)
		plt.plot(range(time_limit), [1/num_channels]*time_limit, color='black', label='random guessing', linestyle='--')
		plt.legend()
		plt.xlabel('Time step $t$')
				
		util.settings.init()
		fig.set_size_inches((2*util.settings.fig_width, util.settings.fig_height), forward=False)
		fig.tight_layout()
		fig.savefig(graph_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)				
		print("Graph saved to " + graph_filename)
		plt.close()



def simulate_prediction_over_time(num_time_steps, num_channels, switching_prob, json_filename):	
	activation_pattern = list(range(0, num_channels))		
	sample_length = 1
	learning_rate = 0.005
	num_training_samples = 1000 * num_channels
	num_hidden_layers = 2
	num_neurons = [200, 150]

	# Get trained neural network.
	data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
	neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers, use_softmax=True)
	# Generate training data.
	observation_mat, labels_mat = data_generator.read_next(num_training_samples)  # num_samples x num_channel matrices
	# reshape to batch x time_step x data
	observation_mat = np.reshape(observation_mat, (num_training_samples, sample_length, num_channels))
	labels_mat = np.reshape(labels_mat, (num_training_samples, sample_length, num_channels))
	# Train.
	neural_network.get_keras_model().fit(observation_mat, labels_mat, shuffle=False, batch_size=1)
	
	# Keep track of the prediction value for the first channel being idle...
	prediction_vec = np.zeros(num_time_steps)
	first_channel_actually_idle = []
	for i in range(num_time_steps):
		observation, label = data_generator.read_next(1)
		if np.argmax(observation) == 0:
			first_channel_actually_idle.append(i)
		observation = np.reshape(observation, (1, sample_length, num_channels))
		prediction = neural_network.get_keras_model().predict(x=observation)
		prediction_on_first_channel_being_idle = prediction[0,0]
		prediction_vec[i] = prediction_on_first_channel_being_idle

	# Write simulation results to file
	json_data = {}
	json_data[json_label_prediction] = prediction_vec.tolist()	
	json_data[json_label_actually_idle] = first_channel_actually_idle
	json_data[json_label_validation_timeslots] = num_time_steps
	json_data[json_label_switching_prob] = switching_prob	
	json_data[json_label_num_channels] = num_channels	
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved simulation results to '" + json_filename + "'.")	


def plot_prediction_over_time(json_filename, graph_filename):
	with open(json_filename) as json_file:
		json_data = json.load(json_file)		
		prediction_vec = np.array(json_data[json_label_prediction])		
		first_channel_actually_idle = np.array(json_data[json_label_actually_idle])
		validation_timeslots = json_data[json_label_validation_timeslots]		
		switching_prob = json_data[json_label_switching_prob]				
		num_channels = json_data[json_label_num_channels]
				
		fig = plt.figure()
		util.settings.init()
		util.settings.set_params()
		plt.xlabel('Time step $t$')
		plt.scatter(range(1, validation_timeslots+1), prediction_vec, label="$h_\Theta{}$(1st channel idle)", color='xkcd:teal', marker='.')
		for i in range(len(first_channel_actually_idle)):
			if i==0:
				plt.axvline(first_channel_actually_idle[i], color='gray', alpha=0.5, linestyle='--', label='1st channel idle', linewidth=.5)
			else:
				plt.axvline(first_channel_actually_idle[i], color='gray', alpha=0.5, linestyle='--', linewidth=.5)
		plt.axhline(switching_prob, label="switching prob.", color='black', alpha=0.9, linestyle='dotted', linewidth=1)
		plt.xticks(range(num_channels, validation_timeslots+1, num_channels))
		plt.yticks([0, 0.25, 0.5, 0.75])						
		plt.legend(loc='center left', bbox_to_anchor=(.0, .65), framealpha=1.0)
				
		fig.set_size_inches((2*util.settings.fig_width, util.settings.fig_height), forward=False)
		fig.tight_layout()
		fig.savefig(graph_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename)
		plt.close()				


def plot_channel_access_pattern(num_channels, switching_prob, time_limit, graph_filename):
	activation_pattern = list(range(0, num_channels))	
	sample_length = 1
	data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
	observation_mat = np.zeros((time_limit, num_channels))
	observation_mat[0][0] = 1
	for timeslot in range(1, time_limit):			
		observation_mat[timeslot], _ = data_generator.read_next(1)			
	util.settings.set_params()
	plt.ylabel('Channel')
	plt.xlabel('Time step')		
	plt.yticks([])
	plt.xticks(range(0, time_limit, num_channels))
	plt.imshow(np.transpose(observation_mat), cmap='Greys') 	
	fig = plt.gcf()
	fig.tight_layout()
	util.settings.init()
	fig.set_size_inches((0.8*util.settings.fig_width, util.settings.fig_height), forward=False)
	fig.savefig(graph_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
	plt.close()
	print("Graph saved to " + graph_filename)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate graphs on channel access predictions of sequential behavior.')	
	parser.add_argument('--imgs_dir', type=str, help='Directory path that contains the graph files.', default='_imgs')
	parser.add_argument('--data_dir', type=str, help='Directory path that contains the result files.', default='_data')
	parser.add_argument('--t', type=int, help='Number of timeslots.', default=160)
	parser.add_argument('--c', type=int, help='Number of frequency channels.', default=16)
	parser.add_argument('--p', type=float, help='Transition probability.', default=1.0)	
	parser.add_argument('--rep', type=int, help='Number of repetitions.', default=20)	
	parser.add_argument('--split', type=int, help='Split simulations into batch means that aggregate this many runs.', default=4)	
	parser.add_argument('--no_sim_loss', action='store_true', help='Whether not to generate lstm_loss_per_timeslot.json.')		
	parser.add_argument('--no_plot_loss', action='store_true', help='Whether not to generate lstm_loss_per_timeslot.pdf.')		
	parser.add_argument('--no_sim_predictions', action='store_true', help='Whether not to generate lstm_prediction_over_timeslots.json.')		
	parser.add_argument('--no_plot_predictions', action='store_true', help='Whether not to generate lstm_prediction_over_timeslots.pdf.')		
	parser.add_argument('--no_plot_channel_access', action='store_true', help='Whether not to generate channel_access_observations.pdf.')			

	args = parser.parse_args()	

	# create dirs if they don't exist
	Path(args.imgs_dir).mkdir(parents=True, exist_ok=True)		
	Path(args.data_dir).mkdir(parents=True, exist_ok=True)		

	filename_base__loss = "sequential_loss_over_time_t-" + str(args.t) + "_p-" + str(args.p) + "_c-" + str(args.c) + "_rep-" + str(args.rep)
	json_results_filename__loss = args.data_dir + "/" + filename_base__loss + ".json"
	graph_filename__loss = args.imgs_dir + "/" + filename_base__loss + ".pdf"
	graph_filename__channel_access = args.imgs_dir + "/channel_access_observations_t-" + str(args.t) + "_p-" + str(args.p) + "_c-" + str(args.c) + ".pdf"
	json_filename__predictions = args.imgs_dir + "/lstm_prediction_over_timeslots.json"		
	graph_filename__predictions = args.imgs_dir + "/lstm_prediction_over_timeslots.pdf"		
	
	if not args.no_plot_channel_access:
		plot_channel_access_pattern(args.c, args.p, args.t, graph_filename__channel_access)	
	if not args.no_sim_loss:
		simulate_loss(num_channels=args.c, switching_prob=args.p, time_limit=args.t, num_reps=args.rep, json_results_filename=json_results_filename__loss)
	if not args.no_plot_loss:
		plot_loss(json_results_filename=json_results_filename__loss, batch_means_split=args.split, time_limit=args.t, num_channels=args.c, graph_filename=graph_filename__loss)
	if not args.no_sim_predictions:					
		simulate_prediction_over_time(args.t, args.c, args.p, json_filename__predictions) 
	if not args.no_plot_predictions:
		plot_prediction_over_time(json_filename__predictions, graph_filename__predictions) 