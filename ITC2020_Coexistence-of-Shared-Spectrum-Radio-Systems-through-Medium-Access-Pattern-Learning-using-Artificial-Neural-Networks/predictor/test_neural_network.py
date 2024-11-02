import unittest
from predictor.neural_network import *
from environment.data_generator import *
import numpy as np


class TestNeuralNetwork(unittest.TestCase):
	def setUp(self):
		self.num_channels = 16
		self.history_size = 1
		self.switching_prob = 1.0
		self.activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
		self.data_generator = DataGenerator(self.num_channels, self.history_size, SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern), update_once=True)

		self.num_hidden_neurons = 200
		self.learning_rate = 0.005
		self.neural_network = NeuralNetwork(self.num_channels, self.num_hidden_neurons, self.learning_rate)

	def test_prediction(self):
		# Generate training data.
		num_channel_updates = 250 * self.num_channels  # observe the pattern this many times
		training_data, training_labels = self.data_generator.generate_data(num_channel_updates)
		# Train network.
		self.neural_network.get_keras_model().fit(x=training_data, y=training_labels)

		# Generate next observation vector.
		observation_vec, label_vec = self.data_generator.read_last_row()
		correct_label = np.argmax(label_vec)
		# Just make sure this is right...
		self.assertEqual(training_data[-1][-1], 1)  # last column in last row contains the idle channel
		self.assertEqual(observation_vec[0], 1)  # first column in vector is the idle channel

		# Can we correctly predict the next idle channel?
		observation_vec = np.reshape(observation_vec, (1, len(observation_vec))) # reshape into 1xnum_channels
		prediction_vec = self.neural_network.predict(observation_vec)
		predicted_label = np.argmax(prediction_vec)
		self.assertEqual(predicted_label, correct_label)

		# And continue as well?
		for i in range(10*self.num_channels):
			self.data_generator.update()
			observation_vec, label_vec = self.data_generator.read_last_row()
			correct_label = np.argmax(label_vec)
			observation_vec = np.reshape(observation_vec, (1, len(observation_vec))) # reshape into 1xnum_channels
			prediction_vec = self.neural_network.predict(observation_vec)
			predicted_label = np.argmax(prediction_vec)
			self.assertEqual(predicted_label, correct_label)


class TestLookbackNeuralNetwork(unittest.TestCase):
	def setUp(self):
		self.num_channels = 16
		self.history_size = 16
		self.switching_prob = 1.0
		self.activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
		self.data_generator = DataGenerator(self.num_channels, self.history_size, SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern), update_once=True)

		self.num_hidden_neurons = [200]
		self.learning_rate = 0.005
		self.neural_network = LookbackNeuralNetwork(self.num_channels, self.num_hidden_neurons, self.learning_rate, lookback_length=self.history_size)

	def test_prediction(self):
		# Generate training data.
		num_channel_updates = 250 * self.num_channels  # observe the pattern this many times
		training_data, training_labels = self.data_generator.generate_data(num_channel_updates)
		# Train network.
		self.neural_network.get_keras_model().fit(x=training_data, y=training_labels)

		# Generate next observation vector.
		observation_mat, correct_label_vec = self.data_generator.generate_data(1)

		# Can we correctly predict the next idle channel?
		observation_mat = np.reshape(observation_mat, (1, self.history_size*self.num_channels)) # reshape into 1xhistory_size*num_channels
		prediction_vec = self.neural_network.predict(observation_mat)
		predicted_label = np.argmax(prediction_vec)
		self.assertEqual(predicted_label, np.argmax(correct_label_vec))

		# And continue as well?
		for i in range(10*self.num_channels):
			self.data_generator.update()
			observation_mat, correct_label_vec = self.data_generator.generate_data(1)
			observation_mat = np.reshape(observation_mat, (1, self.history_size*self.num_channels)) # reshape into 1xhistory_size*num_channels
			prediction_vec = self.neural_network.predict(observation_mat)
			predicted_label = np.argmax(prediction_vec)
			self.assertEqual(predicted_label, np.argmax(correct_label_vec))


class TestTumuluruMLP(unittest.TestCase):
	def setUp(self):
		self.num_channels = 1
		self.history_size = 4
		self.mean_interarrival_time = 5  # in timeslots
		self.mean_busy_period_length = 5  # also in timeslots
		self.channel = PoissonProcessChannelModel(self.mean_interarrival_time, self.mean_busy_period_length)

		self.neural_network = TumuluruMLP(lookback_length=self.history_size)

	def test_prediction(self):
		num_batches = int(1000 / self.history_size)  # Training data size is 1000 in the paper.

		# Training data is a matrix (batch, input).
		training_data = np.zeros((num_batches, self.history_size))
		training_labels = np.zeros(num_batches)
		for batch in range(num_batches):
			for i in range(self.history_size):
				self.channel.update()
				training_data[batch][i] = self.channel.get_state_vector()[0]
				if i == 0 and batch > 0:
					training_labels[batch-1] = self.channel.get_state_vector()[0]
		self.channel.update()
		training_labels[num_batches-1] = self.channel.get_state_vector()[0]

		self.neural_network.get_keras_model().fit(x=training_data, y=training_labels)


# class TestLSTMNeuralNetwork(unittest.TestCase):
# 	def setUp(self):
# 		self.num_channels = 16
# 		self.switching_prob = 1.0
# 		self.activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# 		self.num_hidden_neurons = [200]
# 		self.learning_rate = 0.005
#
# 	def test_prediction_1_lstm_time_step(self):
# 		self.time_steps = 1
# 		self.history_size = self.time_steps
# 		self.data_generator = DataGenerator(self.num_channels, self.history_size, SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern), update_once=True)
# 		self.neural_network = LSTMNetwork(self.num_channels, self.num_hidden_neurons, self.learning_rate, self.time_steps)
#
# 		# Generate training data.
# 		time_limit = 2*self.num_channels
# 		for time_step in range(time_limit):
# 			observation_vec, label_vec = self.data_generator.read()
# 			observation_vec = np.reshape(observation_vec, (1, self.time_steps, len(observation_vec[0])))
# 			label_vec = np.reshape(label_vec, (1, 1, len(label_vec)))
# 			self.neural_network.get_keras_model().fit(x=observation_vec, y=label_vec)
# 			self.data_generator.update()
#
# 		# Generate validation data.
# 		for i in range(10*self.num_channels):
# 			observation_vec, correct_label_index = self.data_generator.read_last_row()
# 			correct_label_index = np.argmax(correct_label_index)
# 			observation_vec = np.reshape(observation_vec, (1, 1, len(observation_vec))) # reshape into 1batch x 1time_step x 16channels
# 			predicted_vec = self.neural_network.predict(observation_vec)
# 			predicted_label_index = np.argmax(predicted_vec[0][self.time_steps-1])
# 			self.assertEqual(predicted_label_index, correct_label_index)
# 			self.data_generator.update()
#
# 	def test_prediction_2_lstm_time_steps(self):
# 		self.time_steps = 2
# 		self.history_size = self.time_steps
# 		self.data_generator = DataGenerator(self.num_channels, self.history_size, SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern), update_once=True)
# 		self.neural_network = LSTMNetwork(self.num_channels, self.num_hidden_neurons, self.learning_rate, self.time_steps)
#
# 		# Generate training data.
# 		time_limit = 2*self.num_channels
# 		for iteration in range(time_limit):
# 			observation_mat = np.zeros((self.time_steps, self.num_channels))
# 			label_vec = np.zeros((self.time_steps, self.num_channels))
# 			for timeslot in range(self.time_steps):
# 				observation_mat[timeslot], label_vec[timeslot] = self.data_generator.read_last_row()
# 				self.data_generator.update()
# 			observation_mat = np.reshape(observation_mat, (1, self.time_steps, self.num_channels))
# 			label_vec = np.reshape(label_vec, (1, self.time_steps, self.num_channels))
# 			self.neural_network.get_keras_model().fit(x=observation_mat, y=label_vec)
#
# 		# Generate validation data.
# 		for i in range(10*self.num_channels):
# 			observation_mat = np.zeros((self.time_steps, self.num_channels))
# 			correct_label_index = None
# 			for timeslot in range(self.time_steps):
# 				observation_mat[timeslot], correct_label_index = self.data_generator.read_last_row()
# 				self.data_generator.update()
# 			observation_mat = np.reshape(observation_mat, (1, self.time_steps, self.num_channels))
# 			correct_label_index = np.argmax(correct_label_index)
#
# 			prediction_vec = self.neural_network.predict(observation_mat)
# 			predicted_label_index = np.argmax(prediction_vec[0][self.time_steps-1])
# 			self.assertEqual(predicted_label_index, correct_label_index)
#
#
# 	def test_batch_learning(self):
# 		self.time_steps = 1
# 		self.history_size = self.time_steps
# 		self.data_generator = DataGenerator(self.num_channels, self.history_size, SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern), update_once=True)
# 		self.neural_network = LSTMNetwork(self.num_channels, self.num_hidden_neurons, self.learning_rate, self.time_steps)
#
# 		# Generate training data.
# 		time_limit = 100 * self.num_channels
# 		# time_limit x num_channel matrices
# 		observation_mat, labels_mat = self.data_generator.read_next(time_limit)
# 		# reshape to batch x time_step x data
# 		observation_mat = np.reshape(observation_mat, (self.time_steps, time_limit, self.num_channels))
# 		labels_mat = np.reshape(labels_mat, (self.time_steps, time_limit, self.num_channels))
# 		# Train.
# 		self.neural_network.get_keras_model().fit(observation_mat, labels_mat)
#
# 		# Generate validation data.
# 		validation_mat, validation_labels = self.data_generator.read_next(time_limit)
# 		validation_mat = np.reshape(validation_mat, (self.time_steps, time_limit, self.num_channels))
# 		validation_labels = np.reshape(validation_labels, (self.time_steps, time_limit, self.num_channels))
# 		# Validate.
# 		predicted_labels = self.neural_network.predict(validation_mat)
# 		output = self.neural_network.get_keras_model().evaluate(x=validation_mat, y=validation_labels)
# 		loss = output[0]
# 		accuracy = output[1]
# 		self.assertEqual(accuracy, 1.0)
# 		for t in range(time_limit):
# 			self.assertEqual(np.argmax(predicted_labels[t]), np.argmax(validation_labels[t]))