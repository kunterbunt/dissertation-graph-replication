import unittest
from environment.data_generator import *


class TestDataGenerator(unittest.TestCase):
	def setUp(self):
		self.num_channels = 16
		self.num_samples = 16
		self.switching_prob = 1.0
		self.activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
		self.data_generator = DataGenerator(self.num_channels, self.num_samples, SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern))

	# Make sure that the initial state throws an exception as it is invalid.
	def test_read_once_initial(self):
		with self.assertRaises(ValueError):
			self.data_generator.read_last_row()

	# Make sure that the initial state throws an exception as it is invalid.
	def test_read_initial(self):
		with self.assertRaises(ValueError):
			self.data_generator.read()

	# Make sure that the updated state can be read with a sensible output.
	def test_read_once_updated(self):
		self.data_generator.update()
		observation_vec, label_vec = self.data_generator.read_last_row()
		label = np.argmax(label_vec)
		# We expect a '1' wherever the activation_pattern puts the initial one.
		expected_vec = np.zeros(len(self.activation_pattern))
		expected_vec[self.activation_pattern[0]] = 1
		# And as a label the next position according to the activation_pattern is expected.
		expected_label = self.activation_pattern[1]

		# Make sure the vectors are equal.
		for i in range(len(observation_vec)):
			self.assertEqual(observation_vec[i], expected_vec[i])
		# And the label, too.
		self.assertEqual(label, expected_label)

	# Make sure that matrix and vector sizes are as expected.
	def test_sizes(self):
		self.data_generator.update()
		observation_mat, label_vec = self.data_generator.read()
		num_rows = len(observation_mat)
		num_cols = len(observation_mat[0])

		# Each row is one item in history, each column corresponds to a frequency channel.
		self.assertEqual(num_rows, self.num_samples)
		self.assertEqual(num_cols, self.num_channels)
		# label_vec is a one-hot-encoded vector with num_channels many items.
		self.assertEqual(len(label_vec), self.num_channels)

	def test_read_next(self):
		num_observations = self.num_channels + 1
		observation_mat, label_mat = self.data_generator.read_next(num_observations)
		self.assertEqual(len(observation_mat), num_observations)
		for i in range(num_observations):
			self.assertEqual(len(observation_mat[i]), self.num_channels)
			self.assertEqual(observation_mat[i][self.activation_pattern[i % self.num_channels]], 1)
			self.assertEqual(np.argmax(label_mat[i]), self.activation_pattern[(i+1) % self.num_channels])

