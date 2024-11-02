import unittest
from util.confidence_intervals import *

class TestNeuralConfidenceIntervals(unittest.TestCase):
	def test_batch_means(self):
		data_mat = np.array([[1,2], [3,4], [5,6], [7,8]])
		split = 2
		batch_means = columnwise_batch_means(data_mat, split)
		# We have a matrix
		# 1 2
		# 3 4
		# 5 6
		# 7 8
		# where rows are repetitions, so rep1 has data points [1 2] in its column
		# and we want to split every split=2 repetitions into one batch-mean for each data point (column)
		# thus we expect the means
		# mean(1,3) = 2
		# mean(2,4) = 3
		# this is the means of the first two repetitions
		# mean(5,7) = 6
		# mean(6,8) = 7
		# this is the means of the second two repetitions
		# thus the final batch-mean-matrix should be
		# 2 3
		# 6 7
		self.assertEqual(len(batch_means), 2)
		self.assertEqual(len(batch_means[0]), 2)
		self.assertEqual(batch_means[0, 0], 2)
		self.assertEqual(batch_means[0, 1], 3)
		self.assertEqual(batch_means[1, 0], 6)
		self.assertEqual(batch_means[1, 1], 7)
