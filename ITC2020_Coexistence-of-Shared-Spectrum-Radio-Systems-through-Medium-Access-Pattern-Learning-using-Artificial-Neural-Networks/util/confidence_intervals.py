from scipy.stats import sem, t
from scipy import mean
import numpy as np


def columnwise_batch_means(data_mat, split):
	"""

	:param data_mat: Data matrix with one row vector per repetition.
	:param split: How many rows to accumulate into one mean.
	:return: The means after splitting the data vectors into 'split'-many groups. For an example see this function's unittest.
	"""
	num_repetitions = data_mat.shape[0]
	num_data_points = data_mat.shape[1]
	assert(num_repetitions % split == 0 and "Can't split the data this way!")
	num_splits = int(num_repetitions / split)
	batch_means = np.zeros((num_splits, num_data_points))  # One vector of means per split, holding as many mean values as there are data points.
	# Go through each data point...
	for data_point in range(num_data_points):
		# ... and through each split
		for rep in range(0, num_repetitions, split):
			mean = data_mat[rep:rep+split, data_point].mean()  # numpy array indexing start:stop EXCLUDES stop
			batch_means[int(rep/split)][data_point] = mean  # the mean over 'split' many repetitions at this data point's position

	return batch_means


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = mean(data)
	std_dev = sem(data)
	h = std_dev * t.ppf((1 + confidence) / 2, n - 1)
	return [m, m-h, m+h]
