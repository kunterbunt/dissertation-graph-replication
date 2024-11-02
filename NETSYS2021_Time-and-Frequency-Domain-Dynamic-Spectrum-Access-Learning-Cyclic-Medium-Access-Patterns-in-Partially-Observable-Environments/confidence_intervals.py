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

import numpy as np
from scipy import mean
from scipy.stats import sem, t


def columnwise_batch_means(data_mat, split):
	"""
	:param data_mat: Data matrix with one row vector per repetition.
	:param split: How many rows to accumulate into one mean.
	:return: The means after splitting the data vectors into 'split'-many groups. For an example see this function's unittest.
	"""
	num_repetitions = data_mat.shape[0]
	num_data_points = data_mat.shape[1]
	assert (num_repetitions % split == 0 and "Can't split the data this way!")
	num_splits = int(num_repetitions / split)
	batch_means = np.zeros((num_splits, num_data_points))  # One vector of means per split, holding as many mean values as there are data points.
	# Go through each data point...
	for data_point in range(num_data_points):
		# ... and through each split
		for rep in range(0, num_repetitions, split):
			mean = data_mat[rep:rep + split, data_point].mean()  # numpy array indexing start:stop EXCLUDES stop
			batch_means[int(rep / split)][data_point] = mean  # the mean over 'split' many repetitions at this data point's position

	return batch_means


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = mean(data)
	std_dev = sem(data)
	h = std_dev * t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]
