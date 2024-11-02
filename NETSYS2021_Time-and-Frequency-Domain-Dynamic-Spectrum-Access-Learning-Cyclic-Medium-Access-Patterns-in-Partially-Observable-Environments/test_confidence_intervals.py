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


import unittest

from confidence_intervals import *


class TestNeuralConfidenceIntervals(unittest.TestCase):
	def test_batch_means(self):
		data_mat = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
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

if __name__ == '__main__':
    unittest.main()