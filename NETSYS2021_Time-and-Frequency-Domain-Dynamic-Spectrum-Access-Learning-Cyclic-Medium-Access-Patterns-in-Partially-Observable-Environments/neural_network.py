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


import tensorflow as tf


class LSTMRNN:
	def loss(self, actual, prediction):
		return tf.math.reduce_max(tf.square(actual * (actual - prediction)), axis=-1)

	def __init__(self, n_channels, n_neurons, learning_rate=.0005, sample_length=1):
		self.n_channels = n_channels
		self.n_neurons = n_neurons
		# We do online learning on single batches...
		self.batch_size = 1
		# ... and this many observations per batch
		self.sample_length = sample_length
		self.learning_rate = learning_rate
		self.model = tf.keras.Sequential([
			tf.keras.layers.LSTM(batch_input_shape=(self.batch_size, self.sample_length, self.n_channels), units=self.n_neurons, activation='tanh', recurrent_activation='sigmoid', return_sequences=False, stateful=True),
			tf.keras.layers.Dense(units=self.n_channels, activation=None, name="output_layer")])
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

