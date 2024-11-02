import numpy as np
from environment.channel import *
import random as rand
import unittest


class DataGenerator:
	def __init__(self, num_channels, history_size, channel, update_once=False):
		"""
		:param num_channels:
		:param history_size: The observation matrix will hold this many rows, where each column indicates whether the respective channel is idle.
		:param channel:
		:param update_once:
		"""
		self.channel = channel
		self.history_size = history_size
		self.num_channels = num_channels
		# Initializes the observation matrix with history_size many rows, where each column indicates whether the respective channel is idle.
		self.observation_mat = np.zeros((self.history_size, self.num_channels)) # m history rows, n channel columns
		self.current_label_vec = np.zeros(self.num_channels)
		if update_once:
			self.update()


	def update(self):
		"""
		Moves all rows up by one, then generates a new last row.
		The last row of the observation matrix is one step into the past, while the label holds the currently true state of the channels.
		"""
		num_rows = len(self.observation_mat)
		for i in range(1, num_rows):
			self.observation_mat[i-1] = self.observation_mat[i]
		self.observation_mat[num_rows-1] = self.channel.get_state_vector()  # Save channel states as the last observation.
		self.channel.update()  # Update.
		self.current_label_vec = self.channel.get_state_vector()  # The label is the currently true state of the channels after updating.
		return self.read()


	def read(self):
		"""
		:return: The entire observation matrix as well as the current label vector.
		"""
		if np.sum(self.current_label_vec) == 0:
			raise ValueError("Attempt to read from invalid state (label vector sums to zero).")
		return self.observation_mat, self.current_label_vec


	def read_last_row(self):
		"""
		:return: The last observation (the last row of the observation matrix).
		"""
		observation_mat, label_vec = self.read()
		observation_vec = observation_mat[self.history_size - 1].copy()
		return observation_vec, label_vec


	def read_next(self, n):
		"""
		:param n:
		:return: The next 'n' observation vectors and corresponding label vectors.
		"""
		# (n x channels)-matrix
		observation_mat = np.ones((n, self.num_channels))
		label_mat = np.ones((n, self.num_channels))
		for timeslot in range(n):
			self.update()
			observation_mat[timeslot], label_mat[timeslot] = self.read_last_row()
		return observation_mat, label_mat


	def generate_data(self, num_channel_updates):
		"""
		The observation matrix is updated num_channel_updates times (can be used for batch-training).
		Each time, the observation_matrix is flattened into a vector and appended to training_data,
		which therefore contains num_channel_updates rows of history_size*num_channels columns (the flattened matrix).

		:param num_channel_updates:
		:return:
		"""
		training_data = np.zeros((num_channel_updates, self.history_size * self.num_channels))
		training_labels = np.zeros((num_channel_updates, self.num_channels))
		for i in range(num_channel_updates):
			observation_matrix, label_vec = self.read()
			training_data[i] = observation_matrix.flatten()
			training_labels[i] = label_vec
			self.update()
		return training_data, training_labels
