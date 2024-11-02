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


class DMEEnvironment:
	class DMEUser:
		"""
		A DME user is a pair of aircraft and ground station, with configured request and response frequency channels.
		They have some periodicity with which they communicate, and are updated on each time slot.
		"""

		# Channel index that a user returns when it does *not* access any channel in the current time slot.
		NO_CHANNEL_ACCESS = -1

		def __init__(self, request_channel, response_channel, periodicity, offset):
			self.request_channel = request_channel
			self.response_channel = response_channel
			self.periodicity = periodicity
			# Starting offset.
			self.offset = offset
			# Current time.
			self.t = 0

		def reset(self):
			self.t = 0

		def step(self):
			# First request at offset
			if self.t == self.offset:
				channel_access_at = self.request_channel
			# First response at offset+1
			elif self.t == self.offset + 1:
				channel_access_at = self.response_channel
			# Everything after that...
			elif self.t > self.offset + 1:
				# ... at the given periodicity, send requests
				if (self.t - self.offset) % self.periodicity == 0:
					channel_access_at = self.request_channel
				# ... and the response one slot later
				elif (self.t - self.offset) % self.periodicity == 1:
					channel_access_at = self.response_channel
				# ... if neither, then don't access the channel
				else:
					channel_access_at = self.NO_CHANNEL_ACCESS
			else:
				channel_access_at = self.NO_CHANNEL_ACCESS
			self.t += 1
			return channel_access_at

	IDLE_CHANNEL_STATE = 1
	BUSY_CHANNEL_STATE = -1

	def __init__(self, n_channels):
		self.n_channels = n_channels
		self.users = []
		self.state = np.full(self.n_channels, self.IDLE_CHANNEL_STATE)

	def add(self, request_channel, response_channel, periodicity, offset):
		self.users.append(self.DMEUser(request_channel, response_channel, periodicity, offset))

	def reset(self):
		for user in self.users:
			user.reset()

	def observe(self):
		return self.state

	def observe_fully_observable(self):
		return self.observe()

	def step(self, action=None):
		# Default to 1 (idle channel).
		self.state = np.full(self.n_channels, self.IDLE_CHANNEL_STATE)
		for user in self.users:
			channel = user.step()
			if channel != DMEEnvironment.DMEUser.NO_CHANNEL_ACCESS:
				self.state[channel] = self.BUSY_CHANNEL_STATE  # Set to -1 (busy) if a user accesses it.
		return self.observe()


class DMEEnvironmentPartiallyObservable(DMEEnvironment):
	UNKNOWN_CHANNEL_STATE = 0

	def __init__(self, n_channels):
		super().__init__(n_channels)
		self.fully_observable_state = np.copy(self.state)

	# For testing purposes.
	def observe_fully_observable(self):
		return self.fully_observable_state

	def step(self, action):
		# Call fully-observable version.
		self.fully_observable_state = super().step()
		self.state = np.copy(self.fully_observable_state)
		for i in range(self.n_channels):
			# Set all non-selected channels to 0.
			if i != action:
				self.state[i] = self.UNKNOWN_CHANNEL_STATE
		return self.observe()


def get_env_fully_observable(n_channels):
	env = DMEEnvironment(n_channels)
	env.add(request_channel=0, response_channel=1, periodicity=4, offset=0)
	if n_channels >= 5:
		env.add(request_channel=3, response_channel=4, periodicity=3, offset=2)
	if n_channels >= 7:
		env.add(request_channel=5, response_channel=6, periodicity=5, offset=4)
	return env


def get_env_partially_observable(n_channels):
	env = DMEEnvironmentPartiallyObservable(n_channels)
	env.add(request_channel=0, response_channel=1, periodicity=4, offset=0)
	if n_channels >= 5:
		env.add(request_channel=3, response_channel=4, periodicity=3, offset=2)
	if n_channels >= 7:
		env.add(request_channel=5, response_channel=6, periodicity=5, offset=4)
	return env
