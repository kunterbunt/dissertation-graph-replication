import random as rand
import numpy as np
import scipy.constants as constants
import math
from util.verbose_print import *

class BaseChannelModel:
	IDLE = 1
	BUSY = 0

	def __init__(self, num_channels):
		self.num_channels = num_channels

	def get_num_channels(self):
		return self.num_channels

	def get_state_vector(self):
		state_vector = np.full(self.num_channels, BaseChannelModel.BUSY)
		return state_vector


class SequentialAccessChannelModel(BaseChannelModel):
	def __init__(self, num_channels, switching_prob, activation_pattern):
		BaseChannelModel.__init__(self, num_channels)
		if len(activation_pattern) != num_channels:
			raise ValueError("activation_pattern length doesn't match num_channels.")
		if switching_prob < 0 or switching_prob > 1:
			raise ValueError("switching_prob out of range [0 1].")
		self.switching_prob = switching_prob
		self.activation_pattern = activation_pattern
		self.idle_channel_index = 0

	def update(self):
		# Draw a random number from [0 1].
		random_number = rand.random()
		# Possibly switch to the next idle channel.
		if random_number <= self.switching_prob:
			self.idle_channel_index = (self.idle_channel_index + 1) % self.num_channels

	def get_idle_channel(self):
		return self.activation_pattern[self.idle_channel_index]

	def get_state_vector(self):
		state_vector = BaseChannelModel.get_state_vector(self)
		state_vector[self.activation_pattern[self.idle_channel_index]] = BaseChannelModel.IDLE
		return state_vector

	# For a given 'observation_vec', find the label that corresponds to the next idle channel if switching occurs.
	def get_label_from_observation(self, observation_vec):		
		# Find which channel is currently idle.
		observed_idle_channel_pos = -1
		for i in range(len(observation_vec)):
			if observation_vec[i] == BaseChannelModel.IDLE:
				observed_idle_channel_pos = i  # this index corresponds to the position of the '1' in 'observation_vec'
				break

		# Look for position in activation pattern...
		for i in range(len(self.activation_pattern)):
			if self.activation_pattern[i] == observed_idle_channel_pos:  # this is the corresponding position in 'activation_pattern'
				# ... and return next position
				return self.activation_pattern[(i + 1) % len(self.activation_pattern)]
		raise ValueError("Could not find label for given 'observation_vec'.")


class ErroneousSequentialAccessChannelModel(SequentialAccessChannelModel):
	def __init__(self, num_channels, switching_prob, activation_pattern, sensing_error_prob):
		SequentialAccessChannelModel.__init__(self, num_channels, switching_prob, activation_pattern)
		self.sensing_error_prob = sensing_error_prob
		self.state_vec = SequentialAccessChannelModel.get_state_vector(self)

	def update(self):
		SequentialAccessChannelModel.update(self)
		# Apply bit flipping errors
		self.state_vec = SequentialAccessChannelModel.get_state_vector(self)
		self.state_vec = self.__apply_sensing_errors__(self.state_vec)

	def __apply_sensing_errors__(self, state_vec):
		for i in range(self.num_channels):
			random_number = rand.random()
			if random_number <= self.sensing_error_prob:
				state_vec[i] = BaseChannelModel.BUSY if state_vec[i] == BaseChannelModel.IDLE else BaseChannelModel.IDLE
		return state_vec

	def get_state_vector(self):
		return self.state_vec


class PoissonProcessChannelModel(BaseChannelModel):
	"""
	Given mean traffic inter-arrival times and mean busy period lengths, a single channel can be modeled through a Poisson process.
	"""
	def __init__(self, mean_idle_slots, mean_busy_slots):
		BaseChannelModel.__init__(self, num_channels=1)
		self.mean_idle_slots = mean_idle_slots
		self.mean_busy_slots = mean_busy_slots
		self.remaining_timeslots_until_idle = 0
		self.remaining_timeslots_until_busy = np.random.geometric(1 / self.mean_idle_slots)

	def __is_busy__(self):
		return self.remaining_timeslots_until_idle > 0

	def update(self):
		if self.__is_busy__():  # channel is busy
			# print("channel busy, " + str(self.remaining_timeslots_until_idle) + " -> ", end='')
			self.remaining_timeslots_until_idle = self.remaining_timeslots_until_idle - 1  # decrement counter until it becomes idle again
			# print(str(self.remaining_timeslots_until_idle) + " slots until idle")
		else:  # channel is idle
			# print("channel idle ", end='')
			if self.remaining_timeslots_until_busy == 0:  # goes busy now
				# We have the mean number of slots the channel is busy for, which is the expectation value
				# of the busy period's geometric distribution. E.g. on average the channel is busy for E[X]=3 slots,
				# then the probability of one slot being busy is 1/E[X]=1/3.
				self.remaining_timeslots_until_idle = np.random.geometric(1 / self.mean_busy_slots)
				self.remaining_timeslots_until_busy = np.random.geometric(1 / self.mean_idle_slots)
				# print("and goes busy now for " + str(self.remaining_timeslots_until_idle) + " slots")
			else:  # goes busy in the future
				# print("and goes busy in " + str(self.remaining_timeslots_until_busy) + " -> ", end='')
				self.remaining_timeslots_until_busy = self.remaining_timeslots_until_busy - 1
				# print(str(self.remaining_timeslots_until_busy))

	def get_state_vector(self):
		return [0 if self.__is_busy__() else 1]

	def get_utilization(self):
		return self.mean_busy_slots / (self.mean_busy_slots + self.mean_idle_slots)


class TransitionProbPoissonProcessChannelModel(BaseChannelModel):
	def __init__(self, p, q):
		BaseChannelModel.__init__(self, num_channels=1)
		self.p = p  # Transition probability idle->busy
		self.q = q  # Transition probability busy->idle
		self.steady_state = [q/(p+q), p/(p+q)]  # Probabilities to be in idle / busy states.
		if np.random.random() <= self.steady_state[0]:
			self.current_state = BaseChannelModel.IDLE  # Start in idle state
		else:
			self.current_state = BaseChannelModel.BUSY  # Start in busy state

	def __is_busy__(self):
		return self.current_state == BaseChannelModel.BUSY

	def update(self):
		transition_prob = self.p if self.current_state == BaseChannelModel.IDLE else self.q
		if np.random.random() <= transition_prob:
			self.current_state = BaseChannelModel.IDLE if self.current_state == BaseChannelModel.BUSY else BaseChannelModel.BUSY

	def get_state_vector(self):
		return [BaseChannelModel.BUSY if self.__is_busy__() else BaseChannelModel.IDLE]

	def get_steady_state(self):
		"""
		:return: A vector [P(being in the idle state), P(being in the busy state)]
		"""
		return self.steady_state

	def get_expectation(self):
		"""
		:return: A vector [E(number of idle time slots), E(number of busy time slots)]
		"""
		return [1/self.p, 1/self.q]


# class InteractiveChannelModel(BaseChannelModel):
# 	"""
# 	The state of the radio spectrum from the viewpoint of one user.
# 	"""

# 	class User:
# 		def __init__(self, x,  y, channel):
# 			self.x = x
# 			self.y = y
# 			self.channel = channel

# 		def set_channel(self, channel):
# 			self.channel = channel

# 		def get_position(self):
# 			return (self.x, self.y)

# 	def __init__(self, num_channels, user, num_timeslots):
# 		"""
# 		:param num_channels: Number of orthogonal frequency channels.
# 		:param user: The user this model bases on. Whether a particular timeslot is busy or not is valid for this user.
# 		"""
# 		BaseChannelModel.__init__(self, num_channels)
# 		self.state_matrix = np.zeros((num_timeslots, num_channels))
# 		self.current_timeslot = 0
# 		self.centered_user = user
# 		self.max_timeslots = num_timeslots
# 		self.max_distance = 480*1000  # DME maximum transmission range

# 	def update(self):
# 		"""
# 		Advances to the next time slot.
# 		"""
# 		self.current_timeslot = self.current_timeslot + 1

# 	def access(self, channel_index, user, offset=0.0):
# 		"""
# 		:param channel_index:
# 		:param user:
# 		:param offset: Offset in milliseconds until radio access is triggered.
# 		:return:
# 		"""
# 		assert(0 <= channel_index <= self.num_channels)
# 		# 'user' transmits now
# 		distance = self.__euclidean_distance__(user, self.centered_user)
# 		if distance > self.max_distance:
# 			vprint("\tSignal won't arrive @user due to exceeded maximum distance: " + str(np.round(distance/1000,2)) + "km > " + str(self.max_distance/1000) + "km")
# 			return
# 		propagation_delay = self.__get_propagation_delay__(distance) + offset
# 		# the propagation delay passes until the radio signal arrives at the assigned 'self.centered_user',
# 		propagation_delay = math.floor(propagation_delay)
# 		reception_timeslot = self.current_timeslot + propagation_delay
# 		vprint("\tSignal arrives in " + str(propagation_delay) + "ms [t=" + str(reception_timeslot) + "] on channel " + str(channel_index) + " @user (" + str(distance/1000) + "km).")
# 		# so mark that timeslot as busy.
# 		if reception_timeslot < self.max_timeslots:
# 			self.state_matrix[reception_timeslot, channel_index] = 1

# 	def __get_propagation_delay__(self, distance):
# 		"""
# 		:param distance: In meters.
# 		:return: Propagation delay in milliseconds.
# 		"""
# 		speed_of_light = constants.c  # meters / second
# 		return np.round(distance / speed_of_light * 1000, 2)  # milliseconds

# 	def __euclidean_distance__(self, user1, user2):
# 		x1, y1 = user1.get_position()
# 		x2, y2 = user2.get_position()
# 		return math.sqrt(pow((x1 - x2),2) + pow((y1 - y2), 2))

# 	def is_busy(self, timeslot, channel_index):
# 		return self.state_matrix[timeslot, channel_index]

# 	def get_state_vector(self):
# 		return self.state_matrix[self.current_timeslot]

# 	def get_current_timeslot(self):
# 		return self.current_timeslot