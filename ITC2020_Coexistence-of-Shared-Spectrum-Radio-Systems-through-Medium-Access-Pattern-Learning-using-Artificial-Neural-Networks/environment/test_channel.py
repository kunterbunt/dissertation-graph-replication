import unittest
from environment.channel import *
import numpy as np

class TestSequentialAccessChannel(unittest.TestCase):
	def setUp(self):
		self.num_channels = 16
		self.switching_prob = 1.0
		self.activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
		self.channel = SequentialAccessChannelModel(self.num_channels, self.switching_prob, self.activation_pattern)

	def test_idle_channel_initial(self):
		idle_channel = self.channel.get_idle_channel()
		self.assertEqual(idle_channel, self.activation_pattern[0])

	def test_idle_channel_iterative(self):
		for i in range(self.num_channels):
			self.assertEqual(self.channel.get_idle_channel(), self.activation_pattern[i])
			self.channel.update()

	def test_state_vector(self):
		for i in range(self.num_channels):
			state_vec = self.channel.get_state_vector()
			# Sum==1 <=> exactly one '1' in the vector.
			self.assertEqual(np.sum(state_vec), 1)
			# And this '1' should be wherever the activation_pattern tells it to be.
			self.assertEqual(state_vec[self.activation_pattern[i]], 1)
			self.channel.update()

	def test_get_label_from_observation(self):
		for i in range(self.num_channels):
			observation_vec = np.zeros(self.num_channels)
			observation_vec[i] = 1
			label = self.channel.get_label_from_observation(observation_vec)
			self.assertEqual(label, self.activation_pattern[(i+1)%len(self.activation_pattern)])


class TestPoissonProcessChannel(unittest.TestCase):
	def setUp(self):
		self.mean_inter_arrival_time = 5
		self.mean_busy_period_length = 3
		self.channel = PoissonProcessChannelModel(self.mean_inter_arrival_time, self.mean_busy_period_length)

	def test_utilization(self):
		evaluate_for = 10000
		state_vec = np.zeros(evaluate_for)
		for i in range(evaluate_for):
			self.channel.update()
			state_vec[i] = self.channel.get_state_vector()[0]
		difference_observed_and_theoretical_utilization = abs(self.channel.get_utilization() - sum(state_vec)/evaluate_for)
		tolerance = 0.075  # allow up to 7.5% gap
		self.assertLess(difference_observed_and_theoretical_utilization, tolerance)


class TestInteractiveChannel(unittest.TestCase):
	def setUp(self):
		self.aircraft = InteractiveChannelModel.User(x=0, y=0, channel=None)
		self.dme = InteractiveChannelModel.User(x=370000, y=0, channel=None)
		self.channel = InteractiveChannelModel(2, self.aircraft, 10)
		self.dme.set_channel(self.channel)
		self.aircraft.set_channel(self.channel)

	def test_propagation_delay(self):
		distance = self.channel.__euclidean_distance__(self.aircraft, self.dme)
		self.assertEqual(int(distance), 370000)  # 370km
		propagation_delay = self.channel.__get_propagation_delay__(distance)
		self.assertEqual(propagation_delay, 1.23)

	def test_access(self):
		self.assertFalse(self.channel.is_busy(self.channel.get_current_timeslot(), 0))
		self.assertFalse(self.channel.is_busy(self.channel.get_current_timeslot() + 1, 0))
		self.channel.access(0, self.dme)
		self.assertTrue(self.channel.is_busy(self.channel.get_current_timeslot() + 1, 0))


class TestTransitionProbPoissonProcessChannelModel(unittest.TestCase):
	def setUp(self):
		self.p = 0.3
		self.q = 0.4
		self.channel = TransitionProbPoissonProcessChannelModel(self.p, self.q)

	def test_expectation_value(self):
		computed_expectation_vec = self.channel.get_expectation()
		self.assertEqual(computed_expectation_vec[0], 1/self.p)
		self.assertEqual(computed_expectation_vec[1], 1/self.q)

	def test_steady_state(self):
		computed_steady_state_vec = self.channel.get_steady_state()
		self.assertEqual(computed_steady_state_vec[0], self.q/(self.q + self.p))
		self.assertEqual(computed_steady_state_vec[1], self.p/(self.q + self.p))
		# These are probabilities, so they better sum to one.
		self.assertEqual(computed_steady_state_vec[0] + computed_steady_state_vec[1], 1.0)

	def test_operation(self):
		num_timeslots = 100000
		has_been_busy_vec = np.zeros(num_timeslots)
		for i in range(num_timeslots):
			self.channel.update()
			has_been_busy_vec[i] = self.channel.get_state_vector()[0]
		# Steady state gives us the expected fraction of timeslots that are busy.
		expected_busy_slots = self.channel.get_steady_state()[1]
		# This is the observed number of busy timeslots.
		actually_busy_slots = np.sum(has_been_busy_vec) / num_timeslots
		# Which won't be exactly identical.
		difference = abs(expected_busy_slots - actually_busy_slots)
		# So allow for some small tolerance.
		tolerance = 0.01  # 1%
		self.assertLess(difference, tolerance)
