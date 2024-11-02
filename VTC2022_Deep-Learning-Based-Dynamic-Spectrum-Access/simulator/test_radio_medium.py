import pytest

import numpy as np
import simpy

from environment.entities import Aircraft, GroundStation
from environment.radio_medium import RadioMedium

precision = 1e-8

def assert_receive(simpy_env, radio_medium, receiver, frequency_channel, expected_signal, expected_time):
	signal = radio_medium.receive(receiver, frequency_channel)
	timeout = simpy_env.timeout(expected_time - simpy_env.now + 0.1)

	result = yield signal | timeout

	if expected_signal != "timeout":
		assert signal in result
		assert result[signal] == expected_signal
		np.testing.assert_allclose(simpy_env.now, expected_time, atol=precision, rtol=0)
	else:
		assert timeout in result


def test_radio_medium():
	simpy_env = simpy.Environment()
	radio_medium = RadioMedium(simpy_env, signal_propagation_speed=1.0, communication_range=0.1)
	aircraft = Aircraft(0.0, 0.0, 0, 0.0, time=simpy_env.now)
	ground_station = GroundStation(0.0, 0.0, time=simpy_env.now)

	radio_medium.add_receiver(ground_station, 0)

	radio_medium.transmit(aircraft, 0, "test")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 0, expected_signal="test", expected_time=simpy_env.now + 0.0))
	simpy_env.run()


def test_radio_medium_propagation():
	simpy_env = simpy.Environment()
	radio_medium = RadioMedium(simpy_env, signal_propagation_speed=1.0, communication_range=0.1)
	aircraft = Aircraft(0.0, 0.0, 0, 0.0, time=simpy_env.now)
	ground_station = GroundStation(0.0, 0.1, time=simpy_env.now)

	radio_medium.add_receiver(ground_station, 0)

	radio_medium.transmit(aircraft, 0, "test")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 0, expected_signal="test", expected_time=simpy_env.now + 0.1))
	simpy_env.run()


def test_radio_medium_out_of_range():
	simpy_env = simpy.Environment()
	radio_medium = RadioMedium(simpy_env, signal_propagation_speed=1.0, communication_range=0.1)
	aircraft = Aircraft(0.0, 0.0, 0, 0.0, time=simpy_env.now)
	ground_station = GroundStation(0.0, 0.2, time=simpy_env.now)

	radio_medium.add_receiver(ground_station, 0)

	radio_medium.transmit(aircraft, 0, "test")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 0, expected_signal="timeout", expected_time=simpy_env.now + 0.2))
	simpy_env.run()


def test_radio_medium_multiple_receivers():
	simpy_env = simpy.Environment()
	radio_medium = RadioMedium(simpy_env, signal_propagation_speed=1.0, communication_range=0.1)
	aircraft = Aircraft(0.0, 0.0, 0, 0.0, time=simpy_env.now)
	ground_station1 = GroundStation(0.0, 0.0, time=simpy_env.now)
	ground_station2 = GroundStation(0.0, 0.1, time=simpy_env.now)
	ground_station3 = GroundStation(0.0, 0.2, time=simpy_env.now)

	radio_medium.add_receiver(ground_station1, 0)
	radio_medium.add_receiver(ground_station2, 0)
	radio_medium.add_receiver(ground_station3, 0)

	radio_medium.transmit(aircraft, 0, "test")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station1, 0, expected_signal="test", expected_time=simpy_env.now + 0.0))
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station2, 0, expected_signal="test", expected_time=simpy_env.now + 0.1))
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station3, 0, expected_signal="timeout", expected_time=simpy_env.now + 0.2))
	simpy_env.run()


def test_radio_medium_mobility():
	simpy_env = simpy.Environment()
	radio_medium = RadioMedium(simpy_env, signal_propagation_speed=1.0, communication_range=0.1)
	aircraft = Aircraft(0.0, 0.0, 0, 0.1, time=simpy_env.now)
	ground_station = GroundStation(0.0, 0.0, time=simpy_env.now)

	radio_medium.add_receiver(ground_station, 0)

	radio_medium.transmit(aircraft, 0, "test1")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 0, expected_signal="test1", expected_time=simpy_env.now + 0.0))
	simpy_env.run(until=1.0)

	radio_medium.transmit(aircraft, 0, "test2")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 0, expected_signal="test2", expected_time=simpy_env.now + 0.1))
	simpy_env.run()


def test_radio_medium_frequency_channels():
	simpy_env = simpy.Environment()
	radio_medium = RadioMedium(simpy_env, signal_propagation_speed=1.0, communication_range=0.1)
	aircraft = Aircraft(0.0, 0.0, 0, 0.0, time=simpy_env.now)
	ground_station = GroundStation(0.0, 0.0, time=simpy_env.now)

	radio_medium.add_receiver(ground_station, 0)
	radio_medium.add_receiver(ground_station, 1)

	radio_medium.transmit(aircraft, 1, "test")
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 0, expected_signal="timeout", expected_time=simpy_env.now + 0.0))
	simpy_env.process(assert_receive(simpy_env, radio_medium, ground_station, 1, expected_signal="test", expected_time=simpy_env.now + 0.0))
	simpy_env.run()
