import pytest

import math
import numpy as np

from environment.entities import *

precision = 1e-8


def test_position():
	entity = Entity(0.3, 0.4)
	np.testing.assert_allclose(entity.get_position(), (0.3, 0.4), atol=precision, rtol=0)

	entity.set_position(0.5, 0.6)
	np.testing.assert_allclose(entity.get_position(), (0.5, 0.6), atol=precision, rtol=0)


def test_distance_x():
	entity1 = Entity(0.0, 0.5)
	entity2 = Entity(1.0, 0.5)

	np.testing.assert_allclose(entity1.distance(entity2), 1.0, atol=precision, rtol=0)


def test_distance_y():
	entity1 = Entity(0.5, 0.0)
	entity2 = Entity(0.5, 1.0)

	np.testing.assert_allclose(entity1.distance(entity2), 1.0, atol=precision, rtol=0)


def test_distance_diag():
	entity1 = Entity(0.0, 0.0)
	entity2 = Entity(1.0, 1.0)

	np.testing.assert_allclose(entity1.distance(entity2), math.sqrt(2.0), atol=precision, rtol=0)


def test_aircraft_rotation():
	aircraft = Aircraft(0.0, 0.0, 90, 1.0, time=0.0)
	np.testing.assert_allclose(aircraft.get_rotation(time=0.0), 90, atol=precision, rtol=0)

	aircraft.set_rotation(45, time=1.0)
	np.testing.assert_allclose(aircraft.get_rotation(time=1.0), 45, atol=precision, rtol=0)

	aircraft.set_rotation(-90, time=2.0)
	np.testing.assert_allclose(aircraft.get_rotation(time=2.0), 270, atol=precision, rtol=0)


def test_aircraft_velocity():
	aircraft = Aircraft(0.0, 0.0, 0, 1.0, time=0.0)
	np.testing.assert_allclose(aircraft.get_velocity(time=0.0), 1.0, atol=precision, rtol=0)

	aircraft.set_velocity(0.5, time=1.0)
	np.testing.assert_allclose(aircraft.get_velocity(time=1.0), 0.5, atol=precision, rtol=0)

	aircraft.set_velocity(-0.5, time=2.0)
	np.testing.assert_allclose(aircraft.get_velocity(time=2.0), -0.5, atol=precision, rtol=0)


def test_aircraft_mobility():
	aircraft = Aircraft(0.0, 0.0, 0, 1.0, time=0.0)

	# Move to (1.0, 0.0)
	np.testing.assert_allclose(aircraft.get_position(time=0.5), (0.5, 0.0), atol=precision, rtol=0)
	np.testing.assert_allclose(aircraft.get_position(time=1.0), (1.0, 0.0), atol=precision, rtol=0)

	# Rotate and move to (1.0, 1.0)
	aircraft.set_rotation(-90, time=1.0)
	np.testing.assert_allclose(aircraft.get_position(time=1.5), (1.0, 0.5), atol=precision, rtol=0)
	np.testing.assert_allclose(aircraft.get_position(time=2.0), (1.0, 1.0), atol=precision, rtol=0)

	# Rotate and move to (0.0, 1.0)
	aircraft.set_rotation(180, time=2.0)
	np.testing.assert_allclose(aircraft.get_position(time=2.5), (0.5, 1.0), atol=precision, rtol=0)
	np.testing.assert_allclose(aircraft.get_position(time=3.0), (0.0, 1.0), atol=precision, rtol=0)

	# Rotate and move to (0.0, 0.0) while halving velocity on the way
	aircraft.set_rotation(90, time=3.0)
	np.testing.assert_allclose(aircraft.get_position(time=3.5), (0.0, 0.5), atol=precision, rtol=0)
	aircraft.set_velocity(0.5, time=3.5)
	np.testing.assert_allclose(aircraft.get_position(time=4.5), (0.0, 0.0), atol=precision, rtol=0)


def test_aircraft_mobility_target():
	aircraft = Aircraft(0.0, 0.0, 0, 1.0, time=0.0)

	# Move to (1.0, 1.0)
	aircraft.set_target_position(1.0, 1.0, time=0.0, target_time=1.0)
	np.testing.assert_allclose(aircraft.get_position(time=0.5), (0.5, 0.5), atol=precision, rtol=0)
	np.testing.assert_allclose(aircraft.get_position(time=1.0), (1.0, 1.0), atol=precision, rtol=0)

	# Move to (0.0, 0.0), but change direction on the way towards (0.0, 1.0)
	aircraft.set_target_position(0.0, 0.0, time=1.0, target_time=2.0)
	np.testing.assert_allclose(aircraft.get_position(time=1.5), (0.5, 0.5), atol=precision, rtol=0)
	aircraft.set_rotation(225, time=1.5)
	np.testing.assert_allclose(aircraft.get_position(time=2.0), (0.0, 1.0), atol=precision, rtol=0)

	# Move to (0.0, 0.0) while halving velocity on the way
	aircraft.set_target_position(0.0, 0.0, time=2.0, target_time=3.0)
	np.testing.assert_allclose(aircraft.get_position(time=2.5), (0.0, 0.5), atol=precision, rtol=0)
	aircraft.set_velocity(0.5, time=2.5)
	np.testing.assert_allclose(aircraft.get_position(time=3.5), (0.0, 0.0), atol=precision, rtol=0)


def test_aircraft_mobility_distance():
	aircraft1 = Aircraft(0.0, 0.0, 0, 1.0, time=0.0)
	aircraft2 = Aircraft(0.0, 0.0, -90, 1.0, time=0.0)

	np.testing.assert_allclose(aircraft1.distance(aircraft2, time=0.0), 0.0, atol=precision, rtol=0)
	np.testing.assert_allclose(aircraft1.distance(aircraft2, time=0.5), math.sqrt(0.5), atol=precision, rtol=0)
	np.testing.assert_allclose(aircraft1.distance(aircraft2, time=1.0), math.sqrt(2.0), atol=precision, rtol=0)
