import math

class Entity():
	def __init__(self, pos_x, pos_y, time=0.0):
		self.set_position(pos_x, pos_y, time)

	def set_position(self, pos_x, pos_y, time=0.0):
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.pos_time = time

	def get_position(self, time=0.0):
		return (self.pos_x, self.pos_y)

	def distance(self, other, time=0.0):
		pos_self_x, pos_self_y = self.get_position(time)
		pos_other_x, pos_other_y = other.get_position(time)

		return math.sqrt((pos_self_x - pos_other_x)**2 + (pos_self_y - pos_other_y)**2)

class Aircraft(Entity):
	def __init__(self, pos_x, pos_y, rot, vel, time=0.0):
		super().__init__(pos_x, pos_y, time)
		self.rot = 0.0
		self.vel = 0.0
		self.set_rotation(rot, time)
		self.set_velocity(vel, time)

	def get_position(self, time=0.0):
		pos_x, pos_y = super().get_position()
		if time > 0.0:
			pos_x = pos_x + (math.cos(self.rot) * self.vel * (time - self.pos_time))
			pos_y = pos_y - (math.sin(self.rot) * self.vel * (time - self.pos_time))
		return (pos_x, pos_y)

	def update_position(self, time=0.0):
		pos_x, pos_y = self.get_position(time)
		self.set_position(pos_x, pos_y, time)
		return (pos_x, pos_y)

	def set_rotation(self, rot, time=0.0):
		self.update_position(time)
		self.rot = math.radians(rot % 360)

	def get_rotation(self, time=0.0):
		return math.degrees(self.rot)

	def set_velocity(self, vel, time=0.0):
		self.update_position(time)
		self.vel = vel

	def get_velocity(self, time=0.0):
		return self.vel

	def set_target_position(self, target_x, target_y, time=0.0, target_time=0.0):
		pos_x, pos_y = self.update_position(time)
		distance = math.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2)
		time_diff = target_time - time

		if distance > 0.0:
			self.rot = math.atan2(-(target_y - pos_y), (target_x - pos_x))

		if time_diff > 0.0:
			self.vel = distance / time_diff

class GroundStation(Entity):
	pass
