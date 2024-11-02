import simpy

class RadioMedium():
	def __init__(self, simpy_env, signal_propagation_speed, communication_range):
		self.simpy_env = simpy_env
		self.signal_propagation_speed = signal_propagation_speed
		self.communication_range = communication_range

		self.receivers = {}

	def propagation_delay(self, delay, store, signal):
		yield self.simpy_env.timeout(delay)
		store.put(signal)

	def transmit(self, sender, frequency_channel, signal):
		for receiver, store in self.receivers[frequency_channel].items():
			distance = sender.distance(receiver, time=self.simpy_env.now)
			if distance > self.communication_range:
				continue

			delay = distance / self.signal_propagation_speed

			self.simpy_env.process(self.propagation_delay(delay, store, signal))

	def add_receiver(self, receiver, frequency_channel):
		if frequency_channel not in self.receivers:
			self.receivers[frequency_channel] = {}

		self.receivers[frequency_channel][receiver] = simpy.Store(self.simpy_env)

	def receive(self, receiver, frequency_channel):
		return self.receivers[frequency_channel][receiver].get()
