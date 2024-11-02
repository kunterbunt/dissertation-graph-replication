import tensorflow.compat.v1 as tf
import progressbar

class LossHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


class AccuracyHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.accuracies = []

	def on_batch_end(self, batch, logs={}):
		self.accuracies.append(logs.get('accuracy'))


class BinaryAccuracyHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.accuracies = []

	def on_batch_end(self, batch, logs={}):
		self.accuracies.append(logs.get('binary_accuracy'))


class PredictionHistory(tf.keras.callbacks.Callback):
	def __init__(self, neural_network, input_vec, num_timeslots, verbose=False):
		self.neural_network = neural_network
		self.input_vec = input_vec
		self.predictions = []
		self.counter = 0		
		self.verbose = verbose
		if self.verbose:
			self.bar = progressbar.ProgressBar(max_value=num_timeslots, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])		

	def on_train_begin(self, logs={}):
		if self.verbose:
			self.bar.start()
		self.predictions = []

	def on_train_end(self, logs={}):
		if self.verbose:
			self.bar.finish()

	def on_batch_end(self, batch, logs={}):
		self.predictions.append(self.neural_network.get_keras_model().predict(x=self.input_vec, batch_size=1, verbose=False))
		if self.verbose:
			self.counter = self.counter + 1
			self.bar.update(self.counter)
