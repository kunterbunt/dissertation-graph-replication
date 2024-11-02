import tensorflow.compat.v1 as tf

class NeuralNetwork:
	"""
	Dense neural network that takes one observation per channel as input.
	"""
	def __init__(self, num_channels, num_hidden_neurons, learning_rate):
		self.num_channels = num_channels
		self.num_hidden_neurons = num_hidden_neurons
		self.learning_rate = learning_rate
		self.__construct__()

	def __construct__(self):
		self.model = tf.keras.Sequential()
		# First hidden layer.
		layer_hidden_1 = tf.keras.layers.Dense(input_shape=(self.num_channels,), units=self.num_hidden_neurons, activation=tf.nn.relu)
		self.model.add(layer_hidden_1)
		# Second hidden layer.
		layer_hidden_2 = tf.keras.layers.Dense(units=self.num_hidden_neurons, activation=tf.nn.relu)
		self.model.add(layer_hidden_2)
		# Output layer.
		layer_output = tf.keras.layers.Dense(units=self.num_channels, name="output_layer")
		self.model.add(layer_output)
		# Compile model with Adam optimizer and Mean Squared Error.
		self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate), loss='mse', metrics=['binary_accuracy'])

	def get_keras_model(self):
		return self.model

	# Expects a batch_size x num_channels matrix.
	def predict(self, input):
		prediction_vector = self.model.predict(x=input)
		return prediction_vector


class LookbackNeuralNetwork(NeuralNetwork):
	"""
	Dense neural network that takes one observation per channel **over some time history** as input.
	"""
	def __init__(self, num_channels, num_hidden_neurons, learning_rate, lookback_length, num_hidden_layers=1):
		self.lookback_length = lookback_length
		self.num_hidden_layers = num_hidden_layers
		if len(num_hidden_neurons) != num_hidden_layers:
			raise ValueError(str(len(num_hidden_neurons)) + " values for the hidden neurons were given for " + str(num_hidden_layers) + " hidden layers!")
		NeuralNetwork.__init__(self, num_channels, num_hidden_neurons, learning_rate)

	def __construct__(self):
		self.model = tf.keras.Sequential()
		# First hidden layer.
		first_hidden_layer = tf.keras.layers.Dense(input_shape=(self.num_channels * self.lookback_length,), units=self.num_hidden_neurons[0])
		self.model.add(first_hidden_layer)
		# Hidden layer(s).
		for i in range(1, self.num_hidden_layers):  # first hidden layer already present
			hidden_layer = tf.keras.layers.Dense(units=self.num_hidden_neurons[i], activation=tf.nn.relu)
			self.model.add(hidden_layer)
		# Output layer.
		layer_output = tf.keras.layers.Dense(units=self.num_channels, name="output_layer")
		self.model.add(layer_output)
		# Compile model with Adam optimizer and Mean Squared Error.
		self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate), loss='mse', metrics=['binary_accuracy'])


class TumuluruMLP(LookbackNeuralNetwork):
	"""
	Multi-layer perceptron modeled as presented in the paper 'A Neural Network Based Spectrum Prediction Scheme for Cognitive Radio" by Tumuluru et al. in 2010, DOI: 10.1109/ICC.2010.5502348
	"""
	def __init__(self, lookback_length=4):
		self.momentum = 0.9
		LookbackNeuralNetwork.__init__(self, num_channels=1, num_hidden_neurons=[15, 20], learning_rate=0.2, lookback_length=lookback_length, num_hidden_layers=2)

	def __construct__(self):
		self.model = tf.keras.Sequential()
		# First hidden layer.
		first_hidden_layer = tf.keras.layers.Dense(input_shape=(self.num_channels * self.lookback_length,), units=self.num_hidden_neurons[0])
		self.model.add(first_hidden_layer)
		# Hidden layer(s).
		for i in range(1, self.num_hidden_layers):  # first hidden layer already present
			hidden_layer = tf.keras.layers.Dense(units=self.num_hidden_neurons[i], activation=tf.nn.relu)
			self.model.add(hidden_layer)
		# Output layer.
		layer_output = tf.keras.layers.Dense(units=self.num_channels, name="output_layer")
		self.model.add(layer_output)
		# Compile model with Stochastic Gradient Descent optimizer and Mean Squared Error.
		self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=False), loss='mse', metrics=['binary_accuracy'])


class TumuluruMLPAdam(TumuluruMLP):
	def __construct__(self):
		self.model = tf.keras.Sequential()
		# First hidden layer.
		first_hidden_layer = tf.keras.layers.Dense(input_shape=(self.num_channels * self.lookback_length,), units=self.num_hidden_neurons[0])
		self.model.add(first_hidden_layer)
		# Hidden layer(s).
		for i in range(1, self.num_hidden_layers):  # first hidden layer already present
			hidden_layer = tf.keras.layers.Dense(units=self.num_hidden_neurons[i], activation=tf.nn.sigmoid)
			self.model.add(hidden_layer)
		# Output layer.
		layer_output = tf.keras.layers.Dense(units=self.num_channels, name="output_layer")
		self.model.add(layer_output)
		# Compile model with Adam optimizer and Mean Squared Error.
		self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse', metrics=['binary_accuracy'])


class LSTMNetwork(NeuralNetwork):
	"""
	LSTM Recurrent Neural Network.
	"""
	def __init__(self, num_channels, num_hidden_neurons, learning_rate, time_steps, num_hidden_layers=1, use_softmax=False):
		"""
		:param num_channels: Number of channels to predict on.
		:param num_hidden_neurons: Vector of hidden neurons per layer: first entry is for the LSTM layer, all following for dense layers.
		:param learning_rate:
		:param time_steps:
		:param num_hidden_layers: Number of hidden layers in total. Setting this to one creates just the LSTM layer and the output layer, larger than one adds dense layers inbetween.
		"""
		self.time_steps = time_steps
		self.num_hidden_layers = num_hidden_layers
		self.use_softmax = use_softmax
		if len(num_hidden_neurons) != num_hidden_layers:
			raise ValueError(str(len(num_hidden_neurons)) + " values for the hidden neurons were given for " + str(num_hidden_layers) + " hidden layers!")
		NeuralNetwork.__init__(self, num_channels, num_hidden_neurons, learning_rate)

	def __construct__(self):
		self.model = tf.keras.Sequential()
		# Keras' LSTM networks requires a 3D input_shape: (batch, time_steps, input_dim) (see https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e)
		self.model.add(tf.keras.layers.LSTM(batch_input_shape=(1, self.time_steps, self.num_channels), units=self.num_hidden_neurons[0], activation='tanh', recurrent_activation='sigmoid', return_sequences=False, stateful=True))
		for i in range(1, self.num_hidden_layers):  # first hidden layer already present
			hidden_layer = tf.keras.layers.Dense(units=self.num_hidden_neurons[i], activation=tf.nn.sigmoid)
			self.model.add(hidden_layer)
		if self.use_softmax:
			self.model.add(tf.keras.layers.Dense(units=self.num_channels, name="output_layer", activation=tf.nn.softmax))
		else:
			self.model.add(tf.keras.layers.Dense(units=self.num_channels, name="output_layer"))
		self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate), loss='mse', metrics=['binary_accuracy'])


