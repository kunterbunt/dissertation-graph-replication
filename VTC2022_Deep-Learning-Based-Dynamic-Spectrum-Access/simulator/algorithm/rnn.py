import random

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from environment.dme_environment import *
from environment.simple_environment import get_env


def get_label(observation):
    if observation[-1] == DMEEnvironmentBase.BUSY:
        return np.array([1])
    elif observation[-1] == DMEEnvironmentBase.IDLE:
        return np.array([0])
    elif observation[-1] == DMEEnvironmentBase.NOT_SENSED:
        return np.array([0.5])


def generate_batch(num_users, num_obs, integer, batch_size=1):
    envs = [DMEEnvironment(get_env(num_users, num_obs, integer)) for _ in range(batch_size)]
    old_obs = [env.reset() for env in envs]
    a = 0  # it should be possible to call the base environment without an action instead of doing this
    for _ in range(num_obs):
        old_obs, _, _, _, _ = zip(*[env.step(a) for env in envs])  # we want an actual observation and not the arbitrary 0 from the reset function
    while True:
        new_obs, _, is_timeout, _, _ = zip(*[env.step(a) for env in envs])
        y = [get_label(obs) for obs in new_obs]
        # input shape: (batch size, "sliding window" size, # channels?)
        # output shape: (batch size, # of classes (always 2))
        yield np.reshape(old_obs, (batch_size, num_obs, 1)), np.reshape(y, (batch_size, 1))
        old_obs = new_obs
        if any(is_timeout):
            print("Maximum Simulation Time has been reached!")
            break


def create_model(config, batch_size=1):
    model = keras.Sequential()

    model.add(keras.Input(shape=(config["num_obs"], 1), batch_size=batch_size))

    for i in range(config["pre_lstm_layers"]):
        model.add(keras.layers.LSTM(units=config["pre_lstm_neurons_per_layer"], activation=config["lstm_activation"], recurrent_activation=config["lstm_recurrent_activation"], return_sequences=True, stateful=False))

    if config["pre_lstm_layers"] > 0:
        model.add(keras.layers.Reshape((1, config["num_obs"] * config["pre_lstm_neurons_per_layer"])))
    else:
        model.add(keras.layers.Reshape((1, config["num_obs"])))

    for i in range(config["post_lstm_layers"]):
        model.add(keras.layers.LSTM(units=config["post_lstm_neurons_per_layer"], activation=config["lstm_activation"], recurrent_activation=config["lstm_recurrent_activation"], return_sequences=(False if i+1 == config["post_lstm_layers"] else True), stateful=True))

    if config["post_lstm_layers"] == 0:
        model.add(keras.layers.Flatten())

    for i in range(config["dense_layers"]):
        model.add(keras.layers.Dense(units=config["dense_neurons_per_layer"], activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='sigmoid', name="output_layer"))

    model.summary()

    # model.compile(optimizer="adam", loss="binary_crossentropy")
    opt = keras.optimizers.Adam(learning_rate=config["lr"])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy", keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC() ,tfa.metrics.F1Score(num_classes=1, threshold=0.5)])
    # find out: are binary crossentropy and categorical crossentropy equivalent, if one uses a single output for binary
    # cross entropy?

    return model


def train(model, config, batch_size=1):
    model.reset_states()
    history = model.fit(x=generate_batch(config["num_users"], config["num_obs"], config["integer"], batch_size), steps_per_epoch=config["steps_per_epoch"], epochs=1)
    return history


def evaluate(model, config, steps, batch_size=1):
    model.reset_states()
    history = model.evaluate(x=generate_batch(config["num_users"], config["num_obs"], config["integer"], batch_size), steps=steps)
    return history


def predict_single(model, opt, loss, x, y_true, train=True):
    # Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example
    if train:
        with tf.GradientTape() as tape:
            y_pred = model(x, training=False)
            gradients = tape.gradient(loss(y_true, y_pred), model.trainable_weights)
            opt.apply_gradients(zip(gradients, model.trainable_weights))
    else:
        y_pred = model(x, training=False)

    return y_pred


def predict(model, config, steps, warmup_steps=0, batch_size=1, train=True):
    model.reset_states()
    opt = keras.optimizers.Adam(learning_rate=config["lr"])
    loss = tf.keras.losses.BinaryCrossentropy()
    generator = generate_batch(config["num_users"], config["num_obs"], config["integer"], batch_size)

    labels_pattern = []
    predicted_pattern = []
    for i in range(warmup_steps):
        if(i % 100 == 0):
            print(f"{config['num_users']} users - Evaluation warmup step {i}")
        model(next(generator)[0])
        val_data = next(generator)
        predict_single(model, opt, loss, val_data[0], val_data[1], train=train).numpy()
    for i in range(steps):
        if(i % 100 == 0):
            print(f"{config['num_users']} users - Evaluation step {i}")
        val_data = next(generator)
        labels_pattern.append(val_data[1])
        predicted_pattern.append(predict_single(model, opt, loss, val_data[0], val_data[1], train=train).numpy())

    labels_pattern = [[labels_pattern[step][run][0] for step in range(steps)] for run in range(batch_size)]
    predicted_pattern = [[predicted_pattern[step][run][0] for step in range(steps)] for run in range(batch_size)]

    return labels_pattern, predicted_pattern
