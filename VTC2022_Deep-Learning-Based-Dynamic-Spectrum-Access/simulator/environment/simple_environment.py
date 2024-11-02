# set up a simple environment with a single user
import random

import numpy as np


def get_env(num_users=1, num_obs=1, integer=True):
    slot_duration = 6e-3
    if isinstance(num_users, list) or isinstance(num_users, tuple):
        num_users = random.randint(num_users[0], num_users[1])
    if integer:
        periodicities = np.array([random.randint(11, 33)*slot_duration for _ in range(num_users)])
        offsets = np.array([random.randint(0, 33)*slot_duration + slot_duration/2 for _ in range(num_users)])
    else:
        periodicities = np.array([random.uniform(0.0625, 0.2) for _ in range(num_users)])
        offsets = np.array([random.uniform(0, 0.2) for _ in range(num_users)])
    delay_vec = np.zeros(num_users)
    config = {
        'airborne_users': [
            {
                'pos_x': 0.0,
                'pos_y': 0.0,
                'rot': 0.0,
                'vel': 0.0,
                'request_freq_channel_index': 0,
                'request_periodicity': periodicities[i],
                'start_offset': offsets[i]
            } for i in range(num_users)
        ],
        'ground_stations': [
            {
                'pos_x': 0.0,
                'pos_y': 0.0,
                'request_freq_channel_index': 0,
                'response_freq_channel_index': -1,
                'response_delay': delay_vec[i]
            } for i in range(1)
        ],
        'num_freq_channels': 1,
        'partially_observable': False,
        'ldacs_time_slot_duration': slot_duration,
        'max_simulation_time': 6000,
        'observation_history_length': num_obs
    }
    return config
