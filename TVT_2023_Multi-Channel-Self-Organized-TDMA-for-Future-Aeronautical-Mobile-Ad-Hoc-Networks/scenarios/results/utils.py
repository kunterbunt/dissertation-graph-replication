import numpy as np
import scipy


def calculate_confidence_interval(data, confidence=.95):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]