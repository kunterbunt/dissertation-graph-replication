import json
import numpy as np
import os


#Source: https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types """
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def save(filename, data):
	if not os.path.exists('output'):
		os.makedirs('output')

	with open(f"output/{filename}.json", "w") as f:
		json.dump(data, f, cls=NumpyEncoder)


def load(filename):
	with open(f"output/{filename}.json", "r") as f:
		data = json.load(f)

	return data
