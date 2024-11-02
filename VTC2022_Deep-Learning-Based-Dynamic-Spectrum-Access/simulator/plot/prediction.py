import os
import settings
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import roc_curve
import tikzplotlib

#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/'

from plot.general import setup as setup_plot

def plot_stats_model(filename):
	plt.figure()
	x = range(30)
	y6 = [(9 / 10) ** x for x in x]
	plt.plot(x, y6, label="6 ms", linewidth=4)
	y12 = [(4 / 5) ** x for x in x]
	plt.plot(x, y12, label="12 ms", linewidth=4)
	plt.xlabel('number of DME users')
	plt.ylabel('expected fraction of available slots')
	# plt.xlim([-0.5,20])
	# plt.ylim([80,100.5])
	plt.legend(loc='upper right')
	plt.grid(True)
	ax = plt.gca()
	# ax.set_aspect('equal')
	plt.savefig(f'output/{filename}.pdf')
	plt.savefig(f'output/{filename}.png', dpi=150)
	tikzplotlib.save(f'output/{filename}.tex', strict=True)


#Source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#plot_the_roc
def plot_roc(filename, data, **kwargs):
	setup_plot()

	settings.init()

	fig, ax = plt.subplots(figsize=(2*settings.fig_width, 2*settings.fig_height))
	axins = ax.inset_axes([0.35, 0.124, 0.57, 0.57])

	colors = [
		'xkcd:dark purple', 
		'xkcd:dark cyan', 
		'xkcd:dark olive green', 
		'xkcd:dark peach', 
		'xkcd:gold', 
		'xkcd:dark pink', 
		'xkcd:burnt orange', 
		'xkcd:deep sky blue', 
		'xkcd:dark lime green', 
		'xkcd:chocolate', 
		'xkcd:dark magenta', 
		'xkcd:dark violet', 
		'xkcd:dark blue grey', 
		'xkcd:dark aquamarine', 
		'xkcd:mustard', 
		'xkcd:dark lilac', 
		'xkcd:dark periwinkle'
	]
	for metadata, (labels, predictions) in data:
		if metadata['algorithm'] == 'rnn':
			label = f"n = {metadata['num_users']}"
			i = metadata['num_users'] - 1
			color = colors[i] if i < len(colors) else f"C{i}"
		elif metadata['algorithm'] == 'baseline':
			if metadata['integer']:
				label = r"Baseline $\frac{d}{l} \in \mathbb{N}$"
				color = 'black'
			else:
				label = r"Baseline $\frac{d}{l} \notin \mathbb{N}$"
				color = 'grey'

		# Source: https://stats.stackexchange.com/a/187003
		# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
		tprs = []
		base_fpr = np.linspace(0, 1, 101)

		for i in range(len(labels)):
			fpr, tpr, _ = roc_curve(labels[i], predictions[i])
			#ax.plot(100*fpr, 100*tpr, color=color, linewidth=0.5, **kwargs)

			tpr = np.interp(base_fpr, fpr, tpr)
			tpr = np.concatenate(([0.0], tpr, [1.0]), axis=None)
			tprs.append(tpr)

		base_fpr = np.concatenate(([0.0], base_fpr, [1.0]), axis=None)

		tprs = np.array(tprs)
		mean_tprs = tprs.mean(axis=0)
		error = scipy.stats.sem(tprs, axis=0) * scipy.stats.t.ppf((1 + 0.95)/2, tprs.shape[0]-1)
		tprs_upper = np.minimum(mean_tprs + error, 1)
		tprs_lower = np.maximum(mean_tprs - error, 0)

		if metadata['num_users'] == 1:
			ax.annotate("$n = 1$", xy=(np.interp([90], 100*mean_tprs, 100*base_fpr), 90), xytext=(0, 110), arrowprops=dict(arrowstyle="->"))
		elif metadata['num_users'] == 15:
			ax.annotate("$n = 15$", xy=(np.interp([90], 100*mean_tprs, 100*base_fpr), 90), xytext=(60, 110), arrowprops=dict(arrowstyle="->"))

		for axis in [ax, axins]:
			axis.plot(100*base_fpr, 100*mean_tprs, label=label, color=color, linewidth=1, **kwargs)
			axis.fill_between(100*base_fpr, 100*tprs_lower, 100*tprs_upper, color=color, alpha=0.3)

			axis.plot(np.interp([90], 100*mean_tprs, 100*base_fpr), 90, color='red', marker='o')

	for axis in [ax, axins]:
		axis.hlines(90, 0, 90, linestyle="dashed", color='red')

		axis.set_aspect('equal')
		axis.set_xticks(np.arange(0, 110, 10))
		axis.set_yticks(np.arange(0, 110, 10))
		axis.grid(True)

	ax.set_xlabel('False Positive Rate [\%]')
	ax.set_ylabel('True Positive Rate [\%]')
	box = ax.get_position()
	#ax.set_position([box.x0 + box.width * 0.075, box.y0 + box.height * 0.19, box.width, box.height])
	#ax.legend(loc='lower center', bbox_to_anchor=(0.409, -0.40), ncol=3, fontsize='small', frameon=False)

	axins.set_xlim(-1, 26)
	axins.set_ylim(74, 101)
	rectangle_patch, connector_lines = ax.indicate_inset_zoom(axins, edgecolor='black', linestyle='--')
	for line in connector_lines:
		line.set(linestyle='--')

	plt.tight_layout()

	plt.savefig(f'output/{filename}.pdf')
	plt.savefig(f'output/{filename}.png', dpi=150)
	print(f'Saved graph in output/{filename}.pdf')
	#tikzplotlib.save(f'output/{filename}.tex', strict=True)


def plot_timeseries(filename, labels, predictions):
	setup_plot()

	plt.figure()
	plt.stairs(predictions[0], linewidth=0.1, fill=True, label='predicted pattern')
	plt.stairs(labels[0], linewidth=0.1, label='labels pattern')
	# plt.legend()
	plt.savefig(f'output/{filename}.pdf')
	plt.savefig(f'output/{filename}.png', dpi=150)
	tikzplotlib.save(f'output/{filename}.tex', strict=True)	
