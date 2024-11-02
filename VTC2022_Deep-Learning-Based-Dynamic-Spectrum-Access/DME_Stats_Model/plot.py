import os
import numpy as np
import matplotlib.pyplot as plt
import settings


if __name__ == "__main__":
	# read from .csv
	filenames_model = ['data/expectation_mats_model_5ppps.csv', 'data/expectation_mats_model_16ppps.csv']
	filenames_sims = ['data/expectation_mats_sim_5ppps.csv', 'data/expectation_mats_model_16ppps.csv']
	ppps_vec = [5, 16]
	print('reading model and simulation data. If these files do not exist, please run resource_opportunity_{8,16}ppps.m using Matlab!')
	for j in range(len(filenames_model)):
		expectation_mats_model = np.loadtxt(open(filenames_model[j]), delimiter=',')
		expectation_mats_sim = np.loadtxt(open(filenames_sims[j]), delimiter=',')	
		x = range(0, 51)
		n_durations = [6, 12, 24]
		# plot
		graph_filename = 'imgs/dme_stats_model_expectation_' + str(ppps_vec[j]) + 'ppps.pdf'
		if not os.path.exists('imgs'):
			os.makedirs('imgs')
		settings.init()
		settings.set_params()
		fig = plt.figure()
		markers = ['v', '*', 'X']
		# model	
		colors = [
			'xkcd:teal',
			'xkcd:coral',
			'xkcd:goldenrod'
		]
		for i in range(len(n_durations)):
			line = plt.plot(x, expectation_mats_model[:,i]*100, label='model' if i==0 else None, color=colors[i], alpha=1, marker=None, markersize=4, linewidth=1, linestyle='--')
		# simulation
		for i in range(len(n_durations)):
			plt.plot(x, expectation_mats_sim[:,i]*100, label='simulation' if i==0 else None, marker=None, linestyle='-', color=colors[i], alpha=.5, markersize=4, linewidth=1)
		if j==0:
			plt.text(x=13.5, y=20, s='$24\,$ms', fontsize=7)
			plt.text(x=20, y=31, s='$12\,$ms', fontsize=7)
			plt.text(x=28, y=45, s='$6\,$ms', fontsize=7)
		else:			
			plt.text(x=-2.5, y=5, s='$24\,$ms', fontsize=7)
			plt.text(x=9, y=15, s='$12\,$ms', fontsize=7)
			plt.text(x=15, y=24, s='$6\,$ms', fontsize=7)
		plt.xticks(range(0, 51, 10))
		plt.xlabel('no. of DME users $n$')
		plt.ylabel('expectedly idle time slots [\%{}]')
		legend = plt.legend(framealpha=0.0)	
		# Change the legend line colors to black
		for line in legend.get_lines():
			line.set_color('black')
		fig.set_size_inches((settings.fig_width*1.1, settings.fig_height*1.1), forward=False)
		plt.tight_layout()
		fig.savefig(graph_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename)
		plt.close()