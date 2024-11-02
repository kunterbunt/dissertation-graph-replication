import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import settings
from datetime import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import scipy.stats
import os
import progressbar


json_label_reps = 'num_reps'
json_label_channel_error_vec = 'channel_error_vec'
json_label_link_estbl_time_mat = 'link_estbl_time_vec'
json_label_broadcast_mac_delay_mat = 'broadcast_mac_delay_mat'


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]


def parse(dir, channel_error_vec, num_reps, json_filename):		
	link_estbl_time_mat = np.zeros((len(channel_error_vec), num_reps))
	broadcast_mac_delay_mat = np.zeros((len(channel_error_vec), num_reps))	
	bar_max_i = num_reps * len(channel_error_vec)
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()			
	# for each channel error
	for i in range(len(channel_error_vec)):
		e = channel_error_vec[i]		
		# for each repetition
		for rep in range(num_reps):
			try:
				filename = dir + '/e=' + str(e) + '-#' + str(rep) if e > 0.0 else dir + '/e=0-#' + str(rep)
				filename_sca = filename + '.sca.csv'				
				results_sca = pd.read_csv(filename_sca)			

				# get mean link establishment time												
				link_estbl_time_mat[i, rep] = results_sca[(results_sca.type=='scalar') & (results_sca.name=='mcsotdma_statistic_pp_link_establishment_time:mean') & (results_sca.module=='NW_LINK_ESTABLISHMENT.txNode[0].wlan[0].linkLayer')].value				
				
				bar_i += 1
				bar.update(bar_i)
			except FileNotFoundError as err:
				print(err)			
	bar.finish()				

	# Save to JSON.
	json_data = {}
	json_data[json_label_reps] = num_reps
	json_data[json_label_channel_error_vec] = channel_error_vec	
	json_data[json_label_link_estbl_time_mat] = link_estbl_time_mat.tolist()	
	json_data[json_label_broadcast_mac_delay_mat] = broadcast_mac_delay_mat.tolist()		
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(json_filename, time_slot_duration, graph_filename_delay, graph_filename_delay_dist):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)
		num_reps = json_data[json_label_reps]
		channel_error_vec = np.array(json_data[json_label_channel_error_vec])		
		link_estbl_time_mat = np.array(json_data[json_label_link_estbl_time_mat])
		broadcast_mac_delay_mat = np.array(json_data[json_label_broadcast_mac_delay_mat])
		# Calculate confidence intervals
		link_estbl_time_means = np.zeros(len(channel_error_vec))
		link_estbl_time_err = np.zeros(len(channel_error_vec))
		broadcast_mac_delay_means = np.zeros(len(channel_error_vec))
		broadcast_mac_delay_err = np.zeros(len(channel_error_vec))
		for i in range(len(channel_error_vec)):				
			link_estbl_time_means[i], _, time_p = calculate_confidence_interval(link_estbl_time_mat[i, :], confidence=.95)				
			link_estbl_time_err[i] = time_p - link_estbl_time_means[i]
			broadcast_mac_delay_means[i], _, delay_p = calculate_confidence_interval(broadcast_mac_delay_mat[i, :], confidence=.95)				
			broadcast_mac_delay_err[i] = delay_p - broadcast_mac_delay_means[i]

		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		# 1st graph: average delay
		fig = plt.figure()		
		line = plt.errorbar(channel_error_vec, link_estbl_time_means[:]*time_slot_duration, link_estbl_time_err[:]*time_slot_duration, alpha=0.75, fmt='o')
		plt.plot(channel_error_vec, link_estbl_time_means[:]*time_slot_duration, linestyle='--', linewidth=.5, color=line[0].get_color(), alpha=.75)
		plt.ylabel('Link establishment time [ms]')		
		plt.xlabel('Channel error $e$')
		# plt.legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.35), ncol=3)		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_delay, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delay)    
		plt.close()  

		# 2nd graph: delay distribution		
		fig = plt.figure()
		n_bins = 50
		# for i in range(0, len(channel_error_vec), 2):
			# plt.hist(link_estbl_time_mat[0,:]*time_slot_duration, n_bins, density=True, histtype='step', cumulative=-1, label='$e=' + str(channel_error_vec[i]) + '$')
		plt.hist(link_estbl_time_mat[0,:]*time_slot_duration, n_bins, density=True, histtype='step', cumulative=-1, label='$e=' + str(channel_error_vec[0]) + '$')
		plt.hist(link_estbl_time_mat[5,:]*time_slot_duration, n_bins, density=True, histtype='step', cumulative=-1, label='$e=' + str(channel_error_vec[5]) + '$')
		plt.hist(link_estbl_time_mat[10,:]*time_slot_duration, n_bins, density=True, histtype='step', cumulative=-1, label='$e=' + str(channel_error_vec[10]) + '$')
		plt.xlabel('Link establishment time $x$ [ms]')
		plt.ylabel('$P(X>x)$')
		plt.legend()
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_delay_dist, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delay_dist)    
		plt.close()  



if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='link_establishment_time')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')				
	parser.add_argument('--e', type=float, nargs='+', help='Channel errors.', default=[0.0])		
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)	
	parser.add_argument('--time_slot_duration', type=int, help='Time slot duration in milliseconds.', default=24)		

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + '_e-' + str(args.e) + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename_delay = "_imgs/" + output_filename_base + "_avg_delay.pdf"	
	graph_filename_delay_dist = "_imgs/" + output_filename_base + "_delay_dist.pdf"	
	if not args.no_parse:		
		parse(args.dir, args.e, args.num_reps, json_filename)
	if not args.no_plot:
		plot(json_filename, args.time_slot_duration, graph_filename_delay, graph_filename_delay_dist) 
    