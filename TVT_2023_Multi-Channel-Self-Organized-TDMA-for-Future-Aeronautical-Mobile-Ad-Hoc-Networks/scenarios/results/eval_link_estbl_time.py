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
json_label_num_nodes_vec = 'num_nodes_vec'
json_label_max_num_pp_links = 'max_num_pp_links'
json_label_num_broadcast_nodes_vec = 'num_broadcast_nodes_vec'
json_label_link_estbl_time_mat = 'link_estbl_time_vec'
json_label_broadcast_mac_delay_mat = 'broadcast_mac_delay_mat'


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]


def parse(dir, num_nodes_vec, num_broadcast_nodes_vec, max_num_pp_links, num_reps, json_filename):		
	link_estbl_time_mat = np.zeros((len(num_nodes_vec), len(num_broadcast_nodes_vec), len(max_num_pp_links), num_reps))
	broadcast_mac_delay_mat = np.zeros((len(num_nodes_vec), len(num_broadcast_nodes_vec), len(max_num_pp_links), num_reps))	
	bar_max_i = num_reps * len(num_nodes_vec) * len(num_broadcast_nodes_vec) * len(max_num_pp_links)
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()			
	# for each no of nodes
	for i in range(len(num_nodes_vec)):
		n = num_nodes_vec[i]
		# for each no of broadcast users
		for j in range(len(num_broadcast_nodes_vec)):
			m = num_broadcast_nodes_vec[j]			
			# for each no of PP links
			for k in range(len(max_num_pp_links)):
				l = max_num_pp_links[k]
				# for each repetition
				for rep in range(num_reps):
					try:
						filename = dir + '/n=' + str(n) + ',m=' + str(m) + ',l=' + str(l) + '-#' + str(rep) 
						filename_sca = filename + '.sca.csv'
						filename_vec = filename + '.vec.csv'
						results_sca = pd.read_csv(filename_sca)
						results_vec = pd.read_csv(filename_vec)					

						# get link establishment time
						establishment_time_vec = np.zeros(n)
						broadcast_mac_delay_vec = np.zeros(n)
						for node in range(n):					
							num_links_estbl = int(results_sca[(results_sca.type=='scalar') & (results_sca.module=='NW_LINK_ESTABLISHMENT.txNode[' + str(node) + '].wlan[0].linkLayer') & (results_sca.name=='mcsotdma_statistic_num_pp_links_established:last')].value)
							results_vec_link_estbl_time = None
							if num_links_estbl > 0:														
								results_vec_link_estbl_time = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_pp_link_establishment_time:vector') & (results_vec.module=='NW_LINK_ESTABLISHMENT.txNode[' + str(node) + '].wlan[0].linkLayer')]
								results_vec_link_estbl_time = results_vec_link_estbl_time['vecvalue']
								results_vec_link_estbl_time = [float(s) for s in results_vec_link_estbl_time.values[0].split(' ')]						
							else:
								print('zero links established in ' + filename)

							results_vec_broadcast_mac_delay = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_broadcast_mac_delay:vector') & (results_vec.module=='NW_LINK_ESTABLISHMENT.txNode[' + str(node) + '].wlan[0].linkLayer')]
							results_vec_broadcast_mac_delay = results_vec_broadcast_mac_delay['vecvalue']
							results_vec_broadcast_mac_delay = [float(s) for s in results_vec_broadcast_mac_delay.values[0].split(' ')]

							# compute means
							establishment_time_vec[node] = np.mean(results_vec_link_estbl_time)
							broadcast_mac_delay_vec[node] = np.mean(results_vec_broadcast_mac_delay)
						link_estbl_time_mat[i, j, k, rep] = np.mean(establishment_time_vec)
						broadcast_mac_delay_mat[i, j, k, rep] = np.mean(broadcast_mac_delay_vec)
						
						bar_i += 1
						bar.update(bar_i)
					except FileNotFoundError as err:
						print(err)			
	bar.finish()				

	# Save to JSON.
	json_data = {}
	json_data[json_label_reps] = num_reps
	json_data[json_label_num_nodes_vec] = num_nodes_vec
	json_data[json_label_max_num_pp_links] = max_num_pp_links
	json_data[json_label_num_broadcast_nodes_vec] = num_broadcast_nodes_vec	
	json_data[json_label_link_estbl_time_mat] = link_estbl_time_mat.tolist()	
	json_data[json_label_broadcast_mac_delay_mat] = broadcast_mac_delay_mat.tolist()		
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(json_filename, time_slot_duration, graph_filename_delay):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)
		num_reps = json_data[json_label_reps]
		max_num_pp_links = np.array(json_data[json_label_max_num_pp_links])
		num_nodes_vec = np.array(json_data[json_label_num_nodes_vec])
		num_broadcast_nodes_vec = np.array(json_data[json_label_num_broadcast_nodes_vec])
		link_estbl_time_mat = np.array(json_data[json_label_link_estbl_time_mat])
		broadcast_mac_delay_mat = np.array(json_data[json_label_broadcast_mac_delay_mat])
		# Calculate confidence intervals
		link_estbl_time_means = np.zeros((len(num_nodes_vec), len(num_broadcast_nodes_vec), len(max_num_pp_links)))
		link_estbl_time_err = np.zeros((len(num_nodes_vec), len(num_broadcast_nodes_vec), len(max_num_pp_links)))	
		broadcast_mac_delay_means = np.zeros((len(num_nodes_vec), len(num_broadcast_nodes_vec), len(max_num_pp_links)))
		broadcast_mac_delay_err = np.zeros((len(num_nodes_vec), len(num_broadcast_nodes_vec), len(max_num_pp_links)))	
		for i in range(len(num_nodes_vec)):	
			for j in range(len(num_broadcast_nodes_vec)):
				for k in range(len(max_num_pp_links)):
					link_estbl_time_means[i, j, k], _, time_p = calculate_confidence_interval(link_estbl_time_mat[i, j, k, :], confidence=.95)				
					link_estbl_time_err[i, j, k] = time_p - link_estbl_time_means[i, j, k]
					broadcast_mac_delay_means[i, j, k], _, delay_p = calculate_confidence_interval(broadcast_mac_delay_mat[i, j, k, :], confidence=.95)				
					broadcast_mac_delay_err[i, j, k] = delay_p - broadcast_mac_delay_means[i, j, k]

		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		colors = ['xkcd:teal', 'xkcd:coral', 'xkcd:goldenrod', 'xkcd:maroon', 'xkcd:sea blue']
		# 1st graph: delay
		fig = plt.figure()		
		for k in range(len(max_num_pp_links)):
			line = plt.errorbar(num_broadcast_nodes_vec, link_estbl_time_means[0, :, k]*time_slot_duration, link_estbl_time_err[0, :, k]*time_slot_duration, markersize=2, fmt='o', label='$l=' + str(max_num_pp_links[k]) + '$', zorder=1, color=colors[k])
			plt.plot(num_broadcast_nodes_vec, link_estbl_time_means[0, :, k]*time_slot_duration, linestyle='--', linewidth=.5, color=line[0].get_color(), zorder=1)
		plt.ylabel('Link establishment time [ms]')		
		plt.xlabel('Number of neighbors $n$')		
		if num_broadcast_nodes_vec[-1] == 60:
			xticklabels = ['' for _ in range(0, num_broadcast_nodes_vec[-1] + 5, 5)]
			xticklabels[2] = '10'
			xticklabels[6] = '30'
			xticklabels[-1] = '60'
			plt.xticks(range(0, num_broadcast_nodes_vec[-1] + 5, 5), xticklabels)			
			plt.axhline(link_estbl_time_means[0, 2, 0]*time_slot_duration, linestyle='--', color='k', linewidth=.75, zorder=0)
			plt.axvline(10, linestyle='--', color='k', linewidth=.75, zorder=0)
			plt.axvline(30, linestyle='--', color='k', linewidth=.75, zorder=0)
			plt.axhline(link_estbl_time_means[0, 6, 0]*time_slot_duration, linestyle='--', color='k', linewidth=.75, zorder=0)
			ymax_val = link_estbl_time_means[0, -1, -1]*time_slot_duration
			
			plt.yticks([link_estbl_time_means[0, 2, 0]*time_slot_duration, link_estbl_time_means[0, 6, 0]*time_slot_duration, ymax_val])
		plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height*1.25), forward=False)
		fig.savefig(graph_filename_delay, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delay)    
		plt.close()  


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='link_establishment_time')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')				
	parser.add_argument('--n', type=int, nargs='+', help='Number of user pairs.', default=[1])	
	parser.add_argument('--m', type=int, nargs='+', help='Number of broadcast users.', default=[0])	
	parser.add_argument('--max_num_links', type=int, nargs='+', help='Maximum number of PP links that should be supported.', default=[1])	
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)		
	parser.add_argument('--time_slot_duration', type=int, help='Time slot duration in milliseconds.', default=24)		

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + '_n-' + str(args.n) + '_m-' + str(args.m) + '_l=' + str(args.max_num_links) + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename_delay = "_imgs/" + output_filename_base + "_delay.pdf"	
	if not args.no_parse:		
		parse(args.dir, args.n, args.m, args.max_num_links, args.num_reps, json_filename)
	if not args.no_plot:
		plot(json_filename, args.time_slot_duration, graph_filename_delay) 
    