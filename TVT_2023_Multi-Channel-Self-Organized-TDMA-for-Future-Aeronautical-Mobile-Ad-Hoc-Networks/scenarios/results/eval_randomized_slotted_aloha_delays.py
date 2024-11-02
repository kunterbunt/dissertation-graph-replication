import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import scipy.stats
import os
import progressbar
import settings


json_label_reps = 'num_reps'
json_label_n_users = 'n_users'
json_label_broadcast_delay_means = 'broadcast_mac_delay_means'
json_label_broadcast_delay_err = 'broadcast_mac_delay_err'


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]


def parse(dir, num_users_vec, num_reps, json_filename):		
	delay_means_mat = np.zeros((len(num_users_vec), num_reps))	
	delay_err_mat = np.zeros((len(num_users_vec), num_reps))	
	bar_max_i = len(num_users_vec)*num_reps
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()	
	# for each number of transmitters
	for i in range(len(num_users_vec)):			
		n = num_users_vec[i]				
		# for each repetition
		for rep in range(num_reps):
			try:				
				filename = dir + '/n=' + str(n) + '-#' + str(rep) + '.vec.csv'
				results = pd.read_csv(filename)				

				# collect all delay values per user		
				delay_means = np.zeros(n)
				delay_err = np.zeros(n)		
				for j in range(n):
					user_results = results[(results.type=='vector') & (results.name=='mcsotdma_statistic_broadcast_mac_delay:vector') & (results.module=='NW_TX_RX.txNodes[' + str(j) + '].wlan[0].linkLayer')]					
					delay_vec = [float(s) for s in user_results['vecvalue'].values[0].split(' ')]					
					# compute confidence intervals					
					delay_means[j], _, plus = calculate_confidence_interval(delay_vec, confidence=.95)										
					delay_err[j] = plus - delay_means[j]
				delay_means_mat[i, rep] = np.mean(delay_means)
				delay_err_mat[i, rep] = np.mean(delay_err)
				bar_i += 1
				bar.update(bar_i)
			except FileNotFoundError as err:
				print(err)			
	bar.finish()		

	# Save to JSON.
	json_data = {}
	json_data[json_label_reps] = num_reps
	json_data[json_label_n_users] = np.array(num_users_vec).tolist()	
	json_data[json_label_broadcast_delay_means] = delay_means_mat.tolist()			
	json_data[json_label_broadcast_delay_err] = delay_err_mat.tolist()			
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(json_filename, graph_filename_delays, time_slot_duration):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)
		num_users_vec = np.array(json_data[json_label_n_users])		
		delay_means_mat = np.array(json_data[json_label_broadcast_delay_means])
		delay_err_mat = np.array(json_data[json_label_broadcast_delay_err])			
		
		mean_of_delays = np.zeros(len(num_users_vec))
		mean_of_errs = np.zeros(len(num_users_vec))
		for i in range(len(num_users_vec)):
			mean_of_delays[i] = np.mean(delay_means_mat[i, :])
			mean_of_errs[i] = np.mean(delay_err_mat[i, :])		
   				
		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		# 1st graph: delay		
		fig = plt.figure()		
		line = plt.errorbar(num_users_vec, mean_of_delays*time_slot_duration, mean_of_errs*time_slot_duration, alpha=0.5, fmt='o')
		plt.plot(num_users_vec, mean_of_delays*time_slot_duration, linestyle='--', linewidth=.5, color=line[0].get_color(), alpha=.5)		
		plt.ylabel('MAC delays [ms]')		
		plt.xlabel('Number of users $n$')		
		# plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.5), ncol=3)		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_delays, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delays)    
		plt.close()  


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='broadcast_delays')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')			
	parser.add_argument('--n', type=int, nargs='+', help='Number of transmitters.', default=[5])		
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)
	parser.add_argument('--time_slot_duration', type=int, help='Duration of a time slot in milliseconds.', default=24)		

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + "_n-" + str(args.n) + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename_delays = "_imgs/" + output_filename_base + "_delay.pdf"	
	if not args.no_parse:		
		parse(args.dir, args.n, args.num_reps, json_filename)
	if not args.no_plot:
		plot(json_filename, graph_filename_delays, args.time_slot_duration) 
    