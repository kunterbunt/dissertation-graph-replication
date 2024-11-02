import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import scipy.stats
import os
import progressbar


json_label_reps = 'num_reps'
json_label_n_users = 'n_users'
json_label_intervals = 'intervals'
json_label_broadcasts_sent = 'broadcasts_sent'
json_label_broadcasts_rcvd = 'broadcasts_rcvd'
json_label_beacons_sent = 'beacons_sent'
json_label_beacons_rcvd = 'beacons_rcvd'
json_label_candidate_slot_set = 'cand_slot_set'
json_label_collisions = 'collisions'


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]


def parse(directory, sending_intervals, num_users_vec, num_reps, json_filename, flip_params, use_random_interval_filenames):		
	"""
	Parses all .csv result files from 'sending_intervals' and 'num_reps' and saves result values to 'json_filename'.
	"""	
	broadcast_sent_mat = np.zeros((len(sending_intervals), len(num_users_vec), num_reps))
	broadcast_rcvd_mat = np.zeros((len(sending_intervals), len(num_users_vec), num_reps))
	candidate_slot_set_mat = np.zeros((len(sending_intervals), len(num_users_vec), num_reps))	

	bar_max_i = len(sending_intervals) * len(num_users_vec) * num_reps
	bar_i = 0
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	print('parsing ' + str(bar_max_i) + ' result files...')
	for i in range(len(sending_intervals)):
		for j in range(len(num_users_vec)):
			interval = sending_intervals[i]
			for rep in range(num_reps):
				try:
					n = num_users_vec[j]
					filename = None
					if use_random_interval_filenames:
						filename = directory + '/e=' + str(interval) + ',n=' + str(n) + '-#' + str(rep) + '.sca.csv'							
					else:
						if flip_params:
							filename = directory + '/s=' + str(interval) + ',n=' + str(n) + '-#' + str(rep) + '.sca.csv'							
						else:
							filename = directory + '/n=' + str(n) + ',s=' + str(interval) + '-#' + str(rep) + '.sca.csv'							
					
					results = pd.read_csv(filename)					
					# Broadcasts & candidate slot set size
					for transmitter in range(n):
						broadcast_sent_mat[i][j][rep] += results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_num_broadcasts_sent:last') & (results.module=='NW_TX_RX.txNodes[' + str(transmitter) + '].wlan[0].linkLayer')].value
						# Mean candidate slot set for each user, summed up
						candidate_slot_set_mat[i][j][rep] += results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_broadcast_candidate_slots:last') & (results.module=='NW_TX_RX.txNodes[' + str(transmitter) + '].wlan[0].linkLayer')].value																			
					candidate_slot_set_mat[i][j][rep] /= n  # mean of means					
					broadcast_rcvd_mat[i][j][rep] = results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_num_broadcasts_received:last') & (results.module=='NW_TX_RX.rxNode.wlan[0].linkLayer')].value																							
					bar_i += 1
					bar.update(bar_i)
					
				except FileNotFoundError as err:
					print(err)				
	# Save to JSON.
	json_data = {}
	json_data[json_label_n_users] = np.array(num_users_vec).tolist()
	json_data[json_label_reps] = num_reps
	json_data[json_label_intervals] = np.array(sending_intervals).tolist()	
	json_data[json_label_broadcasts_sent] = broadcast_sent_mat.tolist()
	json_data[json_label_broadcasts_rcvd] = broadcast_rcvd_mat.tolist()	
	json_data[json_label_candidate_slot_set] = candidate_slot_set_mat.tolist()	
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(sim_time_ms, target_reception_rate, json_filename, graph_broadcasts_filename, graph_beacons_filename, graph_candidate_slots_filename):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)
		sending_intervals = json_data[json_label_intervals]		
		n_users_vec = json_data[json_label_n_users]
		broadcast_sent_mat = json_data[json_label_broadcasts_sent]		
		broadcast_rcvd_mat = json_data[json_label_broadcasts_rcvd]		
		candidate_slot_set_mat = json_data[json_label_candidate_slot_set]				
		# Calculate confidence intervals
		contention_means = np.zeros((len(sending_intervals), len(n_users_vec)))
		contention_err = np.zeros((len(sending_intervals), len(n_users_vec)))		
		broadcasts_rcvd_means = np.zeros((len(sending_intervals), len(n_users_vec)))
		broadcasts_rcvd_err = np.zeros((len(sending_intervals), len(n_users_vec)))				
		candidate_slots_means = np.zeros((len(sending_intervals), len(n_users_vec)))
		candidate_slots_err = np.zeros((len(sending_intervals), len(n_users_vec)))		
		for i in range(len(sending_intervals)):
			for j in range(len(n_users_vec)):				
				# Fraction of received broadcasts
				broadcasts_rcvd_means[i][j], broadcast_rcvd_m, broadcast_rcvd_p = calculate_confidence_interval(np.array(broadcast_rcvd_mat[i][j][:]) / np.array(broadcast_sent_mat[i][j][:]), confidence=.95)
				broadcasts_rcvd_err[i][j] = broadcast_rcvd_p - broadcasts_rcvd_means[i][j]		
				# Candidate slot set size
				candidate_slots_means[i][j], candidate_slots_m, candidate_slots_p = calculate_confidence_interval(candidate_slot_set_mat[i][j][:], confidence=.95)												
				candidate_slots_err[i][j] = candidate_slots_p - candidate_slots_means[i][j]				
   		
		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})		  
		# 1st plot: broadcast reception rate
		fig = plt.figure()
		for n in range(len(n_users_vec)):						
			line = plt.plot(sending_intervals, broadcasts_rcvd_means[:, n] * 100, alpha=.5)					
			plt.errorbar(sending_intervals, broadcasts_rcvd_means[:, n] * 100, broadcasts_rcvd_err[:, n] * 100, alpha=0.5, color=line[0].get_color(), fmt='o', label='$n=' + str(n_users_vec[n]) + '$')							
		plt.axhline(target_reception_rate, color='k', linestyle=':', label='target')
		plt.xlabel(r'Broadcast sending interval [ms/packet]')		
		plt.ylabel('Reception rate [\%]')
		plt.legend()
		plt.gca().legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.55), ncol=3)		
		fig.tight_layout()
		fig.set_size_inches((4.7*.41, 3.5*.425), forward=False)
		fig.savefig(graph_broadcasts_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_broadcasts_filename)
  
		# 2nd plot: candidate slot set
		fig, ax1 = plt.subplots()		
		ax2 = ax1.twinx()
		t_slot = 12  # ms
		for n in range(len(n_users_vec)):						
			line = ax1.plot(sending_intervals, candidate_slots_means[:, n], alpha=.5)					
			ax1.errorbar(sending_intervals, candidate_slots_means[:, n], candidate_slots_err[:, n], alpha=0.5, color=line[0].get_color(), fmt='o', label='$n=' + str(n_users_vec[n]) + '$')							
			line = ax2.plot(sending_intervals, candidate_slots_means[:, n] * t_slot, alpha=0, linestyle=None)
			ax2.errorbar(sending_intervals, candidate_slots_means[:, n] * t_slot, candidate_slots_err[:, n] * t_slot, alpha=0, color=line[0].get_color(), fmt='o')
		ax1.set_xlabel(r'Broadcast sending interval [ms/packet]')		
		ax1.set_ylabel('BC candidate slots [\#{}]')
		ax2.set_ylabel('Delay [ms]')
		# ax1.set_ylim([3, np.max(candidate_slots_means) + 1])
		# ax1.set_yticks(range(3, int(np.max(candidate_slots_means) + 1), 3))
		ax1.legend()
		ax1.legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.55), ncol=2)		
		fig.tight_layout()
		fig.set_size_inches((4.7*.35, 3.5*.425), forward=False)
		fig.savefig(graph_candidate_slots_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_candidate_slots_filename)  			  		


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='default_eval')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--flip_params', action='store_true', help='Flip the parameters when reading files; some simulations have inverted filenames.')
	parser.add_argument('--randomintervals', action='store_true', help='Expect simulation result filesnames as generated for the random sending intervals.')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')		
	parser.add_argument('--sending_interval_start', type=int, help='Sending interval lower bound, e.g. 1 if in OMNeT++ you used ${s=1 .. 613 step 12}ms.')
	parser.add_argument('--sending_interval_stop', type=int, help='Sending interval upper bound, e.g. 613 if in OMNeT++ you used ${s=1 .. 613 step 12}ms.')
	parser.add_argument('--sending_interval_step', type=int, help='Sending interval step size, e.g. 12 if in OMNeT++ you used ${s=1 .. 613 step 12}ms.')
	parser.add_argument('--n', type=int, nargs='+', help='Number of transmitters.', default=[2])
	parser.add_argument('--sim_time', type=int, help='Number of milliseconds the simulation covered.', default=60000)
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)
	parser.add_argument('--target_reception_rate', type=int, help='Plots a horizontal line on reception rate graphs at this percentage.', default=95)	

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + "_n_" + str(args.n) + "_" + str(args.sending_interval_start) + "-" + str(args.sending_interval_stop) + "-" + str(args.sending_interval_step) + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"	
	graph_broadcasts_filename = "_imgs/" +output_filename_base + "_broadcasts.pdf"
	graph_beacons_filename = "_imgs/" +output_filename_base + "_beacons.pdf"
	graph_candidate_slots_filename = "_imgs/" +output_filename_base + "_candidate-slots.pdf"	
	if not args.no_parse:
		sending_intervals = range(args.sending_interval_start, args.sending_interval_stop+args.sending_interval_step, args.sending_interval_step)  			
		parse(args.dir, sending_intervals, args.n, args.num_reps, json_filename, args.flip_params, args.randomintervals)
	if not args.no_plot:
		plot(args.sim_time, args.target_reception_rate, json_filename, graph_broadcasts_filename, graph_beacons_filename, graph_candidate_slots_filename) 
    