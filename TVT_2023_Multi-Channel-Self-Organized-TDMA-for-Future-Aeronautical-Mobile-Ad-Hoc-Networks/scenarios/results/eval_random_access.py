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
json_label_n_users = 'n_users'
json_label_broadcast_delays = 'broadcast_mac_delays'
json_label_candidate_slots = 'broadcast_candidate_slots'
json_label_selected_slots = 'broadcast_selected_slots'
json_label_reception_rate = 'broadcast_reception_rate'
json_label_broadcast_delay_vec = 'broadcast_mac_delay_vec'
json_label_broadcast_delay_vec_time = 'broadcast_mac_delay_vec_time'
json_label_broadcast_mean_candidate_slots = 'broadcast_mac_mean_candidate_slots'
json_label_broadcast_selected_slots = 'broadcast_mac_selected_slots'

def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]


def parse(dir, num_users_vec, num_candidate_slots_vec, num_reps, json_filename):		
	broadcast_delay_mat = np.zeros((len(num_users_vec), len(num_candidate_slots_vec), num_reps))			
	broadcast_reception_rate_mat = np.zeros((len(num_users_vec), len(num_candidate_slots_vec), num_reps))	
	broadcast_mean_num_candidate_slots_mat = np.zeros((len(num_users_vec), len(num_candidate_slots_vec), num_reps))
	broadcast_mean_selected_slots_mat = np.zeros((len(num_users_vec), len(num_candidate_slots_vec), num_reps))
	bar_max_i = len(num_users_vec)*len(num_candidate_slots_vec)*num_reps
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()	
	# for each number of users
	for i in range(len(num_users_vec)):			
		n = num_users_vec[i]		
		for j in range(len(num_candidate_slots_vec)):
			c = num_candidate_slots_vec[j]
			# for each repetition
			for rep in range(num_reps):
				try:				
					filename = dir + '/n=' + str(n) + ',c=' + str(c) + '-#' + str(rep) + '.sca.csv'
					results = pd.read_csv(filename)				
					# get the total number of transmitted broadcasts
					num_broadcasts = 0
					# and the mean delay per transmitter
					delay_vec = np.zeros(n)
					num_candidate_slots_per_transmitter = np.zeros(n)
					selected_slot_per_transmitter = np.zeros(n)
					for user in range(n):											
						num_broadcasts += int(results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_num_broadcasts_sent:last') & (results.module=='NW_TX_RX.txNodes[' + str(user) + '].wlan[0].linkLayer')].value)					
						delay_vec[user] = results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_broadcast_mac_delay:mean') & (results.module=='NW_TX_RX.txNodes[' + str(user) + '].wlan[0].linkLayer')].value
						num_candidate_slots_per_transmitter[user] = results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_broadcast_candidate_slots:mean') & (results.module=='NW_TX_RX.txNodes[' + str(user) + '].wlan[0].linkLayer')].value					
						selected_slot_per_transmitter[user] = results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_broadcast_selected_candidate_slot:mean') & (results.module=='NW_TX_RX.txNodes[' + str(user) + '].wlan[0].linkLayer')].value											
					# take the number of received broadcasts at the RX node
					broadcast_reception_rate_mat[i][j][rep] = int(results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_num_broadcasts_received:last') & (results.module=='NW_TX_RX.rxNode.wlan[0].linkLayer')].value)
					# divide by all broadcasts to get the reception rate
					broadcast_reception_rate_mat[i][j][rep] /= max(1, num_broadcasts)				
					# take the mean over the mean delays of all users
					broadcast_delay_mat[i][j][rep] = np.mean(delay_vec)										
					broadcast_mean_num_candidate_slots_mat[i][j][rep] = np.mean(num_candidate_slots_per_transmitter)										
					broadcast_mean_selected_slots_mat[i][j][rep] = np.mean(selected_slot_per_transmitter)										
					bar_i += 1
					bar.update(bar_i)
				except FileNotFoundError as err:
					print(err)			
	bar.finish()		

	# Save to JSON.
	json_data = {}
	json_data[json_label_n_users] = np.array(num_users_vec).tolist()
	json_data[json_label_candidate_slots] = np.array(num_candidate_slots_vec).tolist()
	json_data[json_label_broadcast_mean_candidate_slots] = broadcast_mean_num_candidate_slots_mat.tolist()
	json_data[json_label_broadcast_selected_slots] = broadcast_mean_selected_slots_mat.tolist()	
	json_data[json_label_reps] = num_reps
	json_data[json_label_broadcast_delays] = broadcast_delay_mat.tolist()	
	json_data[json_label_reception_rate] = broadcast_reception_rate_mat.tolist()		
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(json_filename, graph_filename_delays, graph_filename_reception, graph_filename_no_of_candidate_slots, graph_filename_selected_slot, time_slot_duration, ylim1, ylim2):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)
		num_users_vec = np.array(json_data[json_label_n_users])
		num_candidate_slots_vec = np.array(json_data[json_label_candidate_slots])
		broadcast_delays_mat = np.array(json_data[json_label_broadcast_delays])
		broadcast_reception_rate_mat = np.array(json_data[json_label_reception_rate])
		broadcast_mean_num_candidate_slots_mat = np.array(json_data[json_label_broadcast_mean_candidate_slots])
		broadcast_mean_selected_slots_mat = np.array(json_data[json_label_broadcast_selected_slots])
		# Calculate confidence intervals
		broadcast_delays_means = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		broadcast_delays_err = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))					
		broadcast_reception_rate_means = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		broadcast_reception_rate_err = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		broadcast_mean_num_candidate_slots_mat_means = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		broadcast_mean_num_candidate_slots_mat_err = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		broadcast_mean_selected_slots_mat_means = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		broadcast_mean_selected_slots_mat_err = np.zeros((len(num_users_vec), len(num_candidate_slots_vec)))
		for i in range(len(num_users_vec)):	
			for j in range(len(num_candidate_slots_vec)):
				broadcast_delays_means[i,j], _, delay_p = calculate_confidence_interval(broadcast_delays_mat[i,j,:], confidence=.95)
				broadcast_delays_err[i,j] = delay_p - broadcast_delays_means[i,j]	
				broadcast_reception_rate_means[i,j], _, reception_rate_p = calculate_confidence_interval(broadcast_reception_rate_mat[i,j,:], confidence=.95)
				broadcast_reception_rate_err[i,j] = reception_rate_p - broadcast_reception_rate_means[i,j]
				broadcast_mean_num_candidate_slots_mat_means[i,j], _, candidate_slots_p = calculate_confidence_interval(broadcast_mean_num_candidate_slots_mat[i,j,:], confidence=.95)
				broadcast_mean_num_candidate_slots_mat_err[i,j] = candidate_slots_p - broadcast_mean_num_candidate_slots_mat_means[i,j]
				broadcast_mean_selected_slots_mat_means[i,j], _, selected_slots_p = calculate_confidence_interval(broadcast_mean_selected_slots_mat[i,j,:], confidence=.95)
				broadcast_mean_selected_slots_mat_err[i,j] = selected_slots_p - broadcast_mean_selected_slots_mat_means[i,j]		
   				
		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		# 1st graph: delay		
		fig = plt.figure()
		colors = ['xkcd:teal', 'xkcd:coral', 'xkcd:goldenrod', 'xkcd:maroon', 'xkcd:sea blue']
		for i in range(len(num_candidate_slots_vec)):
			line = plt.errorbar(num_users_vec, broadcast_delays_means[:,i]*time_slot_duration, broadcast_delays_err[:,i], markersize=2, fmt='o', label='$k=' + str(num_candidate_slots_vec[i]) + '$', color=colors[i])
			plt.plot(num_users_vec, broadcast_delays_means[:,i]*time_slot_duration, linestyle='--', linewidth=.5, color=line[0].get_color())		
		plt.ylabel('MAC delays [ms]')		
		plt.xlabel('Number of users $n$')		
		# plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.25), ncol=2)		
		plt.yscale('log')
		if ylim1 is not None and ylim2 is not None:
			plt.ylim([ylim1, ylim2])
		plt.xticks(num_users_vec, [str(num_users_vec[i]) if num_users_vec[i] != 5 else '' for i in range(len(num_users_vec))])			
		plt.gca().yaxis.grid(True)
		plt.gca().grid(which='major', alpha=.0)
		plt.gca().grid(which='minor', alpha=.5, linewidth=.25, linestyle='-')		
		# plt.axhline(y=10**3, color='gray', linestyle='-', alpha=0.5, linewidth=.5)		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_delays, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delays)    
		plt.close()  

		# 2nd graph: reception rate		
		fig = plt.figure()		
		for i in range(len(num_candidate_slots_vec)):
			line = plt.errorbar(num_users_vec, broadcast_reception_rate_means[:,i]*100, broadcast_reception_rate_err[:,i]*100, markersize=2, fmt='o', label='$k=' + str(num_candidate_slots_vec[i]) + '$', color=colors[i])
			plt.plot(num_users_vec, broadcast_reception_rate_means[:,i]*100, linestyle='--', linewidth=.5, color=line[0].get_color())
		plt.ylabel('Reception rate [\%]')		
		plt.ylim([0, 105])
		plt.xlabel('Number of users $n$')		
		plt.xticks(num_users_vec, [str(num_users_vec[i]) if num_users_vec[i] != 5 else '' for i in range(len(num_users_vec))])	
		# plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.25), ncol=2)		
		plt.gca().yaxis.grid(True)
		plt.gca().grid(which='major', alpha=.0)
		plt.gca().grid(which='minor', alpha=.5, linewidth=.25, linestyle='-')		
		fig.tight_layout()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_reception, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_reception)    
		plt.close()  

		# 3rd graph: no. of candidate slots
		fig = plt.figure()		
		for i in range(len(num_candidate_slots_vec)):
			line = plt.errorbar(num_users_vec, broadcast_mean_num_candidate_slots_mat_means[:,i], broadcast_mean_num_candidate_slots_mat_err[:,i], markersize=2, fmt='o', label='$k=' + str(num_candidate_slots_vec[i]) + '$', color=colors[i])
			plt.plot(num_users_vec, broadcast_mean_num_candidate_slots_mat_means[:,i], linestyle='--', linewidth=.5, color=line[0].get_color())
		plt.ylabel('No. of candidate slots')		
		plt.xlabel('Number of users $n$')		
		plt.xticks(num_users_vec, [str(num_users_vec[i]) if num_users_vec[i] != 5 else '' for i in range(len(num_users_vec))])	
		# plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.35), ncol=2)		
		plt.gca().yaxis.grid(True)
		plt.gca().grid(which='major', alpha=.0)
		plt.gca().grid(which='minor', alpha=.5, linewidth=.25, linestyle='-')		
		fig.tight_layout()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_no_of_candidate_slots, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_no_of_candidate_slots)    
		plt.close()  

		# 4th graph: selected slots
		fig = plt.figure()		
		for i in range(len(num_candidate_slots_vec)):
			line = plt.errorbar(num_users_vec, broadcast_mean_selected_slots_mat_means[:,i], broadcast_mean_selected_slots_mat_err[:,i], markersize=2, fmt='o', label='$k=' + str(num_candidate_slots_vec[i]) + '$', color=colors[i])
			plt.plot(num_users_vec, broadcast_mean_selected_slots_mat_means[:,i], linestyle='--', linewidth=.5, color=line[0].get_color())
		plt.ylabel('Mean selected slot')		
		plt.xlabel('Number of users $n$')		
		plt.xticks(num_users_vec, [str(num_users_vec[i]) if num_users_vec[i] != 5 else '' for i in range(len(num_users_vec))])	
		# plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.35), ncol=2)		
		plt.gca().yaxis.grid(True)
		plt.gca().grid(which='major', alpha=.0)
		plt.gca().grid(which='minor', alpha=.5, linewidth=.25, linestyle='-')		
		fig.tight_layout()
		fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
		fig.savefig(graph_filename_selected_slot, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_selected_slot)    
		plt.close()  


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='broadcast_delays')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')			
	parser.add_argument('--n', type=int, nargs='+', help='Number of users.', default=[5])	
	parser.add_argument('--c', type=int, nargs='+', help='Number of candidate slots.', default=[3])	
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)
	parser.add_argument('--time_slot_duration', type=int, help='Duration of a time slot in milliseconds.', default=24)		
	parser.add_argument('--ylim1', type=int, help='Minimum y-limit for delay plots.', default=None)	
	parser.add_argument('--ylim2', type=int, help='Maximum y-limit for delay plots.', default=None)	

	args = parser.parse_args()	

	if ((args.ylim1 is not None and args.ylim2 is None) or (args.ylim2 is not None and args.ylim1 is None)):
		raise RuntimeError('If you set one ylim, you have to set the other, too!')
		exit(1)
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + "_n-" + str(args.n) + '_c-' + str(args.c) + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename_delays = "_imgs/" + output_filename_base + "_delay.pdf"
	graph_filename_reception = "_imgs/" + output_filename_base + "_reception-rate.pdf"	
	graph_filename_no_of_candidate_slots = "_imgs/" + output_filename_base + "_num-candidate-slots.pdf"
	graph_filename_selected_slot = "_imgs/" + output_filename_base + "_selected-slot.pdf"
	if not args.no_parse:		
		parse(args.dir, args.n, args.c, args.num_reps, json_filename)
	if not args.no_plot:
		plot(json_filename, graph_filename_delays, graph_filename_reception, graph_filename_no_of_candidate_slots, graph_filename_selected_slot, args.time_slot_duration, args.ylim1, args.ylim2) 
    