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
json_label_num_users = 'num_users_mat'
json_label_channel_err = 'channel_err_vec'
json_label_reception_rate_mat = 'reception_rate_mat'
json_label_delay_mat = 'mac_delay_mat'
json_label_beacon_rx_time_means = 'beacon_rx_time_means'
json_label_beacon_rx_time_err = 'beacon_rx_time_err'


def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]

def parse(dir, num_users_vec, channel_err_vec, num_reps, json_filename):		
	reception_rate_mat = np.zeros((len(num_users_vec), len(channel_err_vec), num_reps))	
	delay_mat = np.zeros((len(num_users_vec), len(channel_err_vec), num_reps))	
	avg_beacon_rx_mat_means = np.zeros((len(num_users_vec), len(channel_err_vec)))
	avg_beacon_rx_mat_err = np.zeros(((len(num_users_vec), len(channel_err_vec))))			
	bar_max_i = len(num_users_vec) * len(channel_err_vec) * num_reps
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()		
	# for each channel error
	for i in range(len(num_users_vec)):
		n = num_users_vec[i]		
		for j in range(len(channel_err_vec)):
			e = channel_err_vec[j]
			beacon_rx_time_mat = np.zeros(num_reps)
			# for each repetition
			for rep in range(num_reps):
				try:	
					filename = None								
					if e < .1:
						filename = dir + '/n=' + str(n) + ',e=' + str(e) + '-#' + str(rep) + '.sca.csv'
					else:
						filename = dir + '/n=' + str(n) + ',e=' + str("%.2f" % round(e,2)) + '-#' + str(rep) + '.sca.csv'					
					results = pd.read_csv(filename)	
					# get the mean delay
					mean_delay_per_transmitter = np.zeros(n)
					# get the total number of transmitted broadcasts
					num_broadcasts_sent = 0					
					for user in range(n):
						num_broadcasts_sent += int(results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_num_broadcasts_sent:last') & (results.module=='NW_TX_RX.txNodes[' + str(user) + '].wlan[0].linkLayer')].value)												
						mean_delay_per_transmitter[user] = results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_broadcast_mac_delay:mean') & (results.module=='NW_TX_RX.txNodes[' + str(user) + '].wlan[0].linkLayer')].value
					beacon_rx_time_mat[rep] = results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_first_neighbor_beacon_rx_delay:mean') & (results.module=='NW_TX_RX.rxNode.wlan[0].linkLayer')].value
					# take the number of received broadcasts at the RX node
					reception_rate_mat[i][j][rep] = int(results[(results.type=='scalar') & (results.name=='mcsotdma_statistic_num_broadcasts_received:last') & (results.module=='NW_TX_RX.rxNode.wlan[0].linkLayer')].value)										
					# divide by all broadcasts to get the reception rate
					reception_rate_mat[i][j][rep] /= max(1, num_broadcasts_sent)
					# get mean of mean delays
					delay_mat[i][j][rep] = np.mean(mean_delay_per_transmitter)

					bar_i += 1
					bar.update(bar_i)
				except FileNotFoundError as err:
					print(err)	
			avg_beacon_rx_mat_means[i,j], _, plus = calculate_confidence_interval(beacon_rx_time_mat, confidence=.95)												
			avg_beacon_rx_mat_err[i,j] = plus - avg_beacon_rx_mat_means[i,j]
	bar.finish()			

	# Save to JSON.
	json_data = {}
	json_data[json_label_reps] = num_reps
	json_data[json_label_num_users] = np.array(num_users_vec).tolist()
	json_data[json_label_channel_err] = np.array(channel_err_vec).tolist()
	json_data[json_label_beacon_rx_time_means] = avg_beacon_rx_mat_means.tolist()		
	json_data[json_label_beacon_rx_time_err] = avg_beacon_rx_mat_err.tolist()		
	json_data[json_label_reception_rate_mat] = reception_rate_mat.tolist()		
	json_data[json_label_delay_mat] = delay_mat.tolist()
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(json_filename, graph_filename_reception_rate, graph_filename_delay, target_reception_rate, time_slot_duration):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)
		num_reps = json_data[json_label_reps]
		channel_err_vec = np.array(json_data[json_label_channel_err])
		num_users_vec = np.array(json_data[json_label_num_users])
		reception_rate_mat = np.array(json_data[json_label_reception_rate_mat])			
		delay_mat = np.array(json_data[json_label_delay_mat])
		avg_beacon_rx_mat_means = np.array(json_data[json_label_beacon_rx_time_means])
		avg_beacon_rx_mat_err = np.array(json_data[json_label_beacon_rx_time_err])		
		
		# Calculate confidence intervals
		broadcast_reception_rate_means = np.zeros((len(num_users_vec), len(channel_err_vec)))
		broadcast_reception_rate_err = np.zeros((len(num_users_vec), len(channel_err_vec)))
		delay_means = np.zeros((len(num_users_vec), len(channel_err_vec)))
		delay_err = np.zeros((len(num_users_vec), len(channel_err_vec)))
		for i in range(len(num_users_vec)):		
			for j in range(len(channel_err_vec)):
				broadcast_reception_rate_means[i,j], _, reception_rate_p = calculate_confidence_interval(reception_rate_mat[i,j,:], confidence=.95)
				broadcast_reception_rate_err[i,j] = reception_rate_p - broadcast_reception_rate_means[i,j]		
				delay_means[i,j], _, delay_p = calculate_confidence_interval(delay_mat[i,j,:], confidence=.95)
				delay_err[i,j] = delay_p - delay_means[i,j]		
   				
		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		colors = ['xkcd:teal', 'xkcd:coral', 'xkcd:goldenrod', 'xkcd:maroon', 'xkcd:sea blue']
		# 1st graph: reception rate
		fig = plt.figure()		
		for i in range(len(channel_err_vec)):
			line = plt.errorbar(num_users_vec, broadcast_reception_rate_means[:,i]*100, broadcast_reception_rate_err[:,i]*100, markersize=2, fmt='o', label='$p =' + str(channel_err_vec[i]) + '$', zorder=1, color=colors[i])
			plt.plot(num_users_vec, broadcast_reception_rate_means[:,i]*100, linestyle='--', linewidth=.5, color=line[0].get_color(), zorder=1)
			plt.axhline((1.0-channel_err_vec[i])*100, linestyle='--', linewidth=.75, color='k', zorder=0)
		plt.ylabel('Reception rate [\%]')				
		plt.ylim([0, 105])
		plt.xlabel('No. of neighbors $n$')		
		plt.xticks(num_users_vec)
		if target_reception_rate is not None:
			plt.axhline(target_reception_rate*100, linestyle='--', linewidth=.75, color='k', zorder=0)
			plt.yticks([0, target_reception_rate*100, 100])
		plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height*1.25), forward=False)
		fig.savefig(graph_filename_reception_rate, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_reception_rate)    
		plt.close()

		# 2nd graph: delays		
		fig = plt.figure()		
		for i in range(len(channel_err_vec)):
			line = plt.errorbar(num_users_vec, delay_means[:,i]*time_slot_duration, delay_err[:,i]*time_slot_duration, markersize=2, fmt='o', label='$p =' + str(channel_err_vec[i]) + '$', zorder=1, color=colors[i])
			plt.plot(num_users_vec, delay_means[:,i]*time_slot_duration, linestyle='-', linewidth=.5, color=line[0].get_color(), zorder=1)
			colors.append(line[0].get_color())				
		for i in range(len(channel_err_vec)):
			line = plt.errorbar(num_users_vec, avg_beacon_rx_mat_means[:,i]*time_slot_duration, avg_beacon_rx_mat_err[:,i]*time_slot_duration, markersize=6 , fmt='x', zorder=1, color=colors[i])
			plt.plot(num_users_vec, avg_beacon_rx_mat_means[:,i]*time_slot_duration, linestyle='--', linewidth=.5, color=line[0].get_color(), zorder=1)
		# two fake data points to add entries to the legend
		line = plt.errorbar(min(num_users_vec), min(avg_beacon_rx_mat_means[:,0]*time_slot_duration), 0, label='MAC Delay', color='k', markersize=2, fmt='o')
		line.remove()
		line = plt.errorbar(min(num_users_vec), min(avg_beacon_rx_mat_means[:,0]*time_slot_duration), 0, label='E2E Delay', color='k', markersize=6, fmt='x')
		line.remove()				
		plt.ylabel('Delay [ms]')				
		plt.xlabel('No. of neighbors $n$')		
		plt.xticks(num_users_vec)
		plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=3, columnspacing=0.5)				
		fig.tight_layout()
		fig.set_size_inches((settings.fig_width, settings.fig_height*1.25), forward=False)
		fig.savefig(graph_filename_delay, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delay)    
		plt.close()


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='broadcast_packet_error')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')				
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)
	parser.add_argument('--num_users', nargs='+', type=int, help='Number of users.', default=[5, 10])		
	parser.add_argument('--channel_errors', nargs='+', type=float, help='Configured channel error in integer percentage points.', default=[0.0, 0.1, 0.2])		
	parser.add_argument('--target', type=float, help='Target reception rate.', default=None)		
	parser.add_argument('--time_slot_duration', type=float, help='Time slot duration.', default=24)

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + "_e-" + str(args.channel_errors) + "_n-" + str(args.num_users) + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename_reception_rate = "_imgs/" + output_filename_base + "_reception_rate.pdf"	
	graph_filename_delay_delay = "_imgs/" + output_filename_base + "_delay.pdf"	
	if not args.no_parse:		
		parse(args.dir, args.num_users, args.channel_errors, args.num_reps, json_filename)
	if not args.no_plot:
		plot(json_filename, graph_filename_reception_rate, graph_filename_delay_delay, args.target, args.time_slot_duration) 
    