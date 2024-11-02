# The L-Band Digital Aeronautical Communications System (LDACS) simulator provides an installation script for the simulator that downloads the other simulator components, defines simulation scenarios and provides result evaluation and graph creation.
# Copyright (C) 2023  Sebastian Lindner, Konrad Fuger, Musab Ahmed Eltayeb Ahmed, Andreas Timm-Giel, Institute of Communication Networks, Hamburg University of Technology, Hamburg, Germany
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import settings
from datetime import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import sub
import argparse
import json
import scipy.stats
import os
import progressbar
import csv
import math
from utils import calculate_confidence_interval


json_label_num_users_per_swarm = 'num_users_group1'
json_label_reps = 'num_reps'
json_label_beacon_rx_delay_vals_1 = 'beacon_rx_delays_swarm_1'
json_label_beacon_rx_delay_vals_2 = 'beacon_rx_delays_swarm_2'
json_label_beacon_rx_window_times = 'beacon_rx_delays_sliding_window_times'
json_label_num_packet_coll_vals_1 = 'num_packet_coll_vals_1'
json_label_num_packet_coll_vals_2 = 'num_packet_coll_vals_1'
json_label_num_packet_coll_window_times = 'num_packet_coll_window_times'
json_label_max_simtime = 'max_simulation_time'
json_label_num_time_slots = 'num_time_slots'

# from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array/2566508#2566508
def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


def parse(dir_mcsotdma, dir_sotdma, dir_rsaloha, num_reps, num_users_per_swarm, json_filename, time_slot_duration, max_simulation_time):
	avg_beacon_rx_times = []
	avg_beacon_rx_1 = []
	avg_beacon_rx_2 = []
	avg_num_packet_coll_1 = []
	avg_num_packet_coll_2 = []
	avg_num_beacon_times = []

	bar_max_i = num_reps * 3
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	all_three_dirs = [dir_mcsotdma, dir_sotdma, dir_rsaloha]
	for j in range(len(all_three_dirs)):
		dir = all_three_dirs[j]
		avg_beacon_rx_1.append([])
		avg_beacon_rx_2.append([])
		avg_num_packet_coll_1.append([])
		avg_num_packet_coll_2.append([])
		for rep in range(num_reps):
			# append a new list for this repetition		
			avg_beacon_rx_1[j].append([])
			avg_beacon_rx_2[j].append([])
			avg_num_packet_coll_1[j].append([])
			avg_num_packet_coll_2[j].append([])
			try:
				filename = dir + '/#' + str(rep)
				filename_sca = filename + '.sca.csv'
				filename_vec = filename + '.vec.csv'
				results_vec = pd.read_csv(filename_vec)			
							
				this_rep_beacon_rx_vals_swarm_1 = []
				this_rep_beacon_rx_times_swarm_1 = []
				this_rep_beacon_rx_vals_swarm_2 = []
				this_rep_beacon_rx_times_swarm_2 = []
				this_rep_num_packet_coll_vals_swarm_1 = []
				this_rep_num_packet_coll_times_swarm_1 = []
				this_rep_num_packet_coll_vals_swarm_2 = []
				this_rep_num_packet_coll_times_swarm_2 = []
				for i in range(num_users_per_swarm):				
					this_rep_beacon_rx_vals_swarm_1.append([])
					this_rep_beacon_rx_times_swarm_1.append([])
					this_rep_beacon_rx_vals_swarm_2.append([])
					this_rep_beacon_rx_times_swarm_2.append([])
					this_rep_num_packet_coll_vals_swarm_1.append([])
					this_rep_num_packet_coll_times_swarm_1.append([])
					this_rep_num_packet_coll_vals_swarm_2.append([])
					this_rep_num_packet_coll_times_swarm_2.append([])
					
					# beacon reception delays of first neighbor whose beacon has been received
					beacon_rx_delay_results_1 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_avg_beacon_rx_delay:vector') & (results_vec.module=='mobility.aircraft_group1[' + str(i) + '].wlan[0].linkLayer')]				
					this_rep_beacon_rx_vals_swarm_1[-1].extend(np.array([float(s) for s in beacon_rx_delay_results_1['vecvalue'].values[0].split(' ')]))
					this_rep_beacon_rx_times_swarm_1[-1] = [float(s) for s in beacon_rx_delay_results_1['vectime'].values[0].split(' ')]

					beacon_rx_delay_results_2 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_avg_beacon_rx_delay:vector') & (results_vec.module=='mobility.aircraft_group2[' + str(i) + '].wlan[0].linkLayer')]				
					this_rep_beacon_rx_vals_swarm_2[-1].extend(np.array([float(s) for s in beacon_rx_delay_results_2['vecvalue'].values[0].split(' ')]))
					this_rep_beacon_rx_times_swarm_2[-1] = [float(s) for s in beacon_rx_delay_results_2['vectime'].values[0].split(' ')]

					# number of beacons received
					num_packet_coll_results_1 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_num_packet_collisions:vector') & (results_vec.module=='mobility.aircraft_group1[' + str(i) + '].wlan[0].linkLayer')]
					this_rep_num_packet_coll_vals_swarm_1[-1].extend(np.array([float(s) for s in num_packet_coll_results_1['vecvalue'].values[0].split(' ')]))
					this_rep_num_packet_coll_times_swarm_1[-1] = [float(s) for s in num_packet_coll_results_1['vectime'].values[0].split(' ')]

					num_packet_coll_results_2 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_num_packet_collisions:vector') & (results_vec.module=='mobility.aircraft_group2[' + str(i) + '].wlan[0].linkLayer')]
					this_rep_num_packet_coll_vals_swarm_2[-1].extend(np.array([float(s) for s in num_packet_coll_results_2['vecvalue'].values[0].split(' ')]))
					this_rep_num_packet_coll_times_swarm_2[-1] = [float(s) for s in num_packet_coll_results_2['vectime'].values[0].split(' ')]				
																

				# captured beacon RX delays are saved for each user 
				# now get mean delay over all users in a swarm as sliding window over simulation time			
				# for each user
				mean_beacon_rx_delays_per_user_1 = []
				mean_beacon_rx_delays_per_user_2 = []				
				for n in range(num_users_per_swarm):
					# swarm 1
					user_beacon_rx_times = np.array(this_rep_beacon_rx_times_swarm_1[n])
					user_beacon_rx_vals = np.array(this_rep_beacon_rx_vals_swarm_1[n])
					mean_beacon_rx_delays_per_user_1.append([])
					last_i = 0				
					for t in range(5, max_simulation_time, 5):
						# find index closest to this simulation time
						i = find_nearest(user_beacon_rx_times, t)					
						mean_beacon_rx_delays_per_user_1[-1].append(np.mean(user_beacon_rx_vals[last_i:i]))					
						last_i = i				

					# swarm 2
					user_beacon_rx_times = np.array(this_rep_beacon_rx_times_swarm_2[n])
					user_beacon_rx_vals = np.array(this_rep_beacon_rx_vals_swarm_2[n])
					mean_beacon_rx_delays_per_user_2.append([])
					last_i = 0
					for t in range(5, max_simulation_time, 5):
						# find index closest to this simulation time
						i = find_nearest(user_beacon_rx_times, t)
						mean_beacon_rx_delays_per_user_2[-1].append(np.mean(user_beacon_rx_vals[last_i:i]))
						last_i = i
				mean_beacon_rx_delays_per_user_1 = np.array(mean_beacon_rx_delays_per_user_1)
				mean_beacon_rx_delays_per_user_2 = np.array(mean_beacon_rx_delays_per_user_2)
				for i in range(len(mean_beacon_rx_delays_per_user_1[0])):
					# compute mean at this window over all users
					avg_beacon_rx_1[j][rep].append(np.mean(mean_beacon_rx_delays_per_user_1[:,i]))
					avg_beacon_rx_2[j][rep].append(np.mean(mean_beacon_rx_delays_per_user_2[:,i]))
				if rep == 0 and j == 0:
					for t in range(5, max_simulation_time, 5):
						avg_beacon_rx_times.append(t)

				# now do the same for the number of packet collisions
				mean_num_packet_coll_per_user_1 = []
				mean_num_packet_coll_per_user_2 = []
				for n in range(num_users_per_swarm):
					# swarm 1
					user_num_beacon_times = np.array(this_rep_num_packet_coll_times_swarm_1[n])
					user_num_packet_coll_vals = np.array(this_rep_num_packet_coll_vals_swarm_1[n])
					mean_num_packet_coll_per_user_1.append([])
					last_i = 0
					for t in range(5, max_simulation_time, 5):
						# find index closest to this simulation time
						i = find_nearest(user_num_beacon_times, t)					
						mean_num_packet_coll_per_user_1[-1].append(np.mean(user_num_packet_coll_vals[last_i:i]))					
						last_i = i					

					# swarm 2
					user_num_beacon_times = np.array(this_rep_num_packet_coll_times_swarm_2[n])
					user_num_packet_coll_vals = np.array(this_rep_num_packet_coll_vals_swarm_2[n])
					mean_num_packet_coll_per_user_2.append([])
					last_i = 0
					for t in range(5, max_simulation_time, 5):
						# find index closest to this simulation time
						i = find_nearest(user_num_beacon_times, t)
						mean_num_packet_coll_per_user_2[-1].append(np.mean(user_num_packet_coll_vals[last_i:i]))
						last_i = i
				mean_num_packet_coll_per_user_1 = np.array(mean_num_packet_coll_per_user_1)
				mean_num_packet_coll_per_user_2 = np.array(mean_num_packet_coll_per_user_2)
				for i in range(len(mean_num_packet_coll_per_user_1[0])):
					# compute mean at this window over all users
					avg_num_packet_coll_1[j][rep].append(np.mean(mean_num_packet_coll_per_user_1[:,i]))
					avg_num_packet_coll_2[j][rep].append(np.mean(mean_num_packet_coll_per_user_2[:,i]))
				if rep == 0 and j == 0:
					for t in range(5, max_simulation_time, 5):
						avg_num_beacon_times.append(t)


				bar_i += 1
				bar.update(bar_i)
			except FileNotFoundError as err:
				print(err)
	bar.finish()		

	# save to JSON
	json_data = {}	
	json_data[json_label_reps] = num_reps
	json_data[json_label_num_users_per_swarm] = num_users_per_swarm			
	json_data[json_label_beacon_rx_delay_vals_1] = np.array(avg_beacon_rx_1).tolist()
	json_data[json_label_beacon_rx_delay_vals_2] = np.array(avg_beacon_rx_2).tolist()
	json_data[json_label_beacon_rx_window_times] = np.array(avg_beacon_rx_times).tolist()
	json_data[json_label_num_packet_coll_vals_1] = np.array(avg_num_packet_coll_1).tolist()
	json_data[json_label_num_packet_coll_vals_2] = np.array(avg_num_packet_coll_2).tolist()
	json_data[json_label_num_packet_coll_window_times] = np.array(avg_num_beacon_times).tolist()
	json_data[json_label_max_simtime] = max_simulation_time
				
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")  


def plot(json_filename, time_slot_duration, graph_filename_delays):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:		
		# load JSON
		json_data = json.load(json_file)			
		max_simulation_time = json_data[json_label_max_simtime]		
		times_vec = np.zeros(max_simulation_time*2)
		t0 = 140
		t1 = 270.55
		t2 = 564
		
		# plot average beacon RX delay over simulation time
		avg_beacon_rx_delay_1 = np.array(json_data[json_label_beacon_rx_delay_vals_1])				
		avg_beacon_rx_delay_2 = np.array(json_data[json_label_beacon_rx_delay_vals_2])
		avg_beacon_rx_times = np.array(json_data[json_label_beacon_rx_window_times])
		avg_num_packet_coll_1 = np.array(json_data[json_label_num_packet_coll_vals_1])				
		avg_num_packet_coll_2 = np.array(json_data[json_label_num_packet_coll_vals_2])
		avg_num_packet_coll_times = np.array(json_data[json_label_num_packet_coll_window_times])
		# compute confidence interval over all values
		avg_beacon_rx_delay_both_swarms = np.zeros((3, 3, len(avg_beacon_rx_times)))
		avg_num_packet_coll_both_swarms = np.zeros((3, 3, len(avg_beacon_rx_times)))
		for i in range(3):
			for t in range(len(avg_beacon_rx_times)):
				ci_1 = calculate_confidence_interval(avg_beacon_rx_delay_1[i,:,t])
				ci_2 = calculate_confidence_interval(avg_beacon_rx_delay_2[i,:,t])
				avg_beacon_rx_delay_both_swarms[i,:,t] = np.mean([ci_1, ci_2], axis=0)
			for t in range(len(avg_num_packet_coll_times)):
				ci_1 = calculate_confidence_interval(avg_num_packet_coll_1[i,:,t])
				ci_2 = calculate_confidence_interval(avg_num_packet_coll_2[i,:,t])
				avg_num_packet_coll_both_swarms[i,:,t] = np.mean([ci_1, ci_2], axis=0)

		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		fig, ax1 = plt.subplots()		
		plt.xlabel('Simulation time $t$ [s]')
		labels = ['MCSOTDMA', 'SOTDMA', 'RS-ALOHA']
		colors = ['xkcd:teal', 'xkcd:coral', 'xkcd:goldenrod', 'xkcd:maroon', 'xkcd:sea blue']
		for i in range(len(labels)):
			ax1.plot(avg_beacon_rx_times, avg_beacon_rx_delay_both_swarms[i,0,:] * time_slot_duration / 1000, linewidth=.75, alpha=1, linestyle='-', label=labels[i], color=colors[i])
			ax1.fill_between(avg_beacon_rx_times, avg_beacon_rx_delay_both_swarms[i,1,:] * time_slot_duration / 1000, avg_beacon_rx_delay_both_swarms[i,2,:] * time_slot_duration / 1000, alpha=.5, color=colors[i])		
		ax1.plot(0, 0, linestyle=':', linewidth=.75, color='black', label="same protocols' packet collisions")
		plt.legend(framealpha=0.0, loc='upper center', bbox_to_anchor=(.55, 1.35), ncol=2)
		ax1.set_xticks([0, t0, t1, 400, t2, 800])
		ax1.set_xticklabels(['0', '$t_0$', '$t_1$', '400', '$t_2$', '800'])		
		ax1.set_ylabel('sliding window over avg time\n in-between packet receptions [s]')		
		for t in [t0, t1, t2]:
			ax1.axvline(t, linestyle='--', color='k', linewidth=.5)		

		ax2 = ax1.twinx()
		for i in range(len(labels)):
			ax2.plot(avg_num_packet_coll_times, np.nan_to_num(avg_num_packet_coll_1[i,0,:]), linewidth=.75, alpha=1, linestyle=':', color=colors[i])
			ax2.fill_between(avg_num_packet_coll_times, np.nan_to_num(avg_num_packet_coll_1[i,1,:]), np.nan_to_num(avg_num_packet_coll_1[i,2,:]), alpha=.5, color=colors[i])		
		ax2.set_ylabel('sliding window over packet collisions')
		ylabels = ['{:,.0f}'.format(y) + 'k' for y in ax2.get_yticks()/1000]
		ax2.set_yticklabels(ylabels)					

		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width*1.75, settings.fig_height*1.2), forward=False)
		fig.savefig(graph_filename_delays, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delays)    
		plt.close()


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='mobility_comparison')
	parser.add_argument('--dir_mcsotdma', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--dir_sotdma', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--dir_rsaloha', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')				
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)
	parser.add_argument('--num_users_per_swarm', type=int, help='Number of users in either swarm.', default=15)	
	parser.add_argument('--time_slot_duration', type=int, help='Duration of a time slot in milliseconds.', default=24)
	parser.add_argument('--max_simulation_time', type=int, help='Maximum simulation time in seconds.', default=800)

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"	
	graph_filename = "_imgs/" + output_filename_base + ".pdf"	
	if not args.no_parse:		
		parse(args.dir_mcsotdma, args.dir_sotdma, args.dir_rsaloha, args.num_reps, args.num_users_per_swarm, json_filename, args.time_slot_duration, args.max_simulation_time)
	if not args.no_plot:
		plot(json_filename, args.time_slot_duration, graph_filename) 
    