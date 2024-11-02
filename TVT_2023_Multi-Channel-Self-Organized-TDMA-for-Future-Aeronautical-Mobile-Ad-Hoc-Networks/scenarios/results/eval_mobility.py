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
json_label_num_active_neighbors_vals_swarm_1 = 'num_active_neighbors_vals_swarm_1'
json_label_num_active_neighbors_times_swarm_1 = 'num_active_neighbors_times_swarm_1'
json_label_num_active_neighbors_vals_swarm_2 = 'num_active_neighbors_vals_swarm_2'
json_label_num_active_neighbors_times_swarm_2 = 'num_active_neighbors_times_swarm_2'

json_label_num_packet_drop_vals = 'avg_packet_drop_vals'
json_label_num_packet_drop_times = 'avg_packet_drop_times'

json_label_mac_delay_vals_1 = 'mac_delays_swarm_1'
json_label_mac_delay_vals_2 = 'mac_delays_swarm_2'
json_label_mac_window_times = 'mac_delays_sliding_window_times'
json_label_max_simtime = 'max_simulation_time'
json_label_num_time_slots = 'num_time_slots'

# from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array/2566508#2566508
def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


def parse(dir, num_reps, num_users_per_swarm, json_filename, time_slot_duration, max_sim_time):
	# swarm 1
	num_active_neighbors_times_mat_1 = []  # *_times is the simulation time at which the value was saved
	num_active_neighbors_vals_mat_1 = []  # this is the 'value' in question (same applies for the later statistics)

	# swarm 2
	num_active_neighbors_times_mat_2 = []
	num_active_neighbors_vals_mat_2 = []  
	
	avg_packet_drop_times = []
	avg_packet_drop_vals = []
	sliding_window_dropped_packets = []
	intervals_dropped_packets = []

	avg_mac_times = []
	avg_mac_delay_1 = []
	avg_mac_delay_2 = []

	bar_max_i = num_reps
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()			
	for rep in range(num_reps):
		# append a new list for this repetition
		num_active_neighbors_times_mat_1.append([])
		num_active_neighbors_vals_mat_1.append([])
		num_active_neighbors_times_mat_2.append([])
		num_active_neighbors_vals_mat_2.append([])
		avg_packet_drop_vals.append([])
		avg_packet_drop_times.append([])		
		avg_mac_delay_1.append([])
		avg_mac_delay_2.append([])
		try:
			filename = dir + '/#' + str(rep)
			filename_sca = filename + '.sca.csv'
			filename_vec = filename + '.vec.csv'
			results_vec = pd.read_csv(filename_vec)			
			
			this_rep_avg_packet_drops = []			
			this_rep_mac_vals_swarm_1 = []
			this_rep_mac_times_swarm_1 = []
			this_rep_mac_vals_swarm_2 = []
			this_rep_mac_times_swarm_2 = []

			num_time_slots = 0
			for i in range(num_users_per_swarm):
				this_rep_avg_packet_drops.append([])
				this_rep_mac_vals_swarm_1.append([])
				this_rep_mac_times_swarm_1.append([])
				this_rep_mac_vals_swarm_2.append([])
				this_rep_mac_times_swarm_2.append([])
				# no of neighbors
				num_active_neighbors_results_1 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_num_active_neighbors:vector') & (results_vec.module=='mobility.aircraft_group1[' + str(i) + '].wlan[0].linkLayer')]
				num_active_neighbors_results_2 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_num_active_neighbors:vector') & (results_vec.module=='mobility.aircraft_group2[' + str(i) + '].wlan[0].linkLayer')]				
				num_active_neighbors_times_mat_1[rep].append([float(s) for s in num_active_neighbors_results_1['vectime'].values[0].split(' ')])
				num_active_neighbors_times_mat_2[rep].append([float(s) for s in num_active_neighbors_results_2['vectime'].values[0].split(' ')])
				num_active_neighbors_vals_mat_1[rep].append([float(s) for s in num_active_neighbors_results_1['vecvalue'].values[0].split(' ')])
				num_active_neighbors_vals_mat_2[rep].append([float(s) for s in num_active_neighbors_results_2['vecvalue'].values[0].split(' ')])
				
				# packets dropped
				packet_drops_results_1 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_dropped_packets_this_slot:vector') & (results_vec.module=='mobility.aircraft_group1[' + str(i) + '].wlan[0].linkLayer')]
				packet_drops_results_2 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_dropped_packets_this_slot:vector') & (results_vec.module=='mobility.aircraft_group2[' + str(i) + '].wlan[0].linkLayer')]
				avg_packet_drop_vals[rep].append(np.add(np.array([float(s) for s in packet_drops_results_1['vecvalue'].values[0].split(' ')]), np.array([float(s) for s in packet_drops_results_2['vecvalue'].values[0].split(' ')])))				
				avg_packet_drop_times[rep].append(np.array([float(s) for s in packet_drops_results_1['vectime'].values[0].split(' ')]))
				num_time_slots = len(avg_packet_drop_times[rep][-1])

				# MAC delays
				mac_results_1 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_broadcast_mac_delay:vector') & (results_vec.module=='mobility.aircraft_group1[' + str(i) + '].wlan[0].linkLayer')]				
				this_rep_mac_vals_swarm_1[-1].extend(np.array([float(s) for s in mac_results_1['vecvalue'].values[0].split(' ')]))
				this_rep_mac_times_swarm_1[-1] = [float(s) for s in mac_results_1['vectime'].values[0].split(' ')]				

				mac_results_2 = results_vec[(results_vec.type=='vector') & (results_vec.name=='mcsotdma_statistic_broadcast_mac_delay:vector') & (results_vec.module=='mobility.aircraft_group2[' + str(i) + '].wlan[0].linkLayer')]				
				this_rep_mac_vals_swarm_2[-1].extend(np.array([float(s) for s in mac_results_2['vecvalue'].values[0].split(' ')]))
				this_rep_mac_times_swarm_2[-1] = [float(s) for s in mac_results_2['vectime'].values[0].split(' ')]
									
			# in a sliding window over ten time slots, go over the no. of dropped packets
			sum_dropped_packets = np.zeros(int(num_time_slots/10))			
			i = 0
			for t in range(10, num_time_slots, 10):
				if rep == 0:
					intervals_dropped_packets.append(t)
				for n in range(num_users_per_swarm):
					sum_dropped_packets[i] += np.sum(avg_packet_drop_vals[-1][n][t-10:t-1])
				i += 1
			sliding_window_dropped_packets.append(sum_dropped_packets)
			

			# captured MAC delays are saved for each user 
			# now get mean delay over all users in a swarm as sliding window over simulation time			
			# for each user
			mean_mac_delays_per_user_1 = []
			mean_mac_delays_per_user_2 = []
			for n in range(num_users_per_swarm):
				# swarm 1
				user_mac_times = np.array(this_rep_mac_times_swarm_1[n])				
				user_mac_vals = np.array(this_rep_mac_vals_swarm_1[n])
				mean_mac_delays_per_user_1.append([])
				last_i = 0
				for t in range(5, max_sim_time, 5):
					# find index closest to this simulation time
					i = find_nearest(user_mac_times, t)
					mean_mac_delays_per_user_1[-1].append(np.mean(user_mac_vals[last_i:i]))
					last_i = i

				# swarm 2
				user_mac_times = np.array(this_rep_mac_times_swarm_2[n])
				user_mac_vals = np.array(this_rep_mac_vals_swarm_2[n])
				mean_mac_delays_per_user_2.append([])
				last_i = 0
				for t in range(5, max_sim_time, 5):
					# find index closest to this simulation time
					i = find_nearest(user_mac_times, t)
					mean_mac_delays_per_user_2[-1].append(np.mean(user_mac_vals[last_i:i]))
					last_i = i
			mean_mac_delays_per_user_1 = np.array(mean_mac_delays_per_user_1)
			mean_mac_delays_per_user_2 = np.array(mean_mac_delays_per_user_2)
			for i in range(len(mean_mac_delays_per_user_1[0])):
				# compute mean at this window over all users
				avg_mac_delay_1[rep].append(np.mean(mean_mac_delays_per_user_1[:,i]))
				avg_mac_delay_2[rep].append(np.mean(mean_mac_delays_per_user_2[:,i]))
			if rep == 0:
				for t in range(5, max_sim_time, 5):
					avg_mac_times.append(t)

			bar_i += 1
			bar.update(bar_i)
		except FileNotFoundError as err:
			print(err)
	bar.finish()		

	# save to JSON
	json_data = {}	
	json_data[json_label_reps] = num_reps
	json_data[json_label_num_users_per_swarm] = num_users_per_swarm	
	json_data[json_label_num_active_neighbors_times_swarm_1] = np.array(num_active_neighbors_times_mat_1).tolist()
	json_data[json_label_num_active_neighbors_vals_swarm_1] = np.array(num_active_neighbors_vals_mat_1).tolist()
	json_data[json_label_num_active_neighbors_times_swarm_2] = np.array(num_active_neighbors_times_mat_2).tolist()
	json_data[json_label_num_active_neighbors_vals_swarm_2] = np.array(num_active_neighbors_vals_mat_2).tolist()

	json_data[json_label_num_packet_drop_times] = np.array(intervals_dropped_packets).tolist()
	json_data[json_label_num_packet_drop_vals] = np.array(sliding_window_dropped_packets).tolist()

	json_data[json_label_mac_delay_vals_1] = np.array(avg_mac_delay_1).tolist()
	json_data[json_label_mac_delay_vals_2] = np.array(avg_mac_delay_2).tolist()
	json_data[json_label_mac_window_times] = np.array(avg_mac_times).tolist()
	json_data[json_label_max_simtime] = max_simulation_time	
	json_data[json_label_num_time_slots] = num_time_slots	
				
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")  


def get_trajectories(t_max = 9999999):
	trajectories = []
	lines = []
	with open('../mobility_data/mobility_group1.txt') as f:
		lines += f.readlines()

	with open('../mobility_data/mobility_group2.txt') as f:
		lines += f.readlines()

	for ac, l in enumerate(lines):
		path = []
		entries = l.split(' ')
		i = 0
		while i < len(entries):
			t = float(entries[i])
			x = float(entries[i + 1])
			y = float(entries[i + 2])
			z = float(entries[i + 3])
			if t <= t_max:
				path.append({
					'id': f"ac_{ac}",
					't': t,
					'x': x,
					'y': y,
					'z': z
				})

			i += 4

		trajectories.append(path)
	return trajectories

def get_ac_position_at_time(trajectory, time):
	for i in range(1, len(trajectory)):
		if trajectory[i]['t'] >= time:
			prev = trajectory[i-1]
			nxt = trajectory[i]
			x = prev['x'] + (nxt['x']- prev['x'])  * (time - prev['t']) / (nxt['t']- prev['t'])
			y = prev['y'] + (nxt['y']- prev['y'])  * (time - prev['t']) / (nxt['t']- prev['t'])
			z = prev['z'] + (nxt['z']- prev['z'])  * (time - prev['t']) / (nxt['t']- prev['t'])
			return (x,y,z)

def get_left_cluster():
	x = list(np.array([-100, -90, -80, -51, -110]))
	y = list(np.array([-40, 50, -10, 10, 0]) * 0.9 -14)
	return (x,y)

def get_right_cluster():
	x = list(np.array([60, 70, 110, 120, 50]))
	y = list(np.array([-40, 50, -30, 10, 0]) * 0.9 -14)
	return (x,y)

def get_topology_1():
	(x1, y1) = get_left_cluster()
	(x2, y2) = get_right_cluster()
	return (x1+x2, y1+y1)

def get_topology_2():
	(x1, y1) = get_left_cluster()
	(x2, y2) = get_right_cluster()

	x = np.concatenate((np.array(x1) + 10, np.array(x2) -10))
	return (x, np.array(y1+y2))

def get_topology_3():
	(x1, y1) = get_left_cluster()
	(x2, y2) = get_right_cluster()

	x = np.concatenate((np.array(x1) + 70, np.array(x2) -70))
	return (x, np.array(y1+y2))

def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio  	

def plot_inset(ax, center, t, x=[], y=[], R=0, inset_width = 240, name=''):
		inset_height = 7
		aspect = get_aspect(ax)
		ax.plot([center[0] - inset_width/2 , center[0] + inset_width/2, center[0] + inset_width/2, center[0] - inset_width/2, center[0] - inset_width/2],[center[1] - inset_height/2, center[1] - inset_height/2, center[1] + inset_height/2, center[1] + inset_height/2, center[1] - inset_height/2, ], color= '#333', lw=0.7)

		ax.plot([center[0], t],[center[1]- inset_height / 2, -1], color='#333', lw=0.5)
		ax.plot([t, t],[center[1]- inset_height / 2, -1], '--', color='#333', lw=0.5)
		ax.plot([t, t],[center[1]+ inset_height / 2, 30], '--', color='#333', lw=0.5)

		for i in range(len(x)):
			for j in range(len(x)):
				if i!=j:
					x1 = x[i]
					x2 = x[j]
					y1 = y[i]
					y2 = y[j]
					dist = math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))
					if(dist <= R):
						ax.plot([center[0] + x1, center[0] + x2],[center[1] + y1 / aspect,center[1] + y2 / aspect], color='#333', lw=0.2)

		ax.scatter(center[0] + np.array(x), center[1] + np.array(y) / aspect, s=1.5, zorder=20, color='#333')
		ax.text(center[0], center[1]+ inset_height / 2, name, ha='center', va='center', fontsize=8, bbox=dict(pad=2, lw=0.5, fc='#fff', color='#333'))


def plot(json_filename, graph_filename_active_neighbors, time_slot_duration, graph_filename_delays):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		colors = ['xkcd:teal', 'xkcd:coral', 'xkcd:goldenrod', 'xkcd:maroon', 'xkcd:sea blue']
		# load JSON
		json_data = json.load(json_file)	
		num_users_per_swarm = json_data[json_label_num_users_per_swarm]		
		num_reps = np.array(json_data[json_label_reps])		
		
		num_active_neighbors_times_mat_1 = np.array(json_data[json_label_num_active_neighbors_times_swarm_1])		
		num_active_neighbors_vals_mat_1 = np.array(json_data[json_label_num_active_neighbors_vals_swarm_1])
		num_active_neighbors_times_mat_2 = np.array(json_data[json_label_num_active_neighbors_times_swarm_2])
		num_active_neighbors_vals_mat_2 = np.array(json_data[json_label_num_active_neighbors_vals_swarm_2])		

		intervals_dropped_packets = np.array(json_data[json_label_num_packet_drop_times])
		sliding_window_dropped_packets = np.array(json_data[json_label_num_packet_drop_vals])
		avg_dropped_packets = np.zeros((3, len(intervals_dropped_packets)))
		for i in range(len(intervals_dropped_packets)):
			avg_dropped_packets[:,i] = calculate_confidence_interval(sliding_window_dropped_packets[:, i])		

		max_simulation_time = json_data[json_label_max_simtime]
		# find average number of neighbors in half-a-second steps until maximum simulation time
		# no. of active users has shape (num_reps, num_users, time_steps)
		# so we'll compute the three-valued confidence interval (mean, minus, plus) for every point in time
		avg_neighbors_1 = np.zeros((3, max_simulation_time*2))
		avg_neighbors_2 = np.zeros((3, max_simulation_time*2))
		avg_neighbors_both = np.zeros((3, max_simulation_time*2))
		times_vec = np.zeros(max_simulation_time*2)		
		last_i_collisions = 0
		# for each point in time
		for i in range(max_simulation_time*2):
			t = 0 + i*0.5
			times_vec[i] = t
			# swarm 1
			# for every rep
			mean_avg_neighbors_per_rep = []
			for rep in range(num_reps):
				# find the index that is closest to the current moment in time t
				idx_1 = find_nearest(num_active_neighbors_times_mat_1[rep, 0], t)
				# compute the mean no. of neighbors for this repetition over all users at this moment in time
				mean_avg_neighbors_per_rep.append(np.mean(num_active_neighbors_vals_mat_1[rep, :, idx_1]))
			avg_neighbors_1[:,i] = calculate_confidence_interval(mean_avg_neighbors_per_rep)
			
			# swarm 2
			# for every rep
			mean_avg_neighbors_per_rep = []
			for rep in range(num_reps):
				# find the index that is closest to the current moment in time t
				idx_2 = find_nearest(num_active_neighbors_times_mat_2[rep, 0], t)
				# compute the mean no. of neighbors for this repetition over all users at this moment in time
				mean_avg_neighbors_per_rep.append(np.mean(num_active_neighbors_vals_mat_2[rep, :, idx_1]))
			avg_neighbors_2[:,i] = calculate_confidence_interval(mean_avg_neighbors_per_rep)

			# mean over both swarms
			avg_neighbors_both[:,i] = np.mean([avg_neighbors_1[:,i], avg_neighbors_2[:,i]])			

		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		
		# plot no of neighbors over simulation time
		fig, ax1 = plt.subplots()
		ax1.plot(times_vec, avg_neighbors_both[0,:], color=colors[0], linewidth=.75, alpha=1, linestyle=':')
		ax1.fill_between(times_vec, avg_neighbors_both[1,:], avg_neighbors_both[2,:], color=colors[0], alpha=.5)
		ax1.tick_params(axis='y', colors=colors[0])
		ax1.set_ylabel('avg no. of  discovered neighbors')

		scale = 6/10000
		R = 100

		trajectories = get_trajectories()
		t0 = 140
		(x,y) = get_topology_1()
		inset_center = (160, 8.5)
		plot_inset(ax1, inset_center, t0, x, y, R, 280, 'Disconnected')

		t1 = 270.55
		(x,y) = get_topology_2()
		inset_center = (460, 8.5)
		plot_inset(ax1, inset_center, t1, x, y, R, 260, 'First contact')

		t2 = 564
		(x,y) = get_topology_3()
		inset_center = (700, 8.5)
		plot_inset(ax1, inset_center, t2, x, y, R, 160, 'Full mesh')


		ax1.set_ylim([-1, 30])
		
		plt.xlabel('Simulation time $t$ [s]')
		ax1.set_xticks([0, t0, t1, 400, t2, 800])
		ax1.set_xticklabels(['0', '$t_0$', '$t_1$', '400', '$t_2$', '800'])
		# and the no. of dropped packets on a second y-axis
		ax2 = ax1.twinx()		
		ax2.plot(intervals_dropped_packets*time_slot_duration/1000, avg_dropped_packets[0,:], color=colors[1], linewidth=.75, alpha=1, linestyle=':')
		ax2.fill_between(intervals_dropped_packets*time_slot_duration/1000, avg_dropped_packets[1,:], avg_dropped_packets[2,:], color=colors[1], alpha=.5)
		ax2.set_ylabel('sliding window over packet losses')		
		ax2.tick_params(axis='y', colors=colors[1])		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width*1.75, settings.fig_height), forward=False)
		fig.savefig(graph_filename_active_neighbors, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_active_neighbors)
		plt.close()


		# plot average MAC delay over simulation time
		avg_mac_delay_1 = np.array(json_data[json_label_mac_delay_vals_1])
		avg_mac_delay_2 = np.array(json_data[json_label_mac_delay_vals_2])
		avg_mac_times = np.array(json_data[json_label_mac_window_times])
		# compute confidence interval over all delay values
		avg_mac_delay_both_swarms = np.zeros((3, len(avg_mac_times)))
		for t in range(len(avg_mac_times)):
			ci_1 = calculate_confidence_interval(avg_mac_delay_1[:,t])
			ci_2 = calculate_confidence_interval(avg_mac_delay_2[:,t])
			avg_mac_delay_both_swarms[:,t] = np.mean([ci_1, ci_2], axis=0)

		fig, ax1 = plt.subplots()
		ax1.plot(times_vec, avg_neighbors_both[0,:], color=colors[0], linewidth=.75, alpha=1, linestyle=':')
		ax1.fill_between(times_vec, avg_neighbors_both[1,:], avg_neighbors_both[2,:], color=colors[0], alpha=.5)
		ax1.tick_params(axis='y', colors=colors[0])
		ax1.set_ylabel('avg no. of discovered neighbors')
		plt.xlabel('Simulation time $t$ [s]')
		ax2 = ax1.twinx()		
		ax2.plot(avg_mac_times, avg_mac_delay_both_swarms[0,:] * time_slot_duration, color=colors[1], linewidth=.75, alpha=1, linestyle='-')
		ax2.fill_between(avg_mac_times, avg_mac_delay_both_swarms[1,:] * time_slot_duration, avg_mac_delay_both_swarms[2,:] * time_slot_duration, color=colors[1], alpha=.5)
		ax2.set_ylabel('sliding window over MAC delay [ms]', fontsize=8)
		ax2.tick_params(axis='y', colors=colors[1])		
		ax1.set_xticks([0, t0, t1, 400, t2, 800])
		ax1.set_xticklabels(['0', '$t_0$', '$t_1$', '400', '$t_2$', '800'])
		for t in [t0, t1, t2]:
			ax1.axvline(t, linestyle='--', color='k', linewidth=.5)

		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width*1.75, settings.fig_height), forward=False)
		fig.savefig(graph_filename_delays, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename_delays)    
		plt.close()


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='mobility')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')				
	parser.add_argument('--num_reps', type=int, help='Number of repetitions that should be considered.', default=1)
	parser.add_argument('--num_users_per_swarm', type=int, help='Number of users in either swarm.', default=15)	
	parser.add_argument('--time_slot_duration', type=int, help='Duration of a time slot in milliseconds.', default=24)
	parser.add_argument('--max_sim_time', type=int, help='Simulation time in seconds.', default=800)

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + "-rep" + str(args.num_reps)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename_num_active_neighbors = "_imgs/" + output_filename_base + "_num_active_neighbors.pdf"	
	graph_filename_delays = "_imgs/" + output_filename_base + "_delays.pdf"	
	if not args.no_parse:		
		parse(args.dir, args.num_reps, args.num_users_per_swarm, json_filename, args.time_slot_duration, args.max_sim_time)
	if not args.no_plot:
		plot(json_filename, graph_filename_num_active_neighbors, args.time_slot_duration, graph_filename_delays) 
    