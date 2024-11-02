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


json_label_unicast_delay_vecs = ['unicast_mac_delay_vec_1', 'unicast_mac_delay_vec_2', 'unicast_mac_delay_vec_3', 'unicast_mac_delay_vec_4']
json_label_unicast_delay_vec_times = ['unicast_mac_delay_vec_time_1', 'unicast_mac_delay_vec_time_2', 'unicast_mac_delay_vec_time_3', 'unicast_mac_delay_vec_time_4']
json_label_max_num_pp_links = 'max_num_pp_links'

def calculate_confidence_interval(data, confidence):
	n = len(data)
	m = np.mean(data)
	std_dev = scipy.stats.sem(data)
	h = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
	return [m, m - h, m + h]

def parse(max_num_pp_links, dir, json_filename):			
	delay_mat_1 = None
	delay_time_1 = None
	delay_mat_2 = None
	delay_time_2 = None
	delay_mat_3 = None
	delay_time_3 = None
	delay_mat_4 = None
	delay_time_4 = None
	bar_max_i = len(max_num_pp_links)
	bar_i = 0
	print('parsing ' + str(bar_max_i) + ' result files')
	bar = progressbar.ProgressBar(max_value=bar_max_i, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	json_data = {}	
	for i in range(len(max_num_pp_links)):
		l = max_num_pp_links[i]
		try:				
			filename = dir + '/n=1,l=' + str(l) + '-#0.vec.csv'
			results = pd.read_csv(filename)
			delay_mat = results[(results.type=='vector') & (results.name=='mcsotdma_statistic_unicast_mac_delay:vector') & (results.module=='NW_LINK_ESTABLISHMENT.txNode[0].wlan[0].linkLayer')].vecvalue.tolist()			
			delay_time = results[(results.type=='vector') & (results.name=='mcsotdma_statistic_unicast_mac_delay:vector') & (results.module=='NW_LINK_ESTABLISHMENT.txNode[0].wlan[0].linkLayer')].vectime.tolist()
			if i == 0:
				delay_mat_1 = [float(val) for val in delay_mat[0].split(' ')]		
				delay_time_1 = [float(val) for val in delay_time[0].split(' ')]					
				json_data[json_label_unicast_delay_vecs[i]] = delay_mat_1
				json_data[json_label_unicast_delay_vec_times[i]] = delay_time_1
			elif i == 1:
				delay_mat_2 = [float(val) for val in delay_mat[0].split(' ')]		
				delay_time_2 = [float(val) for val in delay_time[0].split(' ')]					
				json_data[json_label_unicast_delay_vecs[i]] = delay_mat_2
				json_data[json_label_unicast_delay_vec_times[i]] = delay_time_2
			elif i == 2:
				delay_mat_3 = [float(val) for val in delay_mat[0].split(' ')]		
				delay_time_3 = [float(val) for val in delay_time[0].split(' ')]					
				json_data[json_label_unicast_delay_vecs[i]] = delay_mat_3
				json_data[json_label_unicast_delay_vec_times[i]] = delay_time_3
			elif i == 3:
				delay_mat_4 = [float(val) for val in delay_mat[0].split(' ')]		
				delay_time_4 = [float(val) for val in delay_time[0].split(' ')]					
				json_data[json_label_unicast_delay_vecs[i]] = delay_mat_4
				json_data[json_label_unicast_delay_vec_times[i]] = delay_time_4
			else:
				print('This script currently supports only up to four values for l.')
				exit(-1)
			bar_i += 1
			bar.update(bar_i)
		except FileNotFoundError as err:
			print(err)			
	bar.finish()	

	# Save to JSON.		
	json_data[json_label_max_num_pp_links] = max_num_pp_links
	with open(json_filename, 'w') as outfile:
		json.dump(json_data, outfile)
	print("Saved parsed results in '" + json_filename + "'.")    	


def plot(json_filename, graph_filename, time_slot_duration):
	"""
	Reads 'json_filename' and plots the values to 'graph_filename'.
	"""
	with open(json_filename) as json_file:
		# Load JSON
		json_data = json.load(json_file)		
		max_num_pp_links = np.array(json_data[json_label_max_num_pp_links])
		   				
		plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
		fig = plt.figure()				
		yticks = []
		colors = ['xkcd:teal', 'xkcd:coral', 'xkcd:goldenrod', 'xkcd:maroon', 'xkcd:sea blue']
		for i in range(len(max_num_pp_links)):
			l = max_num_pp_links[i]
			delay_mat = np.array(json_data[json_label_unicast_delay_vecs[i]])		
			delay_time = np.array(json_data[json_label_unicast_delay_vec_times[i]])		
			plt.scatter(delay_time[1:], delay_mat[1:]*time_slot_duration, label='$l=' + str(l) + '$', s=5, zorder=1, color=colors[i])  # due to the way statistic capturing is implemented in MCSOTDMA, the first value is zero and should be discarded
			plt.axhline(max(set(delay_mat), key=list(delay_mat).count)*time_slot_duration, color='k', linestyle='--', linewidth=0.75, zorder=0)
			yticks.append(max(set(delay_mat), key=list(delay_mat).count)*time_slot_duration)
			if i == len(max_num_pp_links) - 1:
				yticks.append(max(delay_mat)/2*time_slot_duration)
				yticks.append(max(delay_mat)*time_slot_duration)
		plt.yticks(yticks, fontsize=7)
		plt.xlabel('Simulation Time [s]')
		plt.ylabel('MAC Delay [ms]')		
		plt.legend(framealpha=0.0, prop={'size': 7}, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)		
		fig.tight_layout()
		settings.init()
		fig.set_size_inches((settings.fig_width, settings.fig_height*1.25), forward=False)
		fig.savefig(graph_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
		print("Graph saved to " + graph_filename)    
		plt.close()  


if __name__ == "__main__":        	
	parser = argparse.ArgumentParser(description='Parse OMNeT++-generated .csv result files and plot them.')
	parser.add_argument('--filename', type=str, help='Base filename for result and graphs files.', default='pp_voice')
	parser.add_argument('--dir', type=str, help='Directory path that contains the result files.', default='unspecified_directory')
	parser.add_argument('--no_parse', action='store_true', help='Whether *not* to parse result files.')		
	parser.add_argument('--no_plot', action='store_true', help='Whether *not* to plot predictions errors from JSON results.')					
	parser.add_argument('--time_slot_duration', type=float, help='Time slot duration.', default=24)
	parser.add_argument('--max_num_links', type=int, nargs='+', help='Maximum number of PP links that should be supported.', default=[1])	

	args = parser.parse_args()	
 
	expected_dirs = ['_imgs', '_data']
	for dir in expected_dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
		
	output_filename_base = args.filename + '-l=' + str(args.max_num_links)
	json_filename = "_data/" + output_filename_base + ".json"
	graph_filename = "_imgs/" + output_filename_base + ".pdf"		
	if not args.no_parse:		
		parse(args.max_num_links, args.dir, json_filename)
	if not args.no_plot:
		plot(json_filename, graph_filename, args.time_slot_duration) 
    