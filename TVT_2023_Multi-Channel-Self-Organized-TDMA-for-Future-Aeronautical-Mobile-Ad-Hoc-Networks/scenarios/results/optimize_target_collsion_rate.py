# from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
from math import e

def k(n, q):
	return 1. / (1. - np.power((1.-q), (1./float(n))))

def d(k):
	return k/2.

if __name__ == "__main__":	
	q_vec = np.linspace(0.05, .95, 10000)
	n = 30
	graph_filename_delay = '_imgs/optimal_reception_rate_n-' + str(n) + '__delay.pdf'
	graph_filename_delay_with_rtx = '_imgs/optimal_reception_rate_n-' + str(n) + '__delay_rtx.pdf'
	graph_filename_delay_n1000 = '_imgs/optimal_reception_rate_n-1000__delay.pdf'	
	time_slot_duration = 24
	d_vec = [d(k(n,q))*time_slot_duration for q in q_vec]

	plt.rcParams.update({
		'font.family': 'serif',
		"font.serif": 'Times',
		'font.size': 7,
		'text.usetex': True,
		'pgf.rcfonts': False
	})

	d_rtx_vec = [(1./(1-q)) * d(k(n,q)) * time_slot_duration for q in q_vec]
	min_val, min_index = min((val, i) for (i, val) in enumerate(d_rtx_vec))	
	optimal_q = q_vec[min_index]
	fig = plt.figure()
	plt.plot(q_vec, d_rtx_vec, label='expected delay')
	plt.xlabel('target collision probability $q$')
	plt.ylabel('expected delay until reception [ms]')
	plt.axvline(q_vec[min_index], color='k', linewidth=.75, linestyle='--', label='minimum')
	plt.gca().legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=2)		
	plt.xticks([0.05, 0.25, optimal_q, .95])
	fig.tight_layout()
	fig.set_size_inches((4.7*.35, 3.5*.5), forward=False)
	fig.savefig(graph_filename_delay_with_rtx, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
	print("Graph saved to " + graph_filename_delay_with_rtx)    
	plt.close()  

	fig = plt.figure()
	plt.plot(q_vec, d_vec, label='expected delay')
	plt.xlabel('target collision probability $q$')
	plt.ylabel('expected delay [ms]')	
	plt.axhline(d_vec[min_index], color='k', linewidth=.75, linestyle='--', label='$q=' + str("{:.3f}".format(q_vec[min_index])) + '$')
	plt.axvline(q_vec[min_index], color='k', linewidth=.75, linestyle='--')
	plt.gca().legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=2)			
	plt.yticks([d_vec[min_index], max(d_vec)])
	plt.xticks([0.05, 0.25, optimal_q, .95])
	fig.tight_layout()
	fig.set_size_inches((4.7*.35, 3.5*.5), forward=False)
	fig.savefig(graph_filename_delay, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
	print("Graph saved to " + graph_filename_delay)    
	plt.close()  		

	n = 1000
	d_vec = [d(k(n,q))*time_slot_duration for q in q_vec]
	fig = plt.figure()
	plt.plot(q_vec[500:], d_vec[500:], label='expected delay')
	plt.xlabel('target collision probability $q$')	
	plt.axhline(d_vec[min_index], color='k', linewidth=.75, linestyle='--', label='$q=' + str("{:.3f}".format(q_vec[min_index])) + '$')
	plt.axhline(min(d_vec), color='k', linewidth=.5, linestyle=':', label='$q=' + str("{:.3f}".format(max(q_vec))) + '$')
	plt.axvline(q_vec[min_index], color='k', linewidth=.75, linestyle='--')
	plt.gca().legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=3)			
	plt.yticks([min(d_vec), d_vec[min_index]])
	plt.xticks([0.05, 0.25, optimal_q, .95])
	fig.tight_layout()
	fig.set_size_inches((4.7*.5, 3.5*.5), forward=False)
	fig.savefig(graph_filename_delay_n1000, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
	print("Graph saved to " + graph_filename_delay_n1000)    
	plt.close()  		
	
	n_vec = range(1, 100)
	q = 1-1/e
	time_slot_duration = 6
	d_vec_6 = [d(k(n,q))*time_slot_duration for n in n_vec]
	time_slot_duration = 12
	d_vec_12 = [d(k(n,q))*time_slot_duration for n in n_vec]
	time_slot_duration = 24
	d_vec_24 = [d(k(n,q))*time_slot_duration for n in n_vec]
	fig = plt.figure()
	plt.plot(n_vec, d_vec_6, label='6ms slots')
	plt.plot(n_vec, d_vec_12, label='12ms slots')
	plt.plot(n_vec, d_vec_24, label='24ms slots')
	plt.ylabel('expected MAC delay [ms]')
	plt.xlabel('number of neighbors $n$')		
	plt.gca().legend()
	# plt.gca().legend(framealpha=0.0, prop={'size': 8}, loc='upper center', bbox_to_anchor=(.5, 1.5), ncol=1)				
	fig.tight_layout()
	fig.set_size_inches((4.7*.5, 3.5*.5), forward=False)
	graph_filename_delay_n1_100 = '_imgs/optimal_reception_rate_n-1_100__delay.pdf'	
	fig.savefig(graph_filename_delay_n1_100, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)		
	print("Graph saved to " + graph_filename_delay_n1_100)    
	plt.close()  		