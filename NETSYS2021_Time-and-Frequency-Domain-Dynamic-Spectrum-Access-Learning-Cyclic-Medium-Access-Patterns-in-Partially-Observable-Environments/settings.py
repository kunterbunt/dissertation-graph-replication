import matplotlib as plt

def init():
	global fig_width 
	fig_width = 4.77*.475
	global fig_height 
	fig_height = 3.5*.55


def set_params():
	plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})