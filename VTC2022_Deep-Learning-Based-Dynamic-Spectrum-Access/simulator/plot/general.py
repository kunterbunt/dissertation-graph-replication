import os

import matplotlib as mpl


def setup():
	mpl.rcParams.update({
		'font.family': 'serif',
		'font.size': 10,
		'text.usetex': True,
		'text.latex.preamble': r'\usepackage{amsfonts}',
		'pgf.rcfonts': False
	});

	if not os.path.exists('output'):
		os.makedirs('output')
