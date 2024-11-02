.PHONY: all

all: markov sequential_loss sequential_predictions sequential_channel_access_pattern

just_plot_all: just_plot_markov just_plot_sequential_loss just_plot_sequential_predictions sequential_channel_access_pattern

use-x86:
	unlink Pipfile
	ln -s Pipfile_x86 Pipfile
	unlink Pipfile.lock
	ln -s Pipfile_x86.lock Pipfile.lock
	pipenv update

use-mac-arm:
	unlink Pipfile
	ln -s Pipfile_Mac_ARM Pipfile
	unlink Pipfile.lock
	ln -s Pipfile_Mac_ARM.lock Pipfile.lock
	pipenv update

install:
	pipenv install

markov:	
	pipenv run python -m simulator_markov --t 2500

just_plot_markov:
	pipenv run python -m simulator_markov --t 2500 --no_sim_single --no_sim_avg


sequential_loss:
	pipenv run python -m simulator_sequential --rep 20 --split 4 --t 160 --no_sim_predictions --no_plot_predictions --no_plot_channel_access

just_plot_sequential_loss:
	pipenv run python -m simulator_sequential --rep 20 --split 4 --t 160 --no_sim_predictions --no_sim_loss --no_plot_predictions --no_plot_channel_access


sequential_predictions:	
	pipenv run python -m simulator_sequential --c 16 --p 0.75 --no_sim_loss --no_plot_loss --no_plot_channel_access

just_plot_sequential_predictions:
	pipenv run python -m simulator_sequential --c 16 --p 0.75 --no_sim_loss --no_plot_channel_access --no_sim_predictions

sequential_channel_access_pattern:
	pipenv run python -m simulator_sequential --t 65 --p 1 --no_sim_loss --no_plot_loss --no_sim_predictions --no_plot_predictions; \
	pipenv run python -m simulator_sequential --t 65 --p .7 --no_sim_loss --no_plot_loss --no_sim_predictions --no_plot_predictions