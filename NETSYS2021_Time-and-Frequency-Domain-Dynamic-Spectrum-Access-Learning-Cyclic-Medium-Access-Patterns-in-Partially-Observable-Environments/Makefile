#   Code to replicate results of the publication 
#   'Time- and Frequency-Domain Dynamic Spectrum Access: Learning Cyclic Medium Access Patterns in Partially Observable Environments' 
#   published at the Conference on Networked Systems (NetSys) 2021, Lübeck, Germany.
#   https://github.com/ComNetsHH/netsys2021-malene-code-release
#
#     Copyright (C) 2021  Institute of Communication Networks, 
#                         Hamburg University of Technology,
#                         https://www.tuhh.de/comnets
#               (C) 2021  Sebastian Lindner, 
#                         sebastian.lindner@tuhh.de
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Remove the # in the next line to make TensorFlow use your GPU.
ARGUMENTS = #--use_gpu --gpu_mem 1024

all: resource_utilization comparison_loss_over_time comparison_convergence_over_channels fully_observable_comparison_loss_over_time_matrix_input partially_observable_comparison_loss_over_time_matrix_input comparison_convergence_over_sample_length

.PHONY: all

use-x86:
	unlink Pipfile
	ln -s Pipfile_x86 Pipfile	
	pipenv update

use-mac-arm:
	unlink Pipfile
	ln -s Pipfile_Mac_ARM Pipfile	
	pipenv update

install:
	pipenv install

tests:
	pipenv run python -m test_env
	pipenv run python -m test_confidence_intervals

# Note that for all further targets, 50 repetitions and 5 splits may require substantial simulation time.
# For a quick check, consider reducing these numbers. n_reps must be divisible by split, however, to be able to compute confidence intervals on batch means, where batches are grouped through the 'split' argument.
# Check the individual Python files for more command-line parameters: for example, simulations are saved to .json files, and plotting reads these files. Therefore, you can simulate once, and then plot in a number of ways.
comparison_loss_over_time:
	pipenv run python -m comparison_loss_over_time $(ARGUMENTS) --n_channels 5 --n_neurons 128 --max_t 30000 --n_reps 50 --split 5

comparison_convergence_over_channels:
	pipenv run python -m comparison_convergence_over_channels $(ARGUMENTS) --n_channels 2 3 4 5 6 7 --n_neurons 128 --max_t 250000 --n_reps 50 --split 5

fully_observable_comparison_loss_over_time_matrix_input:
	pipenv run python -m fully_observable_comparison_loss_over_time_matrix_input $(ARGUMENTS) --n_channels 5 --n_neurons 128 --n_sample_length 16 --max_t 2500 --n_reps 50 --split 5

partially_observable_comparison_loss_over_time_matrix_input:
	pipenv run python -m partially_observable_comparison_loss_over_time_matrix_input $(ARGUMENTS) --n_channels 5 --n_neurons 128 --n_sample_length 64 --max_t 50000 --n_reps 50 --split 5

comparison_convergence_over_sample_length:
	pipenv run python -m comparison_convergence_over_sample_length $(ARGUMENTS) --n_channels 5 --n_neurons 128 --n_sample_length 1 6 16 32 64 96 --max_t 250000 --n_reps 50 --split 5

resource_utilization:
	pipenv run python -m resource_utilization