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


import unittest
import settings
from env import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TestDMEEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = DMEEnvironment(5)
        self.env.add(request_channel=0, response_channel=1, periodicity=5, offset=0)        
        self.env.add(request_channel=3, response_channel=4, periodicity=3, offset=1)

    def test_steps(self):
        t_max = 16
        M = np.zeros((t_max, self.env.n_channels))        
        for t in range(t_max):
            state = self.env.step()
            M[t] = state
            self.assertTrue(i == DMEEnvironment.IDLE_CHANNEL_STATE or i == DMEEnvironment.BUSY_CHANNEL_STATE for i in state)            
                
        plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
        fig = plt.figure()
        settings.init()        
        plt.imshow(np.transpose(-1*M), cmap='Greys', origin='lower')        
        plt.xlabel('Time step $t$')
        plt.ylabel('Channel $i$')
        ax = plt.gca()
        ax.set_yticks(range(self.env.n_channels))
        fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
        fig.tight_layout()
        pdf_filename = '_imgs/netsys_resource_utilization_fully_observable.pdf'
        fig.savefig(pdf_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)
        plt.close()
        print("Graph saved to " + pdf_filename)


class TestDMEEnvironmentPartiallyObservable(unittest.TestCase):
    def setUp(self):
        self.env = DMEEnvironmentPartiallyObservable(5)
        self.env.add(request_channel=0, response_channel=1, periodicity=5, offset=0)        
        self.env.add(request_channel=3, response_channel=4, periodicity=3, offset=1)

    def test_steps(self):
        t_max = 16
        M = np.zeros((t_max, self.env.n_channels))        
        for t in range(t_max):
            state = self.env.step(action=np.random.randint(0, self.env.n_channels))
            M[t] = state
            self.assertTrue(i == DMEEnvironment.IDLE_CHANNEL_STATE or i == DMEEnvironment.BUSY_CHANNEL_STATE or i == DMEEnvironmentPartiallyObservable.UNKNOWN_CHANNEL_STATE for i in state)
            self.assertTrue(np.sum(state) == 1 or np.sum(state) == -1)            
        
        plt.rcParams.update({
			'font.family': 'serif',
			"font.serif": 'Times',
			'font.size': 9,
			'text.usetex': True,
			'pgf.rcfonts': False
		})
        fig = plt.figure()
        settings.init()
        plt.imshow(np.transpose(-1*M), cmap='Greys', origin='lower')
        # cbar = plt.colorbar()
        # cbar.set_ticks([-1, 0, 1])
        # cbar.set_ticklabels(['idle', 'no info', 'busy'], fontsize=6)
        plt.xlabel('Time step $t$')
        plt.ylabel('Channel $i$')
        plt.xticks([0, 5, 10, 15])
        plt.yticks([0, 2, 4])
        ax = plt.gca()
        ax.set_yticks(range(self.env.n_channels))
        fig.set_size_inches((settings.fig_width, settings.fig_height), forward=False)
        fig.tight_layout()
        pdf_filename = '_imgs/netsys_resource_utilization_partially_observable.pdf'
        fig.savefig(pdf_filename, dpi=500, bbox_inches = 'tight', pad_inches = 0.01)
        plt.close()
        print("Graph saved to " + pdf_filename)


if __name__ == '__main__':
    Path("_imgs").mkdir(parents=True, exist_ok=True)		
    Path("_data").mkdir(parents=True, exist_ok=True)		
    unittest.main()
