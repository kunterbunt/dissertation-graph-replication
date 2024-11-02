//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.ManetRouter;
import inet.node.inet.WirelessHost;
import inet.physicallayer.unitdisk.UnitDiskRadioMedium;
import inet.visualizer.integrated.IntegratedMultiVisualizer;
import inet.visualizer.contract.IIntegratedVisualizer;
import inet.common.scenario.ScenarioManager;
import intairnet_link_layer.LinkLayerLifecycleManager;
import inet.node.ethernet.EtherSwitch;
import ned.DatarateChannel;

//package intairnet_link_layer.simmodel;

network mobility
{	
    parameters:
        int numAircraftsGroup1 = default(0);
        int numAircraftsGroup2 = default(0);
        @display("bgb=10000000,10000000;bgl=2");
        @figure[title](type=label; pos=0,-1; anchor=sw; color=darkblue);
    submodules:
        scenarioManager: ScenarioManager;
        lifecycleManager: LinkLayerLifecycleManager {}
        radioMedium: UnitDiskRadioMedium {
            @display("p=580,275");
        }
        visualizer: IntegratedMultiVisualizer {
           @display("p=100,200");
        }
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }

        aircraft_group1[numAircraftsGroup1]: ManetRouter {
            @display("i=aircraft");
        }
        aircraft_group2[numAircraftsGroup2]: ManetRouter {
            @display("i=aircraft");
        }

}
