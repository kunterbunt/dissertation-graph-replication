import inet.node.inet.AdhocHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.unitdisk.UnitDiskRadioMedium;
import inet.visualizer.contract.IIntegratedVisualizer;
import inet.node.inet.AdhocHost;
import inet.physicallayer.unitdisk.UnitDiskRadioMedium;
import intairnet_link_layer.LinkLayerLifecycleManager;

network NW
{
    parameters:
        int numNodes = default(2);
    submodules:
        lifecycleManager: LinkLayerLifecycleManager {}
        nodes[numNodes]: AdhocHost {}
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        radioMedium: UnitDiskRadioMedium {
            parameters:
                @display("p=100,200;is=s");
        }
}

network NW_TX_RX {
    parameters:
        int numTxNodes = default(2);
    submodules:
        lifecycleManager: LinkLayerLifecycleManager {}
        txNodes[numTxNodes]: AdhocHost {}
        rxNode: AdhocHost {}
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        radioMedium: UnitDiskRadioMedium {
            parameters:
                @display("p=100,200;is=s");
        }
}

network NW_LINK_ESTABLISHMENT {
    parameters:
        int numNodes = default(2);
        int numBroadcastNodes = default(0);
    submodules:
        lifecycleManager: LinkLayerLifecycleManager {}
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        radioMedium: UnitDiskRadioMedium {
            parameters:
                @display("p=100,200;is=s");
        }
        rxNode[numNodes]: AdhocHost {}
        txNode[numNodes]: AdhocHost {}
        broadcastNode[numBroadcastNodes]: AdhocHost {}
}