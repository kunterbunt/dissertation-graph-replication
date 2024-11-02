#!/bin/bash
sbatch s_broadcast-voice.sh
sbatch s_mac-aloha.sh
sbatch s_mac-mcsotdma-05-slotadv.sh
sbatch s_mac-mcsotdma-05.sh
sbatch s_mac-mcsotdma-75-slotadv.sh
sbatch s_mac-mcsotdma-75.sh
sbatch s_mac-mcsotdma-95-slotadv.sh
sbatch s_mac-mcsotdma-95.sh
sbatch s_mac-mcsotdma-opt-slotadv.sh
sbatch s_mac-mcsotdma-opt.sh
sbatch s_mac-sotdma.sh
sbatch s_pp-link-estbl.sh
sbatch s_pp-voice.sh
