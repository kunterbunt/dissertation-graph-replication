#!/bin/bash

# This simulator is a slightly modified version (where graphs are formatted differently to match the dissertation) of the public code release from the research paper
# Multi-Channel Self-Organized TDMA for Future Aeronautical Mobile Ad-Hoc Networks
# IEEE Transactions on Vehicular Technology
# DOI: 10.1109/TVT.2024.3380316
# URL: https://ieeexplore.ieee.org/document/10477589/

# The L-Band Digital Aeronautical Communications System (LDACS) simulator provides an installation script for the simulator that downloads the other simulator components, defines simulation scenarios and provides result evaluation and graph creation.
# Copyright (C) 2024  Sebastian Lindner, Institute of Communication Networks, Hamburg University of Technology, Hamburg, Germany

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

NUM_CPUS=10  # set to your no. of cores

LOC_OMNET=https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6.2/omnetpp-5.6.2-src-linux.tgz
LOC_OMNET_MAC=https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6.2/omnetpp-5.6.2-src-macosx.tgz
LOC_INET=https://github.com/eltayebmusab/inet/archive/refs/tags/v4.2.5.tar.gz
LOC_GLUE=https://zenodo.org/records/10073046/files/ComNetsHH/ldacs_glue-v1.1.zip
LOC_RLC=https://zenodo.org/records/10073044/files/ComNetsHH/ldacs_rlc-v1.1.zip
LOC_ARQ=https://zenodo.org/records/10073043/files/ComNetsHH/ldacs_arq-v1.1.zip
LOC_MCSOTDMA=https://zenodo.org/records/8082927/files/ComNetsHH/ldacs_mcsotdma-v1.0.zip
LOC_RADIO=https://zenodo.org/records/8082925/files/ComNetsHH/ldacs_tracebased_channel_model-v1.0.zip
LOC_APP=https://zenodo.org/records/8082929/files/ComNetsHH/ldacs_tracebased_app-v1.0.zip
LOC_WRAPPER=https://zenodo.org/records/10073049/files/ComNetsHH/ldacs_wrapper-v1.1.zip
LOC_GPSR=https://zenodo.org/records/8082919/files/ComNetsHH/ldacs_gpsr-v1.0.zip
# Download OMNeT++ v5.6.2, unpack and go to directory.
echo -n "Downloading OMNeT++ "
if [ $1 = "mac" ]; then
	echo "for Mac"
	wget $LOC_OMNET_MAC
	echo -e "\n\nUnpacking OMNeT++"
	tar -xvzf omnetpp-5.6.2-src-macosx.tgz
	rm omnetpp-5.6.2-src-macosx.tgz
else
	echo "for Linux"
	wget $LOC_OMNET
	echo -e "\n\nUnpacking OMNeT++"
	tar -xvzf omnetpp-5.6.2-src-linux.tgz
	rm omnetpp-5.6.2-src-linux.tgz
fi
echo -e "\n\nCompiling OMNeT++"
cd omnetpp-5.6.2/
# Set PATH, configure and build.
WORKDIR=$(pwd)
export PATH=${WORKDIR}/bin:$PATH
./configure CC=gcc CXX=g++ WITH_OSG=no WITH_OSGEARTH=no WITH_QTENV=no
make -j$NUM_CPUS MODE=release base
#make -j$NUM_CPUS MODE=debug base

# Download INET, unpack and go to directory.
echo -e "\n\nDownloading INET"
mkdir workspace
cd workspace
wget $LOC_INET
echo -e "\n\nUnpacking INET"
tar -xvzf v4.2.5.tar.gz 
rm -R v4.2.5.tar.gz
mv inet-4.2.5/ inet4
cd inet4/
# Build.
echo -e "\n\nCompiling INET"
make makefiles
make -j$NUM_CPUS MODE=release
#make -j$NUM_CPUS MODE=debug
cd ..

# Compile GLUE
echo -e "\n\nDownloading GLUE lib"
mkdir ldacs_glue
wget $LOC_GLUE
umask 000
unzip ldacs_glue-v1.1.zip -d tmp_extract
mv tmp_extract/*/* ldacs_glue/
mv tmp_extract/*/.* ldacs_glue/
rm -r tmp_extract
cd ldacs_glue
mkdir cmake-build-release
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$NUM_CPUS intairnet_linklayer_glue
# cd ..
# mkdir cmake-build-debug
# cd cmake-build-debug
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# make -j$NUM_CPUS intairnet_linklayer_glue
cd ../..

# Compile RLC
echo -e "\n\nDownloading and compiling RLC lib"
mkdir ldacs_rlc
wget $LOC_RLC
unzip ldacs_rlc-v1.1.zip -d tmp_extract
mv tmp_extract/*/* ldacs_rlc/
mv tmp_extract/*/.* ldacs_rlc/
rm -r tmp_extract
cd ldacs_rlc
ln -s ../ldacs_glue/ glue-lib-headers
mkdir cmake-build-release
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$NUM_CPUS tuhh_intairnet_rlc
# cd ..
# mkdir cmake-build-debug
# cd cmake-build-debug
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# make -j$NUM_CPUS tuhh_intairnet_rlc
cd ../..

# Compile ARQ
echo -e "\n\nDownloading and compiling ARQ lib"
mkdir ldacs_arq
wget $LOC_ARQ
unzip ldacs_arq-v1.1.zip -d tmp_extract
mv tmp_extract/*/* ldacs_arq/
mv tmp_extract/*/.* ldacs_arq/
rm -r tmp_extract
cd ldacs_arq/dev
ln -s ../../ldacs_glue/ glue-lib-headers
mkdir cmake-build-release
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$NUM_CPUS tuhh_intairnet_arq
# cd ..
# mkdir cmake-build-debug
# cd cmake-build-debug
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# make -j$NUM_CPUS tuhh_intairnet_arq
cd ../../..

# Compile MCSOTDMA
echo -e "\n\nDownloading and compiling MCSOTDMA lib"
ldacs_mcsotdma-v1.0.zip
mkdir ldacs_mcsotdma
wget $LOC_MCSOTDMA
unzip ldacs_mcsotdma-v1.0.zip -d tmp_extract
mv tmp_extract/*/* ldacs_mcsotdma/
mv tmp_extract/*/.* ldacs_mcsotdma/
rm -r tmp_extract
cd ldacs_mcsotdma
ln -s ../ldacs_glue/ glue-lib-headers
mkdir cmake-build-release
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$NUM_CPUS tuhh_intairnet_mc-sotdma
# cd ..
# mkdir cmake-build-debug
# cd cmake-build-debug
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# make -j$NUM_CPUS tuhh_intairnet_mc-sotdma
cd ../..

# Clone radio
echo -e "\n\nDownloading channel model"
mkdir ldacs_tracebased_channel_model
wget $LOC_RADIO
unzip ldacs_tracebased_channel_model-v1.0.zip -d tmp_extract
mv tmp_extract/*/* ldacs_tracebased_channel_model/
mv tmp_extract/*/.* ldacs_tracebased_channel_model/
rm -r tmp_extract
cd ldacs_tracebased_channel_model/src
opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET
#opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET_dbg
# make MODE=debug -j$NUM_CPUS
make MODE=release -j$NUM_CPUS
cd ../..

# Clone traceBacedApp
echo -e "\n\nDownloading UdpTracedBasedApp"
mkdir ldacs_tracebased_app
wget $LOC_APP
unzip ldacs_tracebased_app-v1.0.zip -d tmp_extract
mv tmp_extract/*/* ldacs_tracebased_app/
mv tmp_extract/*/.* ldacs_tracebased_app/
rm -r tmp_extract
cd ldacs_tracebased_app/src
opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET
make MODE=release -j$NUM_CPUS

cd ../..

# Clone gpsr
echo -e "\n\nDownloading GPSR modified"
mkdir ldacs_gpsr
wget $LOC_GPSR
unzip ldacs_gpsr-v1.0.zip -d tmp_extract
mv tmp_extract/*/* ldacs_gpsr/
mv tmp_extract/*/.* ldacs_gpsr/
rm -r tmp_extract
cd ldacs_gpsr
cd src
opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET
# make MODE=debug -j$NUM_CPUS
make MODE=release -j$NUM_CPUS

cd ../..

# Clone wrapper
echo -e "\n\nDownloading OMNET++ wrapper"
mkdir ldacs_wrapper
wget $LOC_WRAPPER
unzip ldacs_wrapper-v1.1.zip -d tmp_extract
mv tmp_extract/*/* ldacs_wrapper/
mv tmp_extract/*/.* ldacs_wrapper/
rm -r tmp_extract
cd ldacs_wrapper/intairnet-link-layer
echo "Compiling simulation binary"
ln -s ../../ldacs_glue/cmake-build-release ./glue-lib
ln -s ../../ldacs_glue ./glue-lib-headers
ln -s ../../ldacs_rlc/cmake-build-release ./ldacs_rlc
ln -s ../../ldacs_rlc ./ldacs_rlc-headers
ln -s ../../ldacs_mcsotdma/cmake-build-release ./ldacs_mcsotdma
ln -s ../../ldacs_mcsotdma ./ldacs_mcsotdma-headers
ln -s ../../ldacs_arq/dev/cmake-build-release ./ldacs_arq
ln -s ../../ldacs_arq/dev ./ldacs_arq-headers
cd src
opp_makemake -f --deep -O out -KINET4_PROJ=../../../inet4 -DINET_IMPORT -I../../../inet4 -I../../../ldacs_tracebased_app/src -I../../../ldacs_tracebased_channel_model/src -I../glue-lib-headers -I../ldacs_rlc-headers -I../ldacs_arq-headers -I../ldacs_mcsotdma-headers -I../../../ldacs_gpsr/src -I. -I../../../inet4/src -L../../../ldacs_gpsr/out/gcc-release/src/ -L../../../inet4/src -L../../../ldacs_tracebased_app/out/gcc-release/src/ -L../../../ldacs_tracebased_channel_model/out/gcc-release/src/ -L../glue-lib -L../ldacs_rlc -L../ldacs_arq -L../ldacs_mcsotdma -lINET -lldacs_tracebased_app -lldacs_tracebased_channel_model -lintairnet_linklayer_glue -ltuhh_intairnet_rlc -ltuhh_intairnet_arq -ltuhh_intairnet_mc-sotdma -lldacs_gpsr
#opp_makemake -f --deep -O out -KINET4_PROJ=../../../inet4 -DINET_IMPORT -I../../../inet4 -I../../../ldacs_tracebased_app/src -I../../../ldacs_tracebased_channel_model/src -I../glue-lib-headers -I../ldacs_rlc-headers -I../ldacs_arq-headers -I../ldacs_mcsotdma-headers -I../../../ldacs_gpsr/src -I. -I../../../inet4/src -L../../../inet4/src -L../../../ldacs_gpsr/out/gcc-release/src/ -L../../../ldacs_tracebased_app/out/gcc-release/src/ -L../../../ldacs_tracebased_channel_model/out/gcc-debug/src/ -L../glue-lib -L../ldacs_rlc -L../ldacs_arq -L../ldacs_mcsotdma -lINET_dbg -lldacs_tracebased_app_dbg -lldacs_tracebased_channel_model_dbg -lintairnet_linklayer_glue -ltuhh_intairnet_rlc -ltuhh_intairnet_arq -ltuhh_intairnet_mc-sotdma -lldacs_gpsr
# make MODE=debug -j$NUM_CPUS
make MODE=release -j$NUM_CPUS

cd ../../..

cd ../../scenarios/results
echo -e "\n\nInstall python packages into local pipenv environment"
make install-python-env
echo -e "\n\nAll done! Try it by running the following commands:\ncd scenarios/results\nmake sanity-check\nThis should start simulations and create graphs in the scenarios/results/_imgs/ directory."
echo -e "Please make sure that omnetpp-5.6.2/bin is in your PATH!"
