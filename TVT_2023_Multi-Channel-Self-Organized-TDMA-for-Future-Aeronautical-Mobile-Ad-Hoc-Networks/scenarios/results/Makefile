update-arq:
	cd ../../omnetpp-5.6.2/workspace/ldacs_arq; git pull

update-rlc:
	cd ../../omnetpp-5.6.2/workspace/ldacs_rlc; git pull

update-glue:
	cd ../../omnetpp-5.6.2/workspace/ldacs_glue; git pull

update-wrapper:
	cd ../../omnetpp-5.6.2/workspace/ldacs_wrapper; git pull

update-radio:
	cd ../../omnetpp-5.6.2/workspace/ldacs_tracebased_channel_model; git pull

update-mcsotdma:
	cd ../../omnetpp-5.6.2/workspace/ldacs_mcsotdma; git pull

update-ldacs_gpsr:
	cd ../../omnetpp-5.6.2/workspace/ldacs_gpsr; git pull
	
update-traceBasedUdpApp:
	cd ../../omnetpp-5.6.2/workspace/ldacs_tracebased_app; git pull

update-sim-model:
	git pull

# UPDATE pulls all changes from the currently selected branches of all repositories
update: update-arq update-rlc update-glue update-wrapper update-radio update-mcsotdma update-ldacs_gpsr update-traceBasedUdpApp update-sim-model

# create a virtual Python environment to install all packages that are required for plotting results
install-python-env:
	pipenv install

NUM_CPUS = 6 # number of CPUs to allow OMNeT++ to use
SUFFIX = 

# BUILD-DEBUG compiles all libraries and then the simulation binary in DEBUG mode
build-debug:	
	cd ../../omnetpp-5.6.2/workspace/; \
	echo -e "\nARQ"; \
	cd ldacs_arq/dev/cmake-build-debug; cmake -DCMAKE_BUILD_TYPE=Debug ..; make -j$(NUM_CPUS) tuhh_intairnet_arq; cd ../../..; \
	echo -e "\nRLC"; \
	cd ldacs_rlc/cmake-build-debug; cmake -DCMAKE_BUILD_TYPE=Debug ..; make -j$(NUM_CPUS) tuhh_intairnet_rlc; cd ../..; \
	echo -e "\nGLUE"; \
	cd ldacs_glue/cmake-build-debug; cmake -DCMAKE_BUILD_TYPE=Debug ..; make -j$(NUM_CPUS) intairnet_linklayer_glue; cd ../..; \
	echo -e "\nMCSOTDMA"; \
	cd ldacs_mcsotdma/cmake-build-debug; cmake -DCMAKE_BUILD_TYPE=Debug ..; make -j$(NUM_CPUS) tuhh_intairnet_mc-sotdma; cd ../..; \
	echo -e "\ntracebasedapp"; \
	cd ldacs_tracebased_app/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nRADIO"; \
	cd ldacs_tracebased_channel_model/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nGPSR"; \
	cd ldacs_gpsr/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \	
	echo -e "\nSIMULATION BINARY"; \
	cd ldacs_wrapper/intairnet-link-layer/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../../inet4 -DINET_IMPORT -I../../../inet4 -I../../../ldacs_tracebased_app/src -I../../../ldacs_tracebased_channel_model/src -I../../../ldacs_gpsr/src -I../glue-lib-headers -I../ldacs_rlc-headers -I../ldacs_arq-headers -I../ldacs_mcsotdma-headers -I. -I../../../inet4/src -L../../../inet4/src -L../../../ldacs_tracebased_app/out/gcc-debug/src/ -L../../../ldacs_tracebased_channel_model/out/gcc-debug/src/ -L../../../ldacs_gpsr/out/gcc-debug/src/ -L../glue-lib -L../ldacs_rlc -L../ldacs_arq -L../ldacs_mcsotdma -lINET_dbg -lldacs_tracebased_app_dbg -lldacs_tracebased_channel_model_dbg -lldacs_gpsr_dbg -lintairnet_linklayer_glue -ltuhh_intairnet_rlc -ltuhh_intairnet_arq -ltuhh_intairnet_mc-sotdma; make -j$(NUM_CPUS) MODE=debug; \
	echo -e "\nDon't forget to compile OMNeT++ itself and INET in debug mode, too! This target only compiles the LDACS libraries."

# BUILD-RELEASE compiles all libraries and then the simulation binary in RELEASE mode
build-release:
	cd ../../omnetpp-5.6.2/workspace/; \
	echo -e "\nARQ"; \
	cd ldacs_arq/dev/cmake-build-release; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j$(NUM_CPUS) tuhh_intairnet_arq; cd ../../..; \
	echo -e "\nRLC"; \
	cd ldacs_rlc/cmake-build-release; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j$(NUM_CPUS) tuhh_intairnet_rlc; cd ../..; \
	echo -e "\nGLUE"; \
	cd ldacs_glue/cmake-build-release; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j$(NUM_CPUS) intairnet_linklayer_glue; cd ../..; \
	echo -e "\nMCSOTDMA"; \
	cd ldacs_mcsotdma/cmake-build-release; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j$(NUM_CPUS) tuhh_intairnet_mc-sotdma; cd ../..; \
	echo -e "\ntracebasedapp"; \
	cd ldacs_tracebased_app/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET; make -j$(NUM_CPUS) MODE=release; cd ../..; \
	echo -e "\nRADIO"; \
	cd ldacs_tracebased_channel_model/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET; make -j$(NUM_CPUS) MODE=release; cd ../..; \
	echo -e "\nGPSR"; \
	cd ldacs_gpsr/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET; make -j$(NUM_CPUS) MODE=release; cd ../..; \
	echo -e "\nSIMULATION BINARY"; \
	cd ldacs_wrapper/intairnet-link-layer/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../../inet4 -DINET_IMPORT -I../../../inet4 -I../../../ldacs_tracebased_app/src -I../../../ldacs_tracebased_channel_model/src -I../../../ldacs_gpsr/src -I../glue-lib-headers -I../ldacs_rlc-headers -I../ldacs_arq-headers -I../ldacs_mcsotdma-headers -I. -I../../../inet4/src -L../../../inet4/src -L../../../ldacs_tracebased_app/out/gcc-release/src/ -L../../../ldacs_tracebased_channel_model/out/gcc-release/src/ -L../../../ldacs_gpsr/out/gcc-release/src/ -L../glue-lib -L../ldacs_rlc -L../ldacs_arq -L../ldacs_mcsotdma -lINET -lldacs_tracebased_app -lldacs_tracebased_channel_model -lldacs_gpsr -lintairnet_linklayer_glue -ltuhh_intairnet_rlc -ltuhh_intairnet_arq -ltuhh_intairnet_mc-sotdma; make -j$(NUM_CPUS) MODE=release

# GLOBAL VARIABLES  can be shared among all targets
# default binary in RELEASE mode
SIM_BINARY = ../../omnetpp-5.6.2/workspace/ldacs_wrapper/intairnet-link-layer/out/gcc-release/src/intairnet-link-layer
# debug binary
SIM_BINARY_DBG = ../../omnetpp-5.6.2/workspace/ldacs_wrapper/intairnet-link-layer/out/gcc-debug/src/intairnet-link-layer_dbg

# GLOBAL VARIABLES  can be shared among all targets
# default binary in RELEASE mode
SIM_BINARY = ../../omnetpp-5.6.2/workspace/ldacs_wrapper/intairnet-link-layer/out/gcc-release/src/intairnet-link-layer
# debug binary
SIM_BINARY_DBG = ../../omnetpp-5.6.2/workspace/ldacs_wrapper/intairnet-link-layer/out/gcc-debug/src/intairnet-link-layer_dbg

# include .ned files
INCL = -n ../:./:../../omnetpp-5.6.2/workspace/inet4/src/:../../omnetpp-5.6.2/workspace/ldacs_wrapper/intairnet-link-layer/src/:../../omnetpp-5.6.2/workspace/ldacs_tracebased_channel_model/src:../../omnetpp-5.6.2/workspace/ldacs_tracebased_app/src:../../omnetpp-5.6.2/workspace/ldacs_gpsr/src

# QUICK SANITY CHECK SIMULATION
# This should quickly finish and tests the toolchain.
# After running, four graphs should've been created in the _imgs/ directory.
SANITY_CHECK_SEND_INT_START = 100	# First sending interval (ms)
SANITY_CHECK_SEND_INT_END = 200		# Last sending interval (ms)
SANITY_CHECK_SEND_INT_INCR = 50		# Increment sending intervals by this value (ms)
SANITY_CHECK_NUM_REPS = 2			# Number of repetitions
SANITY_CHECK_NUM_USERS = 5 6		# Number of users (one simulation with each)
sanity-check:	
	rm -rf SanityCheck/; \
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../sanity_check.ini -c SanityCheck $(INCL); \
	./convert-omnet-csv.sh simresults/SanityCheck; \
	pipenv run python3 eval.py --dir simresults/SanityCheck --filename sanity_check --sending_interval_start $(SANITY_CHECK_SEND_INT_START) --sending_interval_stop $(SANITY_CHECK_SEND_INT_END) --sending_interval_step $(SANITY_CHECK_SEND_INT_INCR) --num_reps $(SANITY_CHECK_NUM_REPS) --n $(SANITY_CHECK_NUM_USERS) $(SUFFIX)

# RANDOMIZED SLOTTED ALOHA
SH_MAC_RA_NUM_REPS = 20
SH_MAC_RA_NUM_USERS = 1 5 10 20 30 40 50 60
SH_MAC_RA_NUM_CANDIDATE_SLOTS = 2 5 10 40
SH_MAC_RA_TIME_SLOT_DURATION = 24 #ms
SH_MAC_RA_CONFIGNAME = RandomizedSlottedAloha
SH_MAC_YLIM1 = 20
SH_MAC_YLIM2 = 20000

all: sh-mac-aloha sh-mac-sotdma sh-mac-mcsotdma-95 sh-mac-mcsotdma-95-slotadv sh-mac-mcsotdma-75 sh-mac-mcsotdma-75-slotadv sh-mac-mcsotdma-opt sh-mac-mcsotdma-opt-slotadv sh-mac-mcsotdma-05 sh-mac-mcsotdma-05-slotadv sh-mac-mcsotdma-all_plot sh-mac-mcsotdma-all_plot sh-mac-opt pp-link-estbl pp-voice broadcast-voice sh-sfg-comparison mobility mobility-comparison_plot delay_optimal_plot 
	
sh-mac-aloha_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_RA_CONFIGNAME) $(INCL)

sh-mac-aloha_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_RA_CONFIGNAME)

sh-mac-aloha_plot:
	pipenv run python3 eval_random_access.py --dir simresults/$(SH_MAC_RA_CONFIGNAME) --filename mac_ra --n $(SH_MAC_RA_NUM_USERS) --c $(SH_MAC_RA_NUM_CANDIDATE_SLOTS) --num_reps $(SH_MAC_RA_NUM_REPS) --time_slot_duration $(SH_MAC_RA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-aloha: sh-mac-aloha_sim sh-mac-aloha_parse sh-mac-aloha_plot

# SOTDMA
SH_MAC_SOTDMA_CONFIGNAME = SOTDMA
sh-mac-sotdma_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_SOTDMA_CONFIGNAME) $(INCL)

sh-mac-sotdma_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_SOTDMA_CONFIGNAME)

sh-mac-sotdma_plot:
	pipenv run python3 eval_random_access.py --dir simresults/$(SH_MAC_SOTDMA_CONFIGNAME) --filename mac_sotdma --n $(SH_MAC_RA_NUM_USERS) --c $(SH_MAC_RA_NUM_CANDIDATE_SLOTS) --num_reps $(SH_MAC_RA_NUM_REPS) --time_slot_duration $(SH_MAC_RA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-sotdma: sh-mac-sotdma_sim sh-mac-sotdma_parse sh-mac-sotdma_plot

# MCSOTDMA 
SH_MAC_MCSTODMA_NUM_REPS = 20
SH_MAC_MCSTODMA_NUM_USERS = 1 5 10 20 30 40 50 60
SH_MAC_MCSTODMA_TIME_SLOT_DURATION = 24 #ms

# 95% target reception probability = 5% target collision probability
# without slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_95 = MCSOTDMA95  
SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_95 = 95
sh-mac-mcsotdma-95_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_95) $(INCL)

sh-mac-mcsotdma-95_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_95)

sh-mac-mcsotdma-95_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_95) --filename mac_mcsotdma_95 --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_95) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-95: sh-mac-mcsotdma-95_sim sh-mac-mcsotdma-95_parse sh-mac-mcsotdma-95_plot

sh-mac-mcsotdma-95_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_95)/*

# with slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95 = MCSOTDMA95SlotAdv  
sh-mac-mcsotdma-95-slotadv_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95) $(INCL)

sh-mac-mcsotdma-95-slotadv_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95)

sh-mac-mcsotdma-95-slotadv_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95) --filename mac_mcsotdma_95-slotadv --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_95) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-95-slotadv: sh-mac-mcsotdma-95-slotadv_sim sh-mac-mcsotdma-95-slotadv_parse sh-mac-mcsotdma-95-slotadv_plot

sh-mac-mcsotdma-95-slotadv_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95)/*


# 75% target reception probability = 25% target collision probability
# without slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_75 = MCSOTDMA75
SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_75 = 75
sh-mac-mcsotdma-75_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_75) $(INCL)

sh-mac-mcsotdma-75_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_75)

sh-mac-mcsotdma-75_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_75) --filename mac_mcsotdma_75 --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_75) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-75: sh-mac-mcsotdma-75_sim sh-mac-mcsotdma-75_parse sh-mac-mcsotdma-75_plot

sh-mac-mcsotdma-75_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_75)/*

# with slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_75 = MCSOTDMA75SlotAdv  
sh-mac-mcsotdma-75-slotadv_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_75) $(INCL)

sh-mac-mcsotdma-75-slotadv_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_75)

sh-mac-mcsotdma-75-slotadv_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_75) --filename mac_mcsotdma_75-slotadv --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_75) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-75-slotadv: sh-mac-mcsotdma-75-slotadv_sim sh-mac-mcsotdma-75-slotadv_parse sh-mac-mcsotdma-75-slotadv_plot

sh-mac-mcsotdma-75-slotadv_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95)/*


# 1/e target reception probability = 1-1/e target collision probability
# without slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_OPT = MCSOTDMAOptimal
SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_OPT = 37
sh-mac-mcsotdma-opt_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_OPT) $(INCL)

sh-mac-mcsotdma-opt_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_OPT)

sh-mac-mcsotdma-opt_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_OPT) --filename mac_mcsotdma_opt --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_OPT) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-opt: sh-mac-mcsotdma-opt_sim sh-mac-mcsotdma-opt_parse sh-mac-mcsotdma-opt_plot

sh-mac-mcsotdma-opt_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_OPT)/*

# with slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_OPT = MCSOTDMAOptimalSlotAdv  
sh-mac-mcsotdma-opt-slotadv_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_OPT) $(INCL)

sh-mac-mcsotdma-opt-slotadv_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_OPT)

sh-mac-mcsotdma-opt-slotadv_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_OPT) --filename mac_mcsotdma_opt-slotadv --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_OPT) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-opt-slotadv: sh-mac-mcsotdma-opt-slotadv_sim sh-mac-mcsotdma-opt-slotadv_parse sh-mac-mcsotdma-opt-slotadv_plot

sh-mac-mcsotdma-opt-slotadv_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95)/*


# 5% target reception probability = 95% target collision probability
# without slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_05 = MCSOTDMA05
SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_05 = 5
sh-mac-mcsotdma-05_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_05) $(INCL)

sh-mac-mcsotdma-05_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_05)

sh-mac-mcsotdma-05_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_05) --filename mac_mcsotdma_05 --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_05) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-05: sh-mac-mcsotdma-05_sim sh-mac-mcsotdma-05_parse sh-mac-mcsotdma-05_plot

# with slot advertisement
SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_05 = MCSOTDMA05SlotAdv
sh-mac-mcsotdma-05-slotadv_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_05) $(INCL)

sh-mac-mcsotdma-05-slotadv_parse:
	./convert-omnet-csv.sh simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_05)

sh-mac-mcsotdma-05-slotadv_plot:
	pipenv run python3 eval_randomized_slotted_aloha.py --dir simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_05) --filename mac_mcsotdma_05-slotadv --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rate $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_05) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) --ylim1 $(SH_MAC_YLIM1) --ylim2 $(SH_MAC_YLIM2) $(SUFFIX)

sh-mac-mcsotdma-05-slotadv: sh-mac-mcsotdma-05-slotadv_sim sh-mac-mcsotdma-05-slotadv_parse sh-mac-mcsotdma-05-slotadv_plot

sh-mac-mcsotdma-05-slotadv_clean:
	rm simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95)/*


# AGGREGATE RESULTS INTO ONE GRAPH
sh-mac-mcsotdma-all_plot:
	pipenv run python3 eval_randomized_slotted_aloha_aggregate.py --dirs simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_95) simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_75) simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_OPT) simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_05) --filename mac_mcsotdma_all --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rates $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_95) $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_75) $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_OPT) $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_05) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) $(SUFFIX)

sh-mac-mcsotdma-slotadv-all_plot:
	pipenv run python3 eval_randomized_slotted_aloha_aggregate.py --dirs simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_95) simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_75) simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_OPT) simresults/$(SH_MAC_MCSTODMA_CONFIGNAME_SLOTADV_05) --filename mac_mcsotdma_all_slotadv --n $(SH_MAC_MCSTODMA_NUM_USERS) --num_reps $(SH_MAC_MCSTODMA_NUM_REPS) --target_reception_rates $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_95) $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_75) $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_OPT) $(SH_MAC_MCSTODMA_TARGET_RECEPTION_RATE_05) --time_slot_duration $(SH_MAC_MCSTODMA_TIME_SLOT_DURATION) $(SUFFIX)	

sh-mac-mcsotdma-all_clean: sh-mac-mcsotdma-95_clean sh-mac-mcsotdma-75_clean sh-mac-mcsotdma-50_clean sh-mac-mcsotdma-opt_clean	

# OPTIMAL COLLISION PROBABILITY
sh-mac-opt: 
	pipenv run python3 optimize_target_collsion_rate.py 


# LINK ESTABLISHMENT TIMES
PP_LINK_ESTBL_CONFIGNAME = PPLinkEstablishmentTimes
PP_LINK_ESTBL_NUM_REPS = 50
PP_LINK_ESTBL_NUM_PAIRS = 1
PP_LINK_ESTBL_NUM_BROADCAST_USERS = 0 5 10 15 20 25 30 35 40 45 50 55 60
PP_LINK_ESTBL_MAX_NUM_LINKS = 1 2 3 4
pp-link-estbl_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../point_to_point_channel.ini -c $(PP_LINK_ESTBL_CONFIGNAME) $(INCL)

pp-link-estbl_parse:
	./convert-omnet-csv.sh simresults/$(PP_LINK_ESTBL_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(PP_LINK_ESTBL_CONFIGNAME)

pp-link-estbl_plot:
	pipenv run python3 eval_link_estbl_time.py --dir simresults/$(PP_LINK_ESTBL_CONFIGNAME) --filename link_estbl_time --num_reps $(PP_LINK_ESTBL_NUM_REPS) --n $(PP_LINK_ESTBL_NUM_PAIRS) --m $(PP_LINK_ESTBL_NUM_BROADCAST_USERS) --max_num_links $(PP_LINK_ESTBL_MAX_NUM_LINKS) $(SUFFIX)

pp-link-estbl: pp-link-estbl_sim pp-link-estbl_parse pp-link-estbl_plot

pp-link-estbl_clean:
	rm simresults/$(PP_LINK_ESTBL_CONFIGNAME)/*


# PP VOICE
PP_VOICE_NAIVE_CONFIGNAME = PPVoice
PP_VOICE_TIME_SLOT_DURATION = 24
PP_VOICE_NUM_CONTENDERS = 10
PP_VOICE_MAX_NUM_LINKS = 1 2 3 4
pp-voice_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../point_to_point_channel.ini -c $(PP_VOICE_NAIVE_CONFIGNAME) $(INCL)

pp-voice_parse:
	./convert-omnet-csv.sh simresults/$(PP_VOICE_NAIVE_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(PP_VOICE_NAIVE_CONFIGNAME)

pp-voice_plot:
	pipenv run python3 eval_pp_voice.py --dir simresults/$(PP_VOICE_NAIVE_CONFIGNAME) --filename pp_voice_n_$(PP_VOICE_NUM_CONTENDERS) --time_slot_duration $(PP_VOICE_TIME_SLOT_DURATION) --max_num_links $(PP_VOICE_MAX_NUM_LINKS) $(SUFFIX)

pp-voice: pp-voice_sim pp-voice_parse pp-voice_plot

pp-voice_clean:
	rm simresults/$(PP_VOICE_NAIVE_CONFIGNAME)/*


# BROADCAST VOICE
BROADCAST_VOICE_CONFIGNAME = BroadcastVoice
BROADCAST_VOICE_NUM_REPS = 10
BROADCAST_VOICE_NUM_USERS = 1 5 10 20 30 40 50 60
BROADCAST_VOICE_CHANNEL_ERROR = 0 0.05 0.10 0.15
BROADCAST_VOICE_TIME_SLOT_DURATION = 24
broadcast-voice_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(BROADCAST_VOICE_CONFIGNAME) $(INCL)

broadcast-voice_parse:
	./convert-omnet-csv.sh simresults/$(BROADCAST_VOICE_CONFIGNAME)

broadcast-voice_plot:
	pipenv run python3 eval_broadcast_voice.py --dir simresults/$(BROADCAST_VOICE_CONFIGNAME) --filename broadcast_voice --num_users $(BROADCAST_VOICE_NUM_USERS) --channel_errors $(BROADCAST_VOICE_CHANNEL_ERROR) --num_reps $(BROADCAST_VOICE_NUM_REPS) --target 0.367 $(SUFFIX)

broadcast-voice: broadcast-voice_sim broadcast-voice_parse broadcast-voice_plot

broadcast-voice_clean:
	rm simresults/$(BROADCAST_VOICE_CONFIGNAME)/*


# SIGNAL FLOW GRAPH COMPARISON
SFG_COMPARISON_NUM_REPS = 10
SFG_COMPARISON_NUM_USERS_1 = 10
SFG_COMPARISON_SFG_CSV_FILE_1 = sfg_mat_10.csv
SFG_COMPARISON_NUM_USERS_2 = 30
SFG_COMPARISON_SFG_CSV_FILE_2 = sfg_mat_30.csv
SFG_COMPARISON_TIME_SLOT_DURATION = 24 #ms
SFG_COMPARISON_CONFIGNAME = SignalFlowGraphComparison
sh-sfg-comparison_sim:	
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../shared_channel.ini -c $(SFG_COMPARISON_CONFIGNAME) $(INCL)

sh-sfg-comparison_parse:
	./convert-omnet-csv.sh simresults/$(SFG_COMPARISON_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(SFG_COMPARISON_CONFIGNAME)

sh-sfg-comparison_plot_1:
	echo "Run Matlab script generate_plot__sh_channel_access_1.m in Matlab to generate the CSV files for analytical Signal Flow Graph evaluation if it doesn't exist."; \
	pipenv run python3 eval_sh_sfg_comparison.py --sfg_csv_file $(SFG_COMPARISON_SFG_CSV_FILE_1) --dir simresults/$(SFG_COMPARISON_CONFIGNAME) --filename sfg_comparison --n $(SFG_COMPARISON_NUM_USERS_1) --num_reps $(SFG_COMPARISON_NUM_REPS) --time_slot_duration $(SFG_COMPARISON_TIME_SLOT_DURATION) $(SUFFIX)

sh-sfg-comparison_plot_2:
	echo "Run Matlab script generate_plot__sh_channel_access_2.m in Matlab to generate the CSV files for analytical Signal Flow Graph evaluation if it doesn't exist."; \
	pipenv run python3 eval_sh_sfg_comparison.py --sfg_csv_file $(SFG_COMPARISON_SFG_CSV_FILE_2) --dir simresults/$(SFG_COMPARISON_CONFIGNAME) --filename sfg_comparison --n $(SFG_COMPARISON_NUM_USERS_2) --num_reps $(SFG_COMPARISON_NUM_REPS) --time_slot_duration $(SFG_COMPARISON_TIME_SLOT_DURATION) $(SUFFIX)

sh-sfg-comparison: sh-sfg-comparison_sim sh-sfg-comparison_parse sh-sfg-comparison_plot_1 sh-sfg-comparison_plot_2

sh-sfg-comparison_clean:
	rm simresults/$(SFG_COMPARISON_CONFIGNAME)/*

# MOBILITY SCENARIO
MOBILITY_NUM_REPS = 20
MOBILITY_NUM_USERS_GROUP_1 = 15
MOBILITY_NUM_USERS_GROUP_2 = 15
MOBILITY_TIME_SLOT_DURATION = 24
MOBILITY_CONFIGNAME = NoApps
mobility_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../mobility.ini -c $(MOBILITY_CONFIGNAME) $(INCL)

mobility_sim-qtenv:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../mobility.ini -c $(MOBILITY_CONFIGNAME) $(INCL) -u Qtenv 

mobility_parse:
	./convert-omnet-csv.sh simresults/$(MOBILITY_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(MOBILITY_CONFIGNAME)

mobility_plot:
	pipenv run python3 eval_mobility.py --dir simresults/$(MOBILITY_CONFIGNAME) --filename mobility --num_reps $(MOBILITY_NUM_REPS) --time_slot_duration $(MOBILITY_TIME_SLOT_DURATION) $(SUFFIX)

mobility: mobility_sim mobility_parse mobility_plot

mobility_clean:
	rm simresults/$(MOBILITY_CONFIGNAME)/*


# DELAY / PACKET LOSS COMPARISON IN MOBILITY SCENARIO BETWEEN RS-ALOHA, SOTDMA AND MCSOTDMA
MOBILITY_COMPARISON_RSALOHA_CONFIGNAME = MobilityRSALOHA
MOBILITY_COMPARISON_SOTDMA_CONFIGNAME = MobilitySOTDMA
MOBILITY_COMPARISON_MCSOTDMA_CONFIGNAME = MobilityMCSOTDMA
mobility-comparison-rsaloha_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../mobility.ini -c $(MOBILITY_COMPARISON_RSALOHA_CONFIGNAME) $(INCL)

mobility-comparison-sotdma_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../mobility.ini -c $(MOBILITY_COMPARISON_SOTDMA_CONFIGNAME) $(INCL)

mobility-comparison-mcsotdma_sim:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY) ../mobility.ini -c $(MOBILITY_COMPARISON_MCSOTDMA_CONFIGNAME) $(INCL)

mobility-comparison-rsaloha_parse:
	./convert-omnet-csv.sh simresults/$(MOBILITY_COMPARISON_RSALOHA_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(MOBILITY_COMPARISON_RSALOHA_CONFIGNAME)

mobility-comparison-sotdma_parse:
	./convert-omnet-csv.sh simresults/$(MOBILITY_COMPARISON_SOTDMA_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(MOBILITY_COMPARISON_SOTDMA_CONFIGNAME)

mobility-comparison-mcsotdma_parse:
	./convert-omnet-csv.sh simresults/$(MOBILITY_COMPARISON_MCSOTDMA_CONFIGNAME); \
	./convert-vec-omnet-csv.sh simresults/$(MOBILITY_COMPARISON_MCSOTDMA_CONFIGNAME)

mobility-comparison_plot:
	pipenv run python3 eval_mobility_comparison.py --dir_mcsotdma simresults/$(MOBILITY_COMPARISON_MCSOTDMA_CONFIGNAME) --dir_sotdma simresults/$(MOBILITY_COMPARISON_SOTDMA_CONFIGNAME) --dir_rsaloha simresults/$(MOBILITY_COMPARISON_RSALOHA_CONFIGNAME) --num_reps $(MOBILITY_NUM_REPS) --time_slot_duration $(MOBILITY_TIME_SLOT_DURATION) $(SUFFIX)


# NUMERICAL "PROOF" OF DELAY-OPTIMAL CONFIGURATION
delay_optimal_plot:
	pipenv run python3 plot_delay_optimal_fn.py