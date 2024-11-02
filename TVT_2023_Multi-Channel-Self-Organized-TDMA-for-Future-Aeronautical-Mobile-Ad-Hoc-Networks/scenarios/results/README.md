# Makefile
Consult the `Makefile` to see which simulation targets are available.
Generally, each target will consist of subtargets, one for simulation `_sim`, parsing `_parse`, plotting `_plot` and cleaning `_clean`.
Then, one target without such a suffix will chain them together, like `sh-mac-sotdma: sh-mac-sotdma_sim sh-mac-sotdma_parse sh-mac-sotdma_plot`, which calls them one after another.
So, to run SOTDMA simulations, execute `make sh-mac-sotdma` and wait for everything to finish.

Study the targets in detail to see what steps are performed along the way.
You'll find that simulation scenarios are defined in OMNeT++ `.ini` files, which reside one folder above.
These are passed into the OMNeT++ executable to simulate.
Also one folder above, you'll find `.ned` network configuration files.

To parse simulation result files that are generated under `simuresults/<configname>/` folders, we convert to CSV using `convert-omnet-csv.sh` for scalar and `convert-vec-omnet-csv.sh` for vector result files.
Parsed CSV files will have the original name with a `.csv` suffix.

To parse CSV files, we rely on Python.
Each target will call some Python script in the `_plot` subtarget, which reads the (very large) CSV files, extracts the relevant statistics, and writes them into a `_data/<configname>.json` file.
These contain just the statistics relevant for plotting, and nothing else.
Finally, the Python script will read these JSON files to plot.
The reason for this splitting is the amount of data that is parsed: when you just want to adjust the label of a graph, it becomes cumbersome to wait seconds (or minutes!) for the parsing to finish; reading those JSON files is done much, much quicker.

So, in summary, the toolchain is: Makefile -> execute OMNeT++ Simulator -> parse OMNeT++ result files into CSV -> parse CSV into JSON -> read JSON and generate PDF graph.
Since each step is easily executed in isolation, you can modify and test without having to execute all the other steps.

# Parallel simulations
The `Makefile` passes along the `NUM_CPUS` variable that you can provide when calling a target.
So, for example, to run the SOTDMA simulations in parallel using 14 cores, run `make sh-mac-sotdma NUM_CPUS=14`.

# SLURM Script Files
All `s_<something>.sh` files are SLURM script files, which allow the execution of our simulations on a SLURM-controlled server cluster. 
If you happen to have access to a SLURM system, feel free to modify and use these files.

# Simulation Time
Please be aware that conducting all simulations requires substantial simulation time.
We have run these simulations in a highly parallelized fashion on the High Performance Computing (HPC) Cluster of the Hamburg University of Technology.
The HPC has a large number of servers available, equipped with Intel Xeon E5-2680 CPUs of different versions, ranging from 2.4GHz to 3.3GHz.
When each target is run by 14 cores in parallel (where applicable), allow a simulation time of up to three hours per target.
Scale according to the hardware that you have available, and bring a lot of coffee (or let it run over night).

# tl;dr (too long; didn't read)
The ten graphs in the paper are associated to ten SLURM files.
Open each file, copy&paste the `make` commands therein, and execute them to generate all graphs in the paper.