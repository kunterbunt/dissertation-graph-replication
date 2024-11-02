# Overview
Hi there!
This is the main repository for all things MCSOTDMA-related.

The simulator is OMNeT++ v5.6.2, together with the inet framework v4.2.5.
All custom code is C++, mostly in libraries.
To track development on these, they are in their own repos.
A wrapper integrates the different components into OMNeT++.
All result evaluation and plotting is done with Python.

# Installation
tl;dr: open `install.sh` and adjust the number of cores `NUM_CPUS` to use for compiling; then run `./install.sh` if you're on Linux and `./install.sh mac` if you're on Mac.

Long version: all simulator components are bundled together into one installation script `install.sh`. 
Please run it, and pay attention to the output. 
For example, it downloads the OMNeT++ simulator of the right version from GitHub. If this download via `wget` doesn't succeed, for example because GitHub's servers are beyond their capacity (you'll get a `503` error), then later parts of the script fail. 
This is a very simple Bash script, so peek inside, it's easy to understand, and there's no error recovery built-in, so please debug using common sense!

# Navigation
## Code
All custom code lives under `omnetpp-5.6.2/workspace/<component>`.
Except `omnetpp-5.6.2/workspace/inet4`, that's the untouched `inet` framework that comes from the OMNeT++ developers.
Note that the `omnetpp-5.6.2/workspace` folder is only populated after installation, as each component is cloned from its repo.

## Simulations
All network scenarios live under `scenarios`.
At first, you won't want to touch these files, but to see the nitty and gritty of the simulation setups, this is where you'll find it.

**What you'll be mostly interested in lives under the `scenarios/results` folder.**
Please go there and read its readme after installation!