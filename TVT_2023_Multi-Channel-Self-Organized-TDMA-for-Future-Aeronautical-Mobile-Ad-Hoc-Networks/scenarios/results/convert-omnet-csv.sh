#!/bin/bash
for file in $1/*.sca; do	
	scavetool x $file -o $file.csv
done
