#!/bin/bash
for file in $1/*.vec; do	
	scavetool x $file -o $file.csv
done
