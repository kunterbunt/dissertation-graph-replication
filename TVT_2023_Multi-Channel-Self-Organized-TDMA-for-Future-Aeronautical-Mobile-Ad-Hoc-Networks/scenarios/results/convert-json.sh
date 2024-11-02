#!/bin/bash
for file in ./*.sca; do
	scavetool x $file -o $file.csv
done
