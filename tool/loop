#!/bin/bash

for signal in RSG Radion HVTWZ
do
    echo $signal
    for analysis in resolved merged
    do 
	echo $analysis
	for channel in ggF VBF
	do
	    echo $channel
            python buildDataset.py -a $analysis -c $channel -s $signal # --drawPlots 1 > buildDataSet_r33-24_$analysis$channel$signal.txt
            #python splitDataset.py -a $analysis -c $channel -s $signal #--drawPlots 1 > splitDataSet_r33-24_$analysis$channel$signal.txt
            #python buildPDNN.py -a $analysis -c $channel -s $signal --doTest 0 > buildPDNN_r33-24_$analysis$channel$signal.txt #-n 48 -l 2 #> buildPDNN_r33-24_$analysis$channel$signal.txt
	done
    done
done
