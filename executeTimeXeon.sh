#!/bin/bash

MODEL=CellDEVSBenchmark
COMPUTER=i7
FILENAME="./results/Time${MODEL}${COMPUTER}.csv"
ITERATIONS=10
THREADS=(2 4 8 16 32)
TIME=100
DIM=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

#sudo modprobe msr
mkdir -p results

for d in ${DIM[@]}; do
    echo "DIMENSION: $d"
    echo "DIMENSION: $d" >> $FILENAME
    echo "SEQUENTIAL"
    echo "SEQUENTIAL" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        bin/celldevs_benchmark_sequential $d $TIME >> $FILENAME
    done
    
    echo "NAIVE PARALLEL WITH 16 THREADS"
    echo "NAIVE PARALLEL WITH 16 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        bin/celldevs_benchmark_naive_parallel $d $TIME $t >> $FILENAME
    done

    echo "DYNAMIC PARALLEL WITH 16 THREADS"
    echo "DYNAMIC PARALLEL WITH 16 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        bin/celldevs_benchmark_dynamic_parallel $d $TIME $t >> $FILENAME
    done
    
    for t in ${THREADS[@]}; do
        echo "STATIC PARALLEL WITH ${t} THREADS"
        echo "STATIC PARALLEL WITH ${t} THREADS" >> $FILENAME
        for i in `seq 1 $ITERATIONS`; do
           echo "ITERATION: ${i}"
           bin/celldevs_benchmark_static_parallel $d $TIME $t >> $FILENAME
        done
    done
        
done
