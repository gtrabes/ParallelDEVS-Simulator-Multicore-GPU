#!/bin/bash

MODEL=CellDEVSBenchmark
COMPUTER=i7
FILENAME="./results/TimeGPU${MODEL}${COMPUTER}.csv"
ITERATIONS=10
THREADS=(2 4 8 16)
TIME=100
DIM=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

#sudo modprobe msr
mkdir -p results

ITERATION=0.0
TIME_SEQUENTIAL=0.0
TIME_NAIVE=0.0
TIME_DYNAMIC=0.0
TIME_STATIC=0.0
TIME_STATIC_NUMA=0.0
SIZE=1

for SIZE in ${DIM[@]}; do

#for i in `seq 1 20`; do

    TIME_SEQUENTIAL=0.0
    TIME_STATIC=0.0
    TIME_NAIVE_GPU=0.0
    TIME_GPU=0.0   
        
    echo "DIMENSION: $SIZE"
    echo "DIMENSION: $SIZE" >> $FILENAME
    echo "SEQUENTIAL"
    echo "SEQUENTIAL" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        ITERATION=$(bin/celldevs_benchmark_sequential $SIZE $TIME)
        echo $ITERATION
        TIME_SEQUENTIAL=$(echo "$TIME_SEQUENTIAL + $ITERATION" | bc -l)
#        TIME_SEQUENTIAL=$(echo $TIME_SEQUENTIAL + $TIME | bc)
#        TIME_SEQUENTIAL=$(($TIME_SEQUENTIAL + $ITERATOR))
    done
    
    TIME_SEQUENTIAL=$(echo "$TIME_SEQUENTIAL / $ITERATIONS" | bc -l)
    echo $TIME_SEQUENTIAL >> $FILENAME

    echo "STATIC PARALLEL WITH 8 THREADS"
    echo "STATIC PARALLEL WITH 8 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        ITERATION=$(bin/celldevs_benchmark_static_parallel_numa $SIZE $TIME 6)
        echo $ITERATION
        TIME_STATIC=$(echo "$TIME_STATIC + $ITERATION" | bc -l)
    done
    
    TIME_STATIC=$(echo "$TIME_STATIC / $ITERATIONS" | bc -l)
    echo $TIME_STATIC >> $FILENAME

    echo "NAIVE GPU"
    echo "NAIVE GPU" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        ITERATION=$(bin/celldevs_benchmark_naive_gpu $SIZE $TIME)
        echo $ITERATION
        TIME_NAIVE_GPU=$(echo "$TIME_NAIVE_GPU + $ITERATION" | bc -l)
    done
    
    TIME_NAIVE_GPU=$(echo "$TIME_NAIVE_GPU / $ITERATIONS" | bc -l)
    echo $TIME_NAIVE_GPU >> $FILENAME
    
    
    echo "GPU"
    echo "GPU" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        ITERATION=$(bin/celldevs_benchmark_gpu $SIZE $TIME)
        echo $ITERATION
        TIME_GPU=$(echo "$TIME_GPU + $ITERATION" | bc -l)
    done
    
    TIME_GPU=$(echo "$TIME_GPU / $ITERATIONS" | bc -l)
    echo $TIME_GPU >> $FILENAME
    
    
#    TWO=2
#    SIZE=$(echo "$SIZE * $TWO" | bc -l)
        
done
