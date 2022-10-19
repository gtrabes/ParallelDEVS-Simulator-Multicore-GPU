#!/bin/bash

MODEL=CellDEVSBenchmark
COMPUTER=i7
FILENAME="./results/Time${MODEL}${COMPUTER}.csv"
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

#for d in ${DIM[@]}; do

for i in `seq 1 20`; do

    TIME_SEQUENTIAL=0.0
    TIME_NAIVE=0.0
    TIME_DYNAMIC=0.0
    TIME_STATIC=0.0
    TIME_STATIC_NUMA=0.0
    
        
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
    
    echo "NAIVE PARALLEL WITH 8 THREADS"
    echo "NAIVE PARALLEL WITH 8 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        ITERATION=$(bin/celldevs_benchmark_naive_parallel $SIZE $TIME 8)
        echo $ITERATION
        TIME_NAIVE=$(echo "$TIME_NAIVE + $ITERATION" | bc -l)
    done

    TIME_NAIVE=$(echo "$TIME_NAIVE / $ITERATIONS" | bc -l)
    echo $TIME_NAIVE >> $FILENAME

    echo "DYNAMIC PARALLEL WITH 8 THREADS"
    echo "DYNAMIC PARALLEL WITH 8 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        ITERATION=$(bin/celldevs_benchmark_dynamic_parallel $SIZE $TIME 8)
        echo $ITERATION
        TIME_DYNAMIC=$(echo "$TIME_DYNAMIC + $ITERATION" | bc -l)
    done
    
    TIME_DYNAMIC=$(echo "$TIME_DYNAMIC / $ITERATIONS" | bc -l)
    echo $TIME_DYNAMIC >> $FILENAME
        
    for t in ${THREADS[@]}; do
        TIME_STATIC=0 
        echo "STATIC PARALLEL NUMA WITH ${t} THREADS"
        echo "STATIC PARALLEL NUMA WITH ${t} THREADS" >> $FILENAME
        for i in `seq 1 $ITERATIONS`; do
           echo "ITERATION: ${i}"
           ITERATION=$(bin/celldevs_benchmark_static_parallel_numa $SIZE $TIME $t)
           echo $ITERATION
           TIME_STATIC=$(echo "$TIME_STATIC + $ITERATION" | bc -l)
        done
        
        TIME_STATIC=$(echo "$TIME_STATIC / $ITERATIONS" | bc -l)
        echo $TIME_STATIC >> $FILENAME        
    done
    
    TWO=2
    SIZE=$(echo "$SIZE * $TWO" | bc -l)
        
done
