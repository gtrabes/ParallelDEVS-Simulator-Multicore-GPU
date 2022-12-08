#!/bin/bash

MODEL=CellDEVSSIR
COMPUTER=Xeon16Cores
FILENAME="./results/Time${MODEL}${COMPUTER}.csv"
ITERATIONS=30
THREADS=(2 4 8 16 32)
TIME=500
DIM=(317 448 548 633 708 775 837 895 949 1000)

#sudo modprobe msr
mkdir -p results

for d in ${DIM[@]}; do

    TIME_SEQUENTIAL=0.0
    TIME_NAIVE=0.0
    TIME_DYNAMIC=0.0
    TIME_STATIC=0.0
    TIME_STATIC_NUMA=0.0

    echo "DIMENSION: $d"
    echo "DIMENSION: $d" >> $FILENAME
    echo "SEQUENTIAL"
    echo "SEQUENTIAL" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        #bin/celldevs_benchmark_sequential $d $TIME >> $FILENAME
        ITERATION=$(bin/celldevs_sir_sequential $d $TIME)
        echo $ITERATION
        TIME_SEQUENTIAL=$(echo "$TIME_SEQUENTIAL + $ITERATION" | bc -l)
    done

    TIME_SEQUENTIAL=$(echo "$TIME_SEQUENTIAL / $ITERATIONS" | bc -l)
    echo $TIME_SEQUENTIAL >> $FILENAME

    echo "NAIVE PARALLEL WITH 32 THREADS"
    echo "NAIVE PARALLEL WITH 32 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        #bin/celldevs_benchmark_naive_parallel $d $TIME 32 >> $FILENAME
        ITERATION=$(bin/celldevs_sir_naive_parallel $d $TIME 32)
        echo $ITERATION
        TIME_NAIVE=$(echo "$TIME_NAIVE + $ITERATION" | bc -l)
    done

    TIME_NAIVE=$(echo "$TIME_NAIVE / $ITERATIONS" | bc -l)
    echo $TIME_NAIVE >> $FILENAME

    echo "DYNAMIC PARALLEL WITH 32 THREADS"
    echo "DYNAMIC PARALLEL WITH 32 THREADS" >> $FILENAME
    for i in `seq 1 $ITERATIONS`; do
        echo "ITERATION: ${i}"
        #bin/celldevs_benchmark_dynamic_parallel $d $TIME 32 >> $FILENAME
        ITERATION=$(bin/celldevs_sir_dynamic_parallel $d $TIME 32)
        echo $ITERATION
        TIME_DYNAMIC=$(echo "$TIME_DYNAMIC + $ITERATION" | bc -l)
    done
    
    TIME_DYNAMIC=$(echo "$TIME_DYNAMIC / $ITERATIONS" | bc -l)
    echo $TIME_DYNAMIC >> $FILENAME
    
    for t in ${THREADS[@]}; do
        TIME_STATIC=0
        echo "STATIC PARALLEL WITH ${t} THREADS"
        echo "STATIC PARALLEL WITH ${t} THREADS" >> $FILENAME
        for i in `seq 1 $ITERATIONS`; do
           echo "ITERATION: ${i}"
           #bin/celldevs_benchmark_static_parallel $d $TIME $t >> $FILENAME
           ITERATION=$(bin/celldevs_sir_static_parallel_numa $d $TIME $t)
           echo $ITERATION
           TIME_STATIC=$(echo "$TIME_STATIC + $ITERATION" | bc -l)
        done
        
        TIME_STATIC=$(echo "$TIME_STATIC / $ITERATIONS" | bc -l)
        echo $TIME_STATIC >> $FILENAME
        
    done
        

    for t in ${THREADS[@]}; do
        TIME_STATIC_NUMA=0
        echo "STATIC NUMA PARALLEL WITH ${t} THREADS"
        echo "STATIC NUMA PARALLEL WITH ${t} THREADS" >> $FILENAME
        for i in `seq 1 $ITERATIONS`; do
           echo "ITERATION: ${i}"
           #bin/celldevs_benchmark_static_parallel_numa $d $TIME $t >> $FILENAME
           ITERATION=$(bin/celldevs_sir_static_parallel_numa $d $TIME $t)
           echo $ITERATION
           TIME_STATIC_NUMA=$(echo "$TIME_STATIC_NUMA + $ITERATION" | bc -l)
        done
        
        TIME_STATIC_NUMA=$(echo "$TIME_STATIC_NUMA / $ITERATIONS" | bc -l)
        echo $TIME_STATIC_NUMA >> $FILENAME
        
    done
        
done
