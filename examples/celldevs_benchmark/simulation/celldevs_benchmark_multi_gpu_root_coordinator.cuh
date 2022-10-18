/**
 * Copyright (c) 2022, Guillermo G. Trabes
 * Carleton University, Universidad Nacional de San Luis
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <thread>
#include <omp.h>
#include "../../../affinity/affinity_helpers.hpp"
#include <stdio.h>

const int threadsPerBlock = 256;

__global__ void gpu_output(size_t n_subcomponents, CellDEVSBenchmarkAtomicGPU* subcomponents, size_t first, size_t last, double next_time) {

	size_t i = first + (blockIdx.x*blockDim.x + threadIdx.x);
	if (i < last){
		if (subcomponents[i].next_time == next_time) {
			subcomponents[i].output();
		}
	}


}



__global__ void gpu_route_messages(size_t n_subcomponents, CellDEVSBenchmarkAtomicGPU* subcomponents, size_t first, size_t last, size_t* n_couplings, size_t** couplings) {

	size_t i = first + (blockIdx.x*blockDim.x + threadIdx.x);
	if (i < last){
		for(size_t j=0; j<n_couplings[i]; j++ ){
			subcomponents[i].insert_in_bag(subcomponents[couplings[i][j]].get_out_bag());
		}
	}


}





__global__ void gpu_transition(size_t n_subcomponents, CellDEVSBenchmarkAtomicGPU* subcomponents, size_t first, size_t last, double next_time, double last_time) {
	//printf("Hello World from GPU!\n");
	//size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	//size_t stride = blockDim.x * gridDim.x;

	//for (size_t i = index; i < n_subcomponents; i += stride) {
		//printf("Hello World from GPU!\n");
	//	subcomponents[i].internal_transition();
	//}

	size_t i = first + (blockIdx.x*blockDim.x + threadIdx.x);

	if (i < last){

		if (subcomponents[i].next_time == next_time) {
			if(subcomponents[i].inbag_empty() == true) {
				subcomponents[i].internal_transition();
			} else {
				subcomponents[i].confluent_transition(next_time - last_time);
			}
			//last_time = next_time;
			subcomponents[i].last_time = next_time;
			subcomponents[i].next_time = next_time + subcomponents[i].time_advance();
		} else {
			if(subcomponents[i].inbag_empty() == false){
				subcomponents[i].external_transition(next_time - last_time);
				//last_time = next_time;
				subcomponents[i].last_time = next_time;
				subcomponents[i].next_time = next_time + subcomponents[i].time_advance();
			}
		}
		subcomponents[i].clear_bags();

/*
		subcomponents[i].internal_transition();
		subcomponents[i].last_time = next_time;
		subcomponents[i].next_time = next_time + subcomponents[i].time_advance();
*/
	}

}


__global__ void gpu_next_time(size_t n_subcomponents, CellDEVSBenchmarkAtomicGPU* subcomponents, size_t first, size_t last, double* partial_next_times) {

	__shared__ double blockCache[threadsPerBlock];
	size_t tid = first + (blockIdx.x*blockDim.x + threadIdx.x);
	size_t blockIndex = threadIdx.x;

	//set blockCache values
	if (tid < last){
		blockCache[blockIndex] = subcomponents[tid].next_time;
	}
	//synchronize threads in this block
	__syncthreads();


	//Equivalent to divided by 2
	//size_t jump = blockDim.x>>1;
	size_t jump = blockDim.x>>1;
	//int jump = blockDim.x/2;
/*
	size_t jump;

	if(n_subcomponents < blockDim.x){
		jump = n_subcomponents>>1;
	} else {
		jump = blockDim.x>>1;
	}
*/

	while(jump > 0) {
		if(blockIndex < jump){
			if(blockCache[blockIndex] > blockCache[blockIndex+jump]) {
				blockCache[blockIndex] = blockCache[blockIndex+jump];
			}
		}
		__syncthreads();
		jump = jump>>1;
		//jump/=2;
	}

	if(blockIndex == 0){
		partial_next_times[blockIdx.x] = blockCache[0];
	}

}

void multi_gpu_simulation(size_t n_subcomponents, CellDEVSBenchmarkAtomicGPU* subcomponents, size_t* n_couplings, size_t** couplings , size_t simulation_time) {

	double next_time = 0, last_time = 0;

	int num_gpus = 0;  // number of CUDA GPUs

	/////////////////////////////////////////////////////////////////
	// determine the number of CUDA capable GPUs //
 	cudaGetDeviceCount(&num_gpus);

 	if (num_gpus < 1) {
 		printf("no CUDA capable devices were detected\n");
 	}

 	printf("number of host CPUs:\t%d\n", omp_get_num_procs());
 	printf("number of CUDA devices:\t%d\n", num_gpus);

//	omp_set_num_threads(num_gpus);

	for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");

//	omp_set_num_threads(num_gpus);

	//create parallel region//
	//#pragma omp parallel
	#pragma omp parallel num_threads(num_gpus) shared(next_time, last_time)
	{

		//each thread gets its id//
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();

		#pragma omp critical
		{
			pin_thread_to_core(tid);
		}

//		cudaSetDevice(tid);

// set and check the CUDA device for this CPU thread
        int gpu_id = 0;
//        cudaSetDevice(cpu_thread_id % num_gpus);   // "% >
//		#pragma omp critical
//		{
		cudaSetDevice(tid);
        cudaGetDevice(&gpu_id);
        printf("CPU thread %d (of %d) uses CUDA device %d\n", tid, num_threads, gpu_id);
//		}


		double local_next_time = next_time;

		//calculate number of elements to compute//
		size_t local_n_subcomponents = n_subcomponents/num_threads;

		// calculate start position/
		size_t first_subcomponents = tid*local_n_subcomponents;


		// calculate end position/
		size_t last_subcomponents;

		if(tid != (num_threads-1)){
			last_subcomponents = (tid+1)*local_n_subcomponents;
		} else {
			last_subcomponents = n_subcomponents;
		}



//		int gpu_id = -1;;
// 		cudaGetDevice(&gpu_id);

//    	printf("CPU thread %d (of %d) uses CUDA device %d\n", tid, num_gpus, gpu_id);

		size_t numBlocks = (local_n_subcomponents + threadsPerBlock - 1) / threadsPerBlock;
		double *partial_next_times;

		// Allocate Unified Memory -- accessible from CPU or GPU
		cudaMallocManaged(&partial_next_times, numBlocks*sizeof(double));


		while(next_time < simulation_time) {

			// Launch Step 1 on the GPU
			gpu_output<<<numBlocks, threadsPerBlock>>>(n_subcomponents, subcomponents, first_subcomponents, last_subcomponents, next_time);
			// Wait for GPU to finish
			//cudaDeviceSynchronize();
			// End Step 1

			#pragma omp barrier

			// Launch Step 2 on the GPU
			gpu_route_messages<<<numBlocks, threadsPerBlock>>>(n_subcomponents, subcomponents, first_subcomponents, last_subcomponents, n_couplings, couplings);
			// Wait for GPU to finish
			//cudaDeviceSynchronize();
			// End Step 2

			#pragma omp barrier

			// Launch Step 3 on the GPU
			gpu_transition<<<numBlocks, threadsPerBlock>>>(n_subcomponents, subcomponents, first_subcomponents, last_subcomponents, next_time, last_time);
			// Wait for GPU to finish
			//cudaDeviceSynchronize();
			// End Step 3

			#pragma omp barrier

			// Launch Step 4 on the GPU
			gpu_next_time<<<numBlocks, threadsPerBlock>>>(n_subcomponents, subcomponents, first_subcomponents, last_subcomponents, partial_next_times);
			// Wait for GPU to finish
			cudaDeviceSynchronize();

			// sequential minimum with partial results from GPU
			local_next_time = partial_next_times[0];

			for(size_t i = 1; i < numBlocks; i++){
				if(partial_next_times[i] < next_time) {
					local_next_time = partial_next_times[i];
				}
			}
			#pragma omp barrier

			//only 1 threads initializes shared minimum wuth local minimum
			#pragma omp single
			{
				next_time = local_next_time;
			}

			#pragma omp barrier

			// calculate final result from partial_results sequentially /
			#pragma omp critical
			{
				if(local_next_time < next_time){
					next_time = local_next_time;
				}
			}
			#pragma omp barrier

		}//end loop

	}//end parallel section

}//end function
