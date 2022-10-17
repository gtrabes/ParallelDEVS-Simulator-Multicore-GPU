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

const int threadsPerBlock = 256;

__global__ void gpu_output(size_t n_subcomponents, AtomicGPU* subcomponents, double next_time) {
	//printf("Hello World from GPU!\n");
/*
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = index; i < n_subcomponents; i += stride) {
		//printf("Hello World from GPU!\n");
		//	subcomponents[i].internal_transition();
		if (subcomponents[i].next_time == next_time) {
			subcomponents[i].output();
		}
	}
*/

	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n_subcomponents){
		if (subcomponents[i].next_time == next_time) {
			subcomponents[i].output();
		}
	}


}


__global__ void gpu_transition(size_t n_subcomponents, AtomicGPU* subcomponents, double next_time, double last_time) {
	//printf("Hello World from GPU!\n");
	//size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	//size_t stride = blockDim.x * gridDim.x;

	//for (size_t i = index; i < n_subcomponents; i += stride) {
		//printf("Hello World from GPU!\n");
	//	subcomponents[i].internal_transition();
	//}

	size_t i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n_subcomponents){

		//if (subcomponents[i].next_time == next_time) {
			//if(subcomponents[i].inports_empty() == true) {
				subcomponents[i].internal_transition();
			//} else {
			//	subcomponents[i].confluent_transition(next_time - last_time);
			//}
			//last_time = next_time;
			subcomponents[i].last_time = next_time;
			subcomponents[i].next_time = next_time + subcomponents[i].time_advance();
		//}// else {
		/*	if(subcomponents[i].inports_empty() == false){
				subcomponents[i].external_transition(next_time - last_time);
				//last_time = next_time;
				subcomponents[i].last_time = next_time;
				subcomponents[i].next_time = next_time + subcomponents[i].time_advance();
			}*/
		//}
		subcomponents[i].clear_bags();

/*
		subcomponents[i].internal_transition();
		subcomponents[i].last_time = next_time;
		subcomponents[i].next_time = next_time + subcomponents[i].time_advance();
*/
	}

}


void gpu_simulation(size_t n_subcomponents, AtomicGPU* subcomponents, size_t* n_couplings, size_t** couplings , size_t simulation_time) {
	//printf("Hello World from GPU!\n");
	//size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	//size_t stride = blockDim.x * gridDim.x;

	//for (size_t i = index; i < n_subcomponents; i += stride) {
		//printf("Hello World from GPU!\n");
	//	subcomponents[i].internal_transition();
	//}
/*
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	double

	while()
		if (i<n_subcomponents){
			subcomponents[i].output();
		}
*/

	double next_time = 0, last_time = 0;

	int blockSize = 256;
	int numBlocks = (n_subcomponents + blockSize - 1) / blockSize;

	double *partial_next_times;

	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&partial_next_times, numBlocks*sizeof(double));

	while(next_time < simulation_time) {

		// Launch Step 1 on the GPU
		gpu_output<<<numBlocks, blockSize>>>(n_subcomponents, subcomponents, next_time);
		// Wait for GPU to finish
		cudaDeviceSynchronize();
		// End Step 1

		// Launch Step 2 on the GPU
		//gpu_propaget<<<numBlocks, blockSize>>>(n_subcomponents, subcomponents, next_time, last_time);
		// Wait for GPU to finish
		//cudaDeviceSynchronize();
		// End Step 2

		// Step 2: route messages
		for(size_t i=0; i<n_subcomponents;i++){
			for(size_t j=0; j<n_couplings[i]; j++ ){
				subcomponents[i].insert_in_bag(subcomponents[couplings[i][j]].get_out_bag());
			}
		}
		//end Step 2


		// Launch Step 3 on the GPU
		gpu_transition<<<numBlocks, blockSize>>>(n_subcomponents, subcomponents, next_time, last_time);
		// Wait for GPU to finish
		cudaDeviceSynchronize();
		// End Step 3

/*
		next_time = subcomponents[0].get_next_time();
		for(size_t i=1; i<n_subcomponents;i++){
			if(subcomponents[i].get_next_time() < next_time){
				next_time = subcomponents[i].get_next_time();
			}
		}
*/

/*
		// Launch Step 4 on the GPU
		gpu_next_time<<<numBlocks, blockSize>>>(n_subcomponents, subcomponents, partial_next_times);
		// Wait for GPU to finish
		cudaDeviceSynchronize();

		// sequential minimum with partial results from GPU
		next_time = partial_next_times[0];

		for(size_t i = 0; i < blockSize; i++){
			if(partial_next_times[i] < next_time) {
				next_time = partial_next_times[i];
			}
		}
		//end Step 4
*/
		//Step 4
		next_time = subcomponents[0].get_next_time();
		for(size_t i=1; i<n_subcomponents;i++){
			if(subcomponents[i].get_next_time() < next_time){
				next_time = subcomponents[i].get_next_time();
			}
		}
		// end Step 4


		//printf("TIME: %lf", next_time);

	}
}
