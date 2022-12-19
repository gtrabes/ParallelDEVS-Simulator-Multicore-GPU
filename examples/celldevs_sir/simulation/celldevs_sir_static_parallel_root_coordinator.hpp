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
#include "omp.h"
#include "../../../affinity/affinity_helpers.hpp"

void parallel_simulation(size_t n_subcomponents, CellDEVSSirAtomic* subcomponents, size_t* n_couplings, size_t** couplings, size_t sim_time,
	size_t num_threads = std::thread::hardware_concurrency()) {

/*
void parallel_simulation(size_t n_subcomponents, Atomic* subcomponents, size_t* n_couplings, size_t couplings[][], size_t sim_time,
	size_t num_threads = std::thread::hardware_concurrency()) {
*/
	double next_time = 0, last_time = 0;

	//create threads
	#pragma omp parallel num_threads(num_threads)
	//#pragma omp parallel num_threads(num_threads) shared(next_time, last_time)
	{
		size_t tid = omp_get_thread_num();

		#pragma omp critical
		{
			pin_thread_to_core(tid);
		}

		size_t thread_number = omp_get_num_threads();

        //calculate number of elements to compute//
        size_t local_n_subcomponents = n_subcomponents/thread_number;

        // calculate start position/
        size_t first_subcomponents = tid*local_n_subcomponents;

        // calculate end position/
        size_t last_subcomponents;

        if(tid != (thread_number-1)){
            last_subcomponents = (tid+1)*local_n_subcomponents;
        } else {
            last_subcomponents = n_subcomponents;
        }

		double local_next_time;

		while(next_time < sim_time) {

			// Step 1: execute output functions
			for(size_t i=first_subcomponents; i<last_subcomponents;i++){
				if (subcomponents[i].get_next_time() == next_time) {
					subcomponents[i].output();
				}
			}
			#pragma omp barrier
			//end Step 1


			// Step 2: route messages
			//#pragma omp for schedule(static)
			for(size_t i = first_subcomponents; i < last_subcomponents; i++){
				for(size_t j = 0; j < n_couplings[i]; j++ ){
					//for (size_t k=0; k < 10; k++){
						auto index = couplings[i][j];
						//if(index < n_subcomponents) {
						auto out_bag = subcomponents[index].get_out_bag();
						subcomponents[i].insert_in_bag(out_bag);
					//}
					//}
						//subcomponents[i].insert_in_bag(subcomponents[couplings[i][j]].get_out_bag());

					//subcomponents[i].insert_in_bag(out_bag);
					//subcomponents[i].insert_in_bag(subcomponents[0].get_out_bag());
				}
			}
			#pragma omp barrier
			//end Step 2

			//Step 3: execute state transition
			//#pragma omp for schedule(static)
			for(size_t i=first_subcomponents; i< last_subcomponents;i++){
				if (subcomponents[i].get_next_time() == next_time) {
					if(subcomponents[i].inbag_empty() == true) {
						subcomponents[i].internal_transition();
					} else {
						subcomponents[i].confluent_transition(next_time - last_time);
					}
					last_time = next_time;
					subcomponents[i].last_time = last_time;
					subcomponents[i].next_time = last_time + subcomponents[i].time_advance();
				} else {
					if(subcomponents[i].inbag_empty() == false){
						subcomponents[i].external_transition(next_time - last_time);
						last_time = next_time;
						subcomponents[i].last_time = last_time;
						subcomponents[i].next_time = last_time + subcomponents[i].time_advance();
					}
				}
				subcomponents[i].clear_bags();
			}
			#pragma omp barrier
			//end Step 3

			//Step 4
			local_next_time = subcomponents[first_subcomponents].get_next_time();
			//#pragma omp for schedule(static)
			for(size_t i=first_subcomponents+1; i<last_subcomponents;i++){
				if(subcomponents[i].get_next_time() < local_next_time){
					local_next_time = subcomponents[i].get_next_time();
				}
			}
			#pragma omp single
			{
				next_time = local_next_time;
			}
			#pragma omp barrier
			#pragma omp critical
			{
				if(local_next_time < next_time){
					next_time = local_next_time;
				}
			}
			#pragma omp barrier
			// end Step 4

		}//end simulation loop
	}//end parallel region
}
