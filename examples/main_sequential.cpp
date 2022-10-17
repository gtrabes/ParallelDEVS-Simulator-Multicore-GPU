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

#include <iostream>
#include <chrono>
#include "../modeling/atomic.hpp"
#include "../simulation/sequential/sequential_root_coordinator.hpp"

using namespace std;
using hclock=std::chrono::high_resolution_clock;

int main(int argc, char **argv) {

	auto sequential_begin = hclock::now(), parallel_begin = hclock::now(), gpu_begin = hclock::now();
	auto sequential_end = hclock::now(), parallel_end = hclock::now(), gpu_end = hclock::now();

	// First, we parse the arguments
	if (argc < 5) {
		std::cerr << "ERROR: not enough arguments" << std::endl;
		std::cerr << "    Usage:" << std::endl;
		std::cerr << "    > devs-cuda ATOMICS_NUMBER OUTPUT_FLOPS_NUMBER TRANSITION_FLOPS SIMULATION_TIME" << std::endl;
		std::cerr << "        (ATOMICS_NUMBER OUTPUT_FLOPS TRANSITION_FLOPS SIMULATION_TIME must be greater or equel to 1)" << std::endl;
		return -1;
	}

	size_t n_atomics = std::stoll(argv[1]);
	if (n_atomics < 1) {
		std::cerr << "ERROR: ATOMICS_NUMBER is less than 1 (" << n_atomics << ")" << std::endl;
		return -1;
	}
	size_t output_flops = std::stoll(argv[2]);
	if (output_flops < 1) {
		std::cerr << "ERROR: OUTPUT_FLOPS is less than 1 (" << output_flops << ")" << std::endl;
		return -1;
	}
	size_t transition_flops = std::stoll(argv[3]);
	if (transition_flops < 0) {
		std::cerr << "ERROR: TRANSITION_FLOPS is less than 0 (" << transition_flops << ")" << std::endl;
		return -1;
	}
	size_t simulation_time = std::stoll(argv[4]);
	if (transition_flops < 0) {
		std::cerr << "ERROR: SIMULATION_TIME is less than 0 (" << simulation_time << ")" << std::endl;
		return -1;
	}

//	Atomic **atomic_pointers_array;
	Atomic *atomic_array;

/*
	 // Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&atomic_pointers_array, size*sizeof(Atomic*));

	for(size_t i = 0; i < size; i++) {
		cudaMallocManaged(&atomic_pointers_array[i], sizeof(Atomic));
	}
*/
	// Allocate Unified Memory -- accessible from CPU or GPU
	//cudaMallocManaged(&atomic_array, n_atomics*sizeof(Atomic));
	atomic_array = (Atomic*) malloc(n_atomics*sizeof(Atomic));

	for(size_t i = 0; i < n_atomics; i++) {
		atomic_array[i] = Atomic(output_flops, transition_flops);
	}

	//create data structure for couplings
	size_t **couplings;

	//allocate couplings matrix
	couplings = (size_t**)malloc(n_atomics * sizeof(size_t*));

	for(size_t i = 0; i < n_atomics; i++){
		couplings[i] = (size_t *)malloc(9 * sizeof(size_t));
	}

	//create data structure for couplings
	int *n_couplings;

	//allocate couplings matrix
	n_couplings = (size_t *)malloc(n_atomics * sizeof(size_t));

	//fill data structure for couplings
	for(size_t i=0; i < n_atomics; i++){
		size_t aux = i-5;
		for(size_t j = 0; j < 9; j++){
			if ((aux+j > 0) && (aux+j < n_atomics)){
				couplings[i][j] = aux+j;
				n_couplings[i]++;
			}
		}
	}


	sequential_begin = hclock::now();

	sequential_simulation(n_atomics, atomic_array, n_couplings, couplings, simulation_time);

	sequential_end = hclock::now();

	// calculate and print time
	std::cout << "Sequential time:   "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(sequential_end - sequential_begin).count() << std::endl;

	return 0;
}
