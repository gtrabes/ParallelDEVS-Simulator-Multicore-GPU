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
#include "atomics/celldevs_benchmark_atomic.hpp"
#include "simulation/celldevs_benchmark_naive_parallel_root_coordinator.hpp"
#include "../../affinity/affinity_helpers.hpp"
#include <fstream>
#include "../../rapl-tools/Rapl.h"

using namespace std;
using hclock=std::chrono::high_resolution_clock;

int main(int argc, char **argv) {

	std::ofstream file;    //!< output file stream.
	Rapl *rapl = new Rapl(0);

	//pin core to thread 0
	pin_thread_to_core(0);

	auto parallel_begin = hclock::now(), parallel_end = hclock::now();

	// First, we parse the arguments
	if (argc < 3) {
		std::cerr << "ERROR: not enough arguments" << std::endl;
		std::cerr << "    Usage:" << std::endl;
		std::cerr << "    > devs-cuda ATOMICS_NUMBER OUTPUT_FLOPS_NUMBER TRANSITION_FLOPS SIMULATION_TIME" << std::endl;
		std::cerr << "        (ATOMICS_NUMBER OUTPUT_FLOPS TRANSITION_FLOPS SIMULATION_TIME must be greater or equel to 1)" << std::endl;
		return -1;
	}

	size_t grid_dimension = std::stoll(argv[1]);
	if (grid_dimension < 1) {
		std::cerr << "ERROR: GRID_DIMENSION is less than 1 (" << grid_dimension << ")" << std::endl;
		return -1;
	}
	size_t simulation_time = std::stoll(argv[2]);
	if (simulation_time < 0) {
		std::cerr << "ERROR: SIMULATION_TIME is less than 0 (" << simulation_time << ")" << std::endl;
		return -1;
	}
	size_t num_threads = std::stoll(argv[3]);
	if (num_threads < 1) {
		std::cerr << "ERROR: NUMBER_OF_THREADS is less than 1 (" << num_threads << ")" << std::endl;
		return -1;
	}

	size_t n_atomics;

	n_atomics = grid_dimension* grid_dimension;

	//create data structure for indexes
	size_t **grid_indexs;

	//allocate matrix indexes
	grid_indexs = (size_t**)malloc(n_atomics * sizeof(size_t*));

	for(int i = 0; i < n_atomics; i++){
		grid_indexs[i] = (size_t *)malloc(2 * sizeof(size_t));
	}

	for(size_t i = 0; i < grid_dimension; i++) {
		for(size_t j = 0; j < grid_dimension; j++) {
			grid_indexs[(i*grid_dimension)+(j)][0] = i;
			grid_indexs[(i*grid_dimension)+(j)][1] = j;
		}
	}


//	Atomic **atomic_pointers_array;
	CellDEVSBenchmarkAtomic *atomic_array;

	atomic_array = (CellDEVSBenchmarkAtomic*) malloc(n_atomics*sizeof(CellDEVSBenchmarkAtomic));

	for(size_t i = 0; i < n_atomics; i++) {
		atomic_array[i] = CellDEVSBenchmarkAtomic();
	}

	//create data structure for couplings
	size_t **couplings;

	//allocate couplings matrix
	couplings = (size_t**)malloc(n_atomics * sizeof(size_t*));

	for(int i = 0; i < n_atomics; i++){
		couplings[i] = (size_t *)malloc(9 * sizeof(size_t));
	}

	//create data structure for couplings
	size_t *n_couplings;

	//allocate couplings matrix
	n_couplings = (size_t *)malloc(n_atomics * sizeof(size_t));

	int indexX = 0, indexY = 0, array_index = 0;

	//fill data structure for couplings
	for(int i = 0; i < n_atomics; i++){
		n_couplings[i] = 0;

		for(int j=-1; j<= 1; j++) {
			for(int k=-1; k<= 1; k++) {
				indexX = grid_indexs[i][0]+j;
				indexY = grid_indexs[i][1]+k;

				if (indexX >= 0 && indexX < grid_dimension && indexY >= 0 && indexY < grid_dimension) {
					array_index = (indexX * grid_dimension)+(indexY);
					couplings[i][n_couplings[i]] = array_index;
					n_couplings[i]++;
				}
			}
		}

	}

	//parallel_begin = hclock::now();
	rapl->measure_begin();

	parallel_simulation(n_atomics, atomic_array, n_couplings, couplings, simulation_time, num_threads);

	// log results
	file.open("celldevs_benchmark_naive_parallel_log.csv");
	file << "time" << ";" << "model_id" << ";" << "model_name" << ";" << "state" << std::endl;

	for(size_t i=0; i<n_atomics; i++){
		file << simulation_time << ";" << i << ";" << "<" << grid_indexs[i][0] << "," << grid_indexs[i][1] << ">" << ";" << "<" << atomic_array[i].state << ">" << std::endl;
	}

	//parallel_end = hclock::now();
	rapl->measure_end();

	// calculate and print time
	//std::cout << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(parallel_end - parallel_begin).count() << std::endl;
	std::cout << rapl->total_time() << " " << rapl->total_energy() << " " << rapl->total_power() << " " << rapl->total_time()*rapl->total_energy() << std::endl;

	return 0;
}
