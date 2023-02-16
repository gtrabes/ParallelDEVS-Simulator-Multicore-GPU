/**
 * Abstract implementation of a DEVS atomic model.
 * Copyright (C) 2022 Guillermo Trabes
 * ARSLab - Carleton University
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CELL_DEVS_SIR_ATOMIC_GPU
#define CELL_DEVS_SIR_ATOMIC_GPU

#include <cmath>

	//! Susceptible-Infected-Recovered state.
	struct SIRState {
		int p;     //!< Cell population.
		double s;  //!< Ratio of susceptible people (from 0 to 1).
		double i;  //!< Ratio of infected people (from 0 to 1).
		double r;  //!< Ratio of recovered people (from 0 to 1).

		//! Default constructor function. By default, cells are unoccupied and all the population is considered susceptible.
		SIRState() : p(1000), s(1), i(0), r(0) {
		}
	};

	/**
	 * @brief DEVS atomic model.
	 *
	 * The Atomic class is closer to the DEVS formalism than the AtomicInterface class.
	 * @tparam S the data type used for representing a cell state.
	 */
    class CellDEVSSirAtomicGPU {
     public:
    	SIRState state;
    	SIRState lastState;
    	double rec;   //!< recovery factor.
    	double susc;  //!< susceptibility factor.
    	double vir;   //!< virulence factor.

    	double last_time, next_time;
    	bool is_inbag_empty;
    	SIRState in_bag[9];
    	SIRState out_bag;
    	size_t num_messages_received;
    	size_t num_neighbors;

    	__host__ CellDEVSSirAtomicGPU(double p, double s, double i, double r){

    		state.p = p;
    		state.s = s;
    		state.i = i;
    		state.r = r;

    		rec = 0.2;
    		susc = 0.8;
    		vir = 0.4;

    		next_time = 0;

    		if(s != 1.0 || i != 0.0 || r != 0.0){
    			next_time++;
    		}

    		last_time = 0;
    		is_inbag_empty = true;
    		num_messages_received = 0;
    	}

		/**
		 * Sends a new Job that needs to be processed via the Generator::outGenerated port.
		 * @param s reference to the current generator model state.
		 * @param y reference to the atomic model output port set.
		 */
		__host__ __device__ void output() {
			out_bag = state;
		}

		/**
		 * Updates GeneratorState::clock and GeneratorState::sigma and increments GeneratorState::jobCount by one.
		 * @param s reference to the current generator model state.
		 */
		__host__ __device__ void internal_transition() {
			if (lastState.s != state.s || lastState.i != state.i || lastState.r != state.r) {
				lastState = state;
				next_time = next_time+0.1;
			}
		}


		/**
		 * Updates GeneratorState::clock and GeneratorState::sigma.
		 * If it receives a true message via the Generator::inStop port, it passivates and stops generating Job objects.
		 * @param s reference to the current generator model state.
		 * @param e time elapsed since the last state transition function was triggered.
		 * @param x reference to the atomic model input port set.
		 */
		__host__ __device__ void external_transition(double e) {
			auto nextState = localComputation(state, in_bag);
			if (nextState.s != state.s || nextState.i != state.i || nextState.r != state.r) {
				lastState = state;
				state = nextState;
			}
		}


		__host__ __device__ SIRState localComputation(SIRState state, SIRState neighborhood[9]) {
			auto newI = newInfections(state, neighborhood);
			auto newR = newRecoveries(state);

			// We round the outcome to three decimals:
			state.r = std::round((state.r + newR) * 1000) / 1000;
			state.i = std::round((state.i + newI - newR) * 1000) / 1000;
			state.s = 1 - state.i - state.r;
			return state;
		}


		__host__ __device__ double newInfections(SIRState& state, SIRState neighborhood[9]) {
			double aux = 0;
			for (int i=0; i<num_messages_received; i++) {
				auto s = neighborhood[i];
				//aux += s.i * (double)s.p * 0.111;
				//aux += s.i * (double)s.p * (double)(1.0/num_neighbors);
				aux += s.i * (double)s.p * (double)(0.111);
			}
			double result;
			if((vir * aux / state.p) < 1 ){
				result = state.s * susc * (vir * aux / state.p);
			} else {
				result = state.s * susc;
			}

			return result;
		}

		__host__ __device__ double newRecoveries(SIRState& state){
			return state.i * rec;
		}


		/**
		 * Updates GeneratorState::clock and GeneratorState::sigma.
		 * If it receives a true message via the Generator::inStop port, it passivates and stops generating Job objects.
		 * @param s reference to the current generator model state.
		 * @param e time elapsed since the last state transition function was triggered.
		 * @param x reference to the atomic model input port set.
		 */
		__host__ __device__ void confluent_transition(double e) {
			internal_transition();
			external_transition(e);
		}

		/**
		 * It returns the value of GeneratorState::sigma.
		 * @param s reference to the current generator model state.
		 * @return the sigma value.
		 */
		__host__ __device__ double time_advance() {
			return 1;
		}

		/**
		* It returns the value of GeneratorState::sigma.
		* @param s reference to the current generator model state.
		* @return the sigma value.
		*/
		__host__ __device__ double get_next_time() {
			return next_time;
		}

		/**
		* It returns the value of GeneratorState::sigma.
		* @param s reference to the current generator model state.
		* @return the sigma value.
		*/
/*
		bool inbag_empty() {
			for(size_t i = 0; i < 9; i++){
				if(in_bag[i] != -1.0){
					is_inbag_empty == false;
				}
			}
			return is_inbag_empty;
		}
*/
		__host__ __device__ bool inbag_empty() {
			if(num_messages_received == 0){
				is_inbag_empty = true;
			} else {
				is_inbag_empty = false;
			}
			return is_inbag_empty;
		}

		/**
		* It returns the value of GeneratorState::sigma.
		* @param s reference to the current generator model state.
		* @return the sigma value.
		*/
		__host__ __device__ void insert_in_bag(SIRState in_message) {
			if(num_messages_received < 9){
				in_bag[num_messages_received] = in_message;
				num_messages_received++;
			}
		}

		/**
		* It returns the value of GeneratorState::sigma.
		* @param s reference to the current generator model state.
		* @return the sigma value.
		*/
		__host__ __device__ SIRState get_out_bag() {
			return out_bag;
		}


		__host__ __device__ void clear_bags() {
/*			for(size_t i = 0; i < 9; i++) {
				in_bag[i] = new;
			}
			out_bag = 0;
*/
			num_messages_received = 0;
		}



    };

#endif //CADMIUM_CORE_SIMULATION_ABS_SIMULATOR_HPP_
