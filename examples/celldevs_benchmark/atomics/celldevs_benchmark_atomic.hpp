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

#ifndef CELL_DEVS_BENCHMARK_ATOMIC
#define CELL_DEVS_BENCHMARK_ATOMIC

	/**
	 * @brief DEVS atomic model.
	 *
	 * The Atomic class is closer to the DEVS formalism than the AtomicInterface class.
	 * @tparam S the data type used for representing a cell state.
	 */
    class CellDEVSBenchmarkAtomic {
     public:
    	bool state;
    	double last_time, next_time;
    	bool is_inbag_empty;
    	bool in_bag[9];
    	bool out_bag;
    	size_t num_messages_received;

    	CellDEVSBenchmarkAtomic(){
    		state = 0;
    		next_time = 0;
    		last_time = 0;
    		is_inbag_empty = true;
    		for(size_t i = 0; i < 9; i++){
    			in_bag[i] = 0;
    		}
    		out_bag = 0;
    		num_messages_received = 0;
    	}

		/**
		 * Sends a new Job that needs to be processed via the Generator::outGenerated port.
		 * @param s reference to the current generator model state.
		 * @param y reference to the atomic model output port set.
		 */
		void output() {
			out_bag = state;
		}

		/**
		 * Updates GeneratorState::clock and GeneratorState::sigma and increments GeneratorState::jobCount by one.
		 * @param s reference to the current generator model state.
		 */
		void internal_transition() {
			if(state == 0){
				state = 1;
			} else {
				state = 0;
			}
		}

		/**
		 * Updates GeneratorState::clock and GeneratorState::sigma.
		 * If it receives a true message via the Generator::inStop port, it passivates and stops generating Job objects.
		 * @param s reference to the current generator model state.
		 * @param e time elapsed since the last state transition function was triggered.
		 * @param x reference to the atomic model input port set.
		 */
		void external_transition(double e) {
			if(state == 0){
				state = 1;
			} else {
				state = 0;
			}
		}

		/**
		 * Updates GeneratorState::clock and GeneratorState::sigma.
		 * If it receives a true message via the Generator::inStop port, it passivates and stops generating Job objects.
		 * @param s reference to the current generator model state.
		 * @param e time elapsed since the last state transition function was triggered.
		 * @param x reference to the atomic model input port set.
		 */
		void confluent_transition(double e) {
			if(state == 0){
				state = 1;
			} else {
				state = 0;
			}
		}

		/**
		 * It returns the value of GeneratorState::sigma.
		 * @param s reference to the current generator model state.
		 * @return the sigma value.
		 */
		double time_advance() {
			return 1;
		}

		/**
		* It returns the value of GeneratorState::sigma.
		* @param s reference to the current generator model state.
		* @return the sigma value.
		*/
		double get_next_time() {
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
		bool inbag_empty() {
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
		void insert_in_bag(double in_message) {
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
		bool get_out_bag() {
			return out_bag;
		}

		void clear_bags() {
			for(size_t i = 0; i < 9; i++) {
				in_bag[i] = 0;
			}
			out_bag = 0;
			num_messages_received = 0;
		}



    };

#endif //CADMIUM_CORE_SIMULATION_ABS_SIMULATOR_HPP_
