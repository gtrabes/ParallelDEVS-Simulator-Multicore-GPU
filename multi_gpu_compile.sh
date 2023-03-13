nvcc examples/celldevs_benchmark/celldevs_benchmark_multi_gpu.cu rapl-tools/Rapl.cpp -o bin/celldevs_benchmark_multi_gpu -Xcompiler -fopenmp
nvcc examples/celldevs_sir/celldevs_sir_multi_gpu.cu -o bin/celldevs_sir_multi_gpu -Xcompiler -fopenmp 
