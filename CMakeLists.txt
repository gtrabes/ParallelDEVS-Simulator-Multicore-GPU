cmake_minimum_required (VERSION 3.1)

project(heterogeneous_devs LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(flops_benchmark_sequential ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_sequential.cpp)
add_executable(flops_benchmark_static_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_static_parallel.cpp)
add_executable(flops_benchmark_static_parallel_numa ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_static_parallel_numa.cpp)
add_executable(flops_benchmark_dynamic_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_dynamic_parallel.cpp)
add_executable(flops_benchmark_naive_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_naive_parallel.cpp)
#add_executable(flops_benchmark_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_gpu.cu)
#add_executable(flops_benchmark_naive_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/flops_benchmark/flops_benchmark_naive_gpu.cu)

add_executable(celldevs_benchmark_sequential ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_sequential.cpp)
add_executable(celldevs_benchmark_naive_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_naive_parallel.cpp)
add_executable(celldevs_benchmark_static_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_static_parallel.cpp)
add_executable(celldevs_benchmark_dynamic_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_dynamic_parallel.cpp)
#add_executable(celldevs_benchmark_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_gpu.cu)
add_executable(celldevs_benchmark_static_parallel_numa ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_static_parallel_numa.cpp)
#add_executable(celldevs_benchmark_naive_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_naive_gpu.cu)
#add_executable(celldevs_benchmark_multi_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_benchmark/celldevs_benchmark_multi_gpu.cu)

add_executable(celldevs_sir_sequential ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_sequential.cpp)
add_executable(celldevs_sir_naive_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_naive_parallel.cpp)
add_executable(celldevs_sir_static_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_static_parallel.cpp)
add_executable(celldevs_sir_dynamic_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_dynamic_parallel.cpp)
#add_executable(celldevs_sir_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_gpu.cu)
add_executable(celldevs_sir_static_parallel_numa ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_static_parallel_numa.cpp)
#add_executable(celldevs_sir_naive_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_naive_gpu.cu)
#add_executable(celldevs_sir_multi_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/celldevs_sir/celldevs_sir_multi_gpu.cu)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flops_benchmark_static_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(flops_benchmark_dynamic_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(flops_benchmark_naive_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(flops_benchmark_static_parallel_numa PUBLIC OpenMP::OpenMP_CXX)
    
    target_link_libraries(celldevs_benchmark_naive_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(celldevs_benchmark_static_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(celldevs_benchmark_dynamic_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(celldevs_benchmark_static_parallel_numa PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(celldevs_benchmark_multi_gpu PUBLIC OpenMP::OpenMP_CXX)
    
    target_link_libraries(celldevs_sir_naive_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(celldevs_sir_static_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(celldevs_sir_dynamic_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(celldevs_sir_static_parallel_numa PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(celldevs_sir_multi_gpu PUBLIC OpenMP::OpenMP_CXX)
    
endif()

#set_target_properties(flops_benchmark_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
#set_target_properties(flops_benchmark_naive_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")

#set_target_properties(celldevs_benchmark_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
#set_target_properties(celldevs_benchmark_naive_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
#set_target_properties(celldevs_benchmark_multi_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")

#set_target_properties(celldevs_sir_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
#set_target_properties(celldevs_sir_naive_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
#set_target_properties(celldevs_sir_multi_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
