cmake_minimum_required (VERSION 3.1)

project(heterogeneous_devs LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(sequential ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_sequential.cpp)
add_executable(static_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_static_parallel.cpp)
add_executable(static_parallel_numa ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_static_parallel_numa.cpp)
add_executable(dynamic_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_dynamic_parallel.cpp)
add_executable(naive_parallel ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_naive_parallel.cpp)
add_executable(gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_gpu.cu)
add_executable(naive_gpu ${CMAKE_CURRENT_SOURCE_DIR}/examples/main_naive_gpu.cu)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(static_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dynamic_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(naive_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(static_parallel_numa PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
set_target_properties(naive_gpu PROPERTIES CUDA_ARCHITECTURES "61;75")
