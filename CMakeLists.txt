cmake_minimum_required (VERSION 3.1)

project(cuda-example LANGUAGES CXX CUDA)

add_executable(devs-cuda devs-cuda.cu)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(devs-cuda PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(devs-cuda PROPERTIES CUDA_ARCHITECTURES "61;75")