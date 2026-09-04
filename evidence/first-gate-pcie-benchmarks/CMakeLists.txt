cmake_minimum_required(VERSION 3.24)

project(pc_slotstream LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(slotstream_bench
    benchmarks/pcie_expert_bench.cu
)

target_compile_features(slotstream_bench PRIVATE cxx_std_17)
target_link_libraries(slotstream_bench PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(slotstream_bench PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

if(MSVC)
    target_compile_options(slotstream_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
endif()
