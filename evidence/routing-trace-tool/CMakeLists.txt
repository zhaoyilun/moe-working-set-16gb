cmake_minimum_required(VERSION 3.24)
project(slotstream_routing_trace LANGUAGES C CXX)

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to the llama.cpp checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

add_executable(slotstream-routing-trace llama_routing_trace.cpp)
target_compile_features(slotstream-routing-trace PRIVATE cxx_std_17)
target_link_libraries(slotstream-routing-trace PRIVATE llama ggml)

find_package(CUDAToolkit REQUIRED)
add_executable(slotstream-e2e-profile ../profiling/end_to_end_runner.cpp)
target_compile_features(slotstream-e2e-profile PRIVATE cxx_std_17)
target_link_libraries(slotstream-e2e-profile PRIVATE llama ggml CUDA::cudart)

add_executable(slotstream-prefix-cache ../profiling/prefix_cache_runner.cpp)
target_compile_features(slotstream-prefix-cache PRIVATE cxx_std_17)
target_link_libraries(slotstream-prefix-cache PRIVATE llama ggml)

add_executable(slotstream-hybrid-decode ../profiling/hybrid_decode_runner.cpp)
target_compile_features(slotstream-hybrid-decode PRIVATE cxx_std_17)
target_link_libraries(slotstream-hybrid-decode PRIVATE llama ggml)

add_executable(slotstream-context-prompt ../profiling/context_prompt_tool.cpp)
target_compile_features(slotstream-context-prompt PRIVATE cxx_std_17)
target_include_directories(slotstream-context-prompt PRIVATE "${LLAMA_CPP_DIR}/vendor")
target_link_libraries(slotstream-context-prompt PRIVATE llama ggml)
