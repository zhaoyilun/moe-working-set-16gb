cmake_minimum_required(VERSION 3.24)
project(pc_slotstream_tools LANGUAGES C CXX)

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to the patched llama.cpp checkout")
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

add_executable(slotstream-prefix-cache prefix_cache_runner.cpp)
target_compile_features(slotstream-prefix-cache PRIVATE cxx_std_17)
target_link_libraries(slotstream-prefix-cache PRIVATE llama ggml)

add_executable(slotstream-hybrid-decode hybrid_decode_runner.cpp)
target_compile_features(slotstream-hybrid-decode PRIVATE cxx_std_17)
target_link_libraries(slotstream-hybrid-decode PRIVATE llama ggml)
