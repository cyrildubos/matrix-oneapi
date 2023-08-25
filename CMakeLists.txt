cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(matrix-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(buffer src/buffer.cpp)
add_executable(data   src/data.cpp)
add_executable(vector src/vector.cpp)
