cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
# CUDA SETUP

project(OpenQMC LANGUAGES CXX CUDA)

# STD
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)