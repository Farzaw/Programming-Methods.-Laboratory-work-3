cmake_minimum_required(VERSION 3.14)
project(psp_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(psp_lab main.cpp)

if(MSVC)
    target_compile_options(psp_lab PRIVATE /W4)
else()
    target_compile_options(psp_lab PRIVATE -Wall -Wextra -O2)
endif()
