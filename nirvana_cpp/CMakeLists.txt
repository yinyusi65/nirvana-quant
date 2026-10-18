cmake_minimum_required(VERSION 3.20)
project(nirvana_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings and optimizations
if (MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Header-only library interface
add_library(nirvana INTERFACE)
target_include_directories(nirvana INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Test Executable
add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE nirvana)

# Examples Executable
add_executable(demo_banking examples/demo_banking.cpp)
target_link_libraries(demo_banking PRIVATE nirvana)

# Enable CTest
enable_testing()
add_test(NAME NirvanaCoreTest COMMAND test_core)
