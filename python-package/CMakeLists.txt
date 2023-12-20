cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 14)
project(pystochtree LANGUAGES C CXX)
option(USE_DEBUG "Set to ON for Debug mode" OFF)

cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH MAIN_PROJECT_DIR)

# Debug flags
if(USE_DEBUG)
    add_definitions(-DDEBUG)
endif()

if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -w"
  )
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
endif()

# Header file directory
set(StochTree_HEADER_DIR ${MAIN_PROJECT_DIR}/include)

# boost.math header file directory
set(BOOSTMATH_HEADER_DIR ${MAIN_PROJECT_DIR}/dependencies/boost_math/include)

# Eigen header file directory
set(EIGEN_HEADER_DIR ${MAIN_PROJECT_DIR}/dependencies/eigen)

# Aggregate the source files underpinning the implementation in the C++ library
file(
  GLOB 
  SOURCES 
  ${MAIN_PROJECT_DIR}/src/config.cpp
  ${MAIN_PROJECT_DIR}/src/cutpoint_candidates.cpp
  ${MAIN_PROJECT_DIR}/src/data.cpp
  ${MAIN_PROJECT_DIR}/src/interface.cpp
  ${MAIN_PROJECT_DIR}/src/io.cpp
  ${MAIN_PROJECT_DIR}/src/json11.cpp
  ${MAIN_PROJECT_DIR}/src/model/univariate_xbart.cpp
  ${MAIN_PROJECT_DIR}/src/model/univariate_bart.cpp
  ${MAIN_PROJECT_DIR}/src/model_draw.cpp
  ${MAIN_PROJECT_DIR}/src/optimization.cpp
  ${MAIN_PROJECT_DIR}/src/partition_tracker.cpp
  ${MAIN_PROJECT_DIR}/src/tree.cpp
)

# Define the C++ source code as a target
add_library(stochtree_objs OBJECT ${SOURCES})

# Include the headers in the source library
target_include_directories(stochtree_objs PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR})

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

# Add pybind11 module
add_subdirectory(pybind11)
pybind11_add_module(stochtree_cpp src/stochtree.cpp)

# Link to C++ source and headers
target_include_directories(stochtree_cpp PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR})
target_link_libraries(stochtree_cpp PRIVATE stochtree_objs)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(stochtree_cpp
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
