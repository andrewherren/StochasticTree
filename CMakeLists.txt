# Build options
option(USE_DEBUG "Set to ON for Debug mode" ON)
option(USE_SANITIZER "Use santizer flags" OFF)
option(BUILD_TEST "Build C++ tests with Google Test" ON)
option(BUILD_DEBUG_TARGETS "Build Standalone C++ Programs for Debugging" ON)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to CMake 3.16
cmake_minimum_required(VERSION 3.16)

# Define the project
project(stochtree LANGUAGES C CXX)

# Sanitizer support
set(
  ENABLED_SANITIZERS
  "address" "undefined"
  CACHE
  STRING
  "Semicolon separated list of sanitizer names, e.g., 'address;leak'. \
Supported sanitizers are address, leak, undefined and thread."
)
if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

# Debug flags
if(USE_DEBUG)
    add_definitions(-DDEBUG)
endif()

if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pthread -w"
  )
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
endif()

# Header file directory
set(StochTree_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

# boost.math header file directory
set(BOOSTMATH_HEADER_DIR ${PROJECT_SOURCE_DIR}/dependencies/boost_math/include)

# Eigen header file directory
set(EIGEN_HEADER_DIR ${PROJECT_SOURCE_DIR}/dependencies/eigen)

# simdjson header file directory
set(SIMDJSON_HEADER_DIR ${PROJECT_SOURCE_DIR}/dependencies/simdjson_singleheader)

# Library directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Aggregate the source files underpinning the implementation in the C++ library
file(
  GLOB 
  SOURCES 
  src/container.cpp
  src/cutpoint_candidates.cpp
  src/data.cpp
  src/io.cpp
  src/json11.cpp
  src/leaf_model.cpp
  src/partition_tracker.cpp
  src/random_effects.cpp
  src/tree.cpp
)

# Define the simdjson source as a compilation target
add_library(simdjson_objs OBJECT ${SIMDJSON_HEADER_DIR}/simdjson.cpp)

# Include the headers in the source library
target_include_directories(simdjson_objs PRIVATE ${SIMDJSON_HEADER_DIR})

# Define the C++ source code as a target
add_library(stochtree_objs OBJECT ${SOURCES})

# Include the headers in the source library
target_include_directories(stochtree_objs PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${SIMDJSON_HEADER_DIR})

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

if(BUILD_TEST)
  # Download the GoogleTest dependency if necessary
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.14.0
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Build test suite
  enable_testing()
  include(GoogleTest)
  file(GLOB CPP_TEST_SOURCES test/cpp/*.cpp)
  add_executable(teststochtree ${CPP_TEST_SOURCES})
  set(STOCHTREE_TEST_HEADER_DIR ${PROJECT_SOURCE_DIR}/test/cpp)
  target_include_directories(teststochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${SIMDJSON_HEADER_DIR} ${STOCHTREE_TEST_HEADER_DIR})
  target_link_libraries(teststochtree PRIVATE stochtree_objs simdjson_objs GTest::gtest_main)
  gtest_discover_tests(teststochtree)
endif()

if(BUILD_DEBUG_TARGETS)
  # Debug executable
  add_executable(debugstochtree debug/cpp/api_debug.cpp)
  set(StochTree_DEBUG_HEADER_DIR ${PROJECT_SOURCE_DIR}/debug/cpp)
  target_include_directories(debugstochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${SIMDJSON_HEADER_DIR} ${StochTree_DEBUG_HEADER_DIR})
  target_link_libraries(debugstochtree PRIVATE stochtree_objs simdjson_objs)
endif()

