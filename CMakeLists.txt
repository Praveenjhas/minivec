cmake_minimum_required(VERSION 3.20)

project(MiniVec
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(minivec
    src/cosine_similarity.cpp
    src/brute_force_index.cpp
    src/vector_store.cpp
)

target_include_directories(minivec
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(minivec PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(minivec_demo
    examples/main.cpp
)

target_link_libraries(minivec_demo
    PRIVATE minivec
)

enable_testing()

find_package(GTest REQUIRED)

add_executable(minivec_tests
    tests/cosine_similarity_test.cpp
    tests/brute_force_index_test.cpp
)

target_link_libraries(minivec_tests
    PRIVATE
        minivec
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(minivec_tests)