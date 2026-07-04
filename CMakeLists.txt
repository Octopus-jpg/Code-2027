cmake_minimum_required(VERSION 3.16)
project(htc_research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(htc_core STATIC
    src/random.cpp
    src/graph.cpp
    src/cascade.cpp
    src/io.cpp
    src/rtw.cpp
    src/im.cpp
    src/ic.cpp
    src/baselines.cpp
    src/estimator.cpp
    src/synthetic.cpp
    src/runtime_clock.cpp
)

target_include_directories(htc_core PUBLIC include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(htc_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(sanity_check apps/sanity_check.cpp)
target_link_libraries(sanity_check PRIVATE htc_core)

add_executable(htc_im apps/run_im.cpp)
target_link_libraries(htc_im PRIVATE htc_core)

add_executable(htc_ie apps/run_ie.cpp)
target_link_libraries(htc_ie PRIVATE htc_core)

add_executable(bench_m5 apps/bench_m5.cpp)
target_link_libraries(bench_m5 PRIVATE htc_core)

add_executable(compare_opim11 apps/compare_opim11.cpp)
target_link_libraries(compare_opim11 PRIVATE htc_core)

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_executable(test_rtw_equivalence tests/test_rtw_equivalence.cpp)
    target_link_libraries(test_rtw_equivalence PRIVATE htc_core)

    add_executable(test_rtw_incremental_vs_oracle tests/test_rtw_incremental_vs_oracle.cpp)
    target_link_libraries(test_rtw_incremental_vs_oracle PRIVATE htc_core)

    add_executable(test_rtw_greedy tests/test_rtw_greedy.cpp)
    target_link_libraries(test_rtw_greedy PRIVATE htc_core)

    add_executable(test_baselines tests/test_baselines.cpp)
    target_link_libraries(test_baselines PRIVATE htc_core)

    add_executable(test_estimator tests/test_estimator.cpp)
    target_link_libraries(test_estimator PRIVATE htc_core)

    add_executable(test_synthetic_smoke tests/test_synthetic_smoke.cpp)
    target_link_libraries(test_synthetic_smoke PRIVATE htc_core)

    add_executable(test_opim_proxy tests/test_opim_proxy.cpp)
    target_link_libraries(test_opim_proxy PRIVATE htc_core)

    add_executable(test_jsonl_output tests/test_jsonl_output.cpp)
    target_link_libraries(test_jsonl_output PRIVATE htc_core)

    add_executable(test_io_loader tests/test_io_loader.cpp)
    target_link_libraries(test_io_loader PRIVATE htc_core)

    add_test(NAME test_rtw_equivalence COMMAND test_rtw_equivalence)
    add_test(NAME test_rtw_incremental_vs_oracle COMMAND test_rtw_incremental_vs_oracle)
    add_test(NAME test_rtw_greedy COMMAND test_rtw_greedy)
    add_test(NAME test_baselines COMMAND test_baselines)
    add_test(NAME test_estimator COMMAND test_estimator)
    add_test(NAME test_synthetic_smoke COMMAND test_synthetic_smoke)
    add_test(NAME test_opim_proxy COMMAND test_opim_proxy)
    add_test(NAME test_jsonl_output COMMAND test_jsonl_output)
    add_test(NAME test_io_loader COMMAND test_io_loader)
elseif(BUILD_TESTING)
    message(STATUS "BUILD_TESTING=ON but tests/ directory not found; skipping test targets.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sanity_check PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(htc_im PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(htc_ie PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(bench_m5 PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(compare_opim11 PRIVATE -Wall -Wextra -Wpedantic)
    if(TARGET test_rtw_equivalence)
        target_compile_options(test_rtw_equivalence PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_rtw_incremental_vs_oracle PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_rtw_greedy PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_baselines PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_estimator PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_synthetic_smoke PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_opim_proxy PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_jsonl_output PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_io_loader PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
