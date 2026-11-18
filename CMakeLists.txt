cmake_minimum_required(VERSION 3.16)
project(ZIKA-AI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de compilação" FORCE)
endif()

option(ZIKA_NATIVE "Otimizar para o CPU desta máquina (-march=native); desligar para binários portáteis" ON)
option(ZIKA_BUILD_TESTS "Compilar os testes" ON)

find_package(OpenMP)

# ---- Biblioteca do núcleo ----
add_library(zika_core STATIC
  core/src/format.cpp
  core/src/spec.cpp
  core/src/ops.cpp
  core/src/transformer.cpp
  core/src/optimizer.cpp
  core/src/training.cpp
)
target_include_directories(zika_core PUBLIC core/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(zika_core PUBLIC -Wall -Wextra)
  if(ZIKA_NATIVE)
    target_compile_options(zika_core PUBLIC -march=native)
  endif()
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(zika_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---- Executável zika ----
add_executable(zika cli/main.cpp)
target_link_libraries(zika PRIVATE zika_core)
target_compile_definitions(zika PRIVATE ZIKA_VERSION="\${PROJECT_VERSION}")

# ---- Testes ----
if(ZIKA_BUILD_TESTS)
  enable_testing()
  add_executable(zika_tests tests/test_format.cpp)
  target_link_libraries(zika_tests PRIVATE zika_core)
  add_test(NAME formato COMMAND zika_tests)

  add_executable(zika_core_tests tests/test_core.cpp)
  target_link_libraries(zika_core_tests PRIVATE zika_core)
  add_test(NAME core_gradients COMMAND zika_core_tests)

  add_executable(zika_training_tests tests/test_training.cpp)
  target_link_libraries(zika_training_tests PRIVATE zika_core)
  add_test(NAME training_real_dialogues_checkpoint_resume COMMAND zika_training_tests)
endif()
