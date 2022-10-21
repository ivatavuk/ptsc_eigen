cmake_minimum_required(VERSION 2.8.3)
project(ptsc_eigen)

find_package(OsqpEigen REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

option(BUILD_SHARED_LIBS "Build libraries as shared as opposed to static" ON)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Choose the type of build, recommanded options are: Debug or Release" FORCE)
endif()

include(CMakePackageConfigHelpers)

include_directories(include)

set(LIBRARY_TARGET_NAME PtscEigen)
set(LIBRARY_VERSION 0.0)
set(LIBRARY_HEADERS 
  include/PtscEigen.hpp 
  include/QpProblem.hpp
  include/OsqpEigenOptimization.hpp
)

add_library(${LIBRARY_TARGET_NAME}
  src/PtscEigen.cpp
  src/OsqpEigenOptimization.cpp
)
target_link_libraries(${LIBRARY_TARGET_NAME}
  PUBLIC OsqpEigen::OsqpEigen
)
install(TARGETS ${LIBRARY_TARGET_NAME}
  EXPORT  PtscEigenTargets
  COMPONENT runtime
  LIBRARY   DESTINATION   lib   
  ARCHIVE   DESTINATION   lib
  RUNTIME   DESTINATION   bin
  INCLUDES  DESTINATION   include/${LIBRARY_TARGET_NAME})

install(FILES ${LIBRARY_HEADERS}
        DESTINATION "include/${LIBRARY_TARGET_NAME}")

install(EXPORT PtscEigenTargets
  FILE PtscEigenTargets.cmake
  NAMESPACE PtscEigen::
  DESTINATION lib/cmake/${LIBRARY_TARGET_NAME})

write_basic_package_version_file(
  "PtscEigenConfigVersion.cmake"
  VERSION ${LIBRARY_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES "cmake/PtscEigenConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/PtscEigenConfigVersion.cmake"
        DESTINATION lib/cmake/${LIBRARY_TARGET_NAME})

add_custom_target(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake"
)

#-----------------tests-----------------
add_executable(test_example_unconstrained
  tests/test_example_unconstrained.cpp
)
target_link_libraries(test_example_unconstrained
  PRIVATE ${LIBRARY_TARGET_NAME}
)

add_executable(test_example_bound_constrained
  tests/test_example_bound_constrained.cpp
)
target_link_libraries(test_example_bound_constrained
  PRIVATE ${LIBRARY_TARGET_NAME}
)

add_executable(test_example_fully_constrained
  tests/test_example_fully_constrained.cpp
)
target_link_libraries(test_example_fully_constrained
  PRIVATE ${LIBRARY_TARGET_NAME}
)