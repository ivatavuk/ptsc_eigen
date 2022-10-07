cmake_minimum_required(VERSION 2.8.3)
project(ptsc_eigen)

# Find Eigen
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
endif()

find_package(osqp REQUIRED)

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

include_directories(include
  ${EIGEN3_INCLUDE_DIR}
)

set(LIBRARY_TARGET_NAME ptsc_eigen)
add_library(${LIBRARY_TARGET_NAME}
  src/PtscEigen.cpp
  src/OsqpEigenOptimization.cpp
)
target_link_libraries(${LIBRARY_TARGET_NAME}
  PRIVATE osqp::osqp
  OsqpEigen::OsqpEigen
)

install(TARGETS ${LIBRARY_TARGET_NAME}
  EXPORT  ${PROJECT_NAME}
  COMPONENT runtime
  LIBRARY       DESTINATION lib                            COMPONENT shlib
  ARCHIVE       DESTINATION lib                            COMPONENT lib
  RUNTIME       DESTINATION bin                            COMPONENT bin)

install(FILES include/PtscEigen.hpp include/QpProblem.hpp 
        DESTINATION "include/${LIBRARY_TARGET_NAME}")

add_executable(test_example_unconstrained
  tests/test_example_unconstrained.cpp
)
target_link_libraries(test_example_unconstrained
  PRIVATE osqp::osqp
  OsqpEigen::OsqpEigen
  ${LIBRARY_TARGET_NAME}
)

add_executable(test_example_bound_constrained
  tests/test_example_bound_constrained.cpp
)
target_link_libraries(test_example_bound_constrained
  PRIVATE osqp::osqp
  OsqpEigen::OsqpEigen
  ${LIBRARY_TARGET_NAME}
)

add_executable(test_example_fully_constrained
  tests/test_example_fully_constrained.cpp
)
target_link_libraries(test_example_fully_constrained
  PRIVATE osqp::osqp
  OsqpEigen::OsqpEigen
  ${LIBRARY_TARGET_NAME}
)