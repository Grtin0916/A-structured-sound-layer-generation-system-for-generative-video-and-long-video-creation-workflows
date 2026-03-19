cmake_minimum_required(VERSION 3.16)

project(cmake_eigen_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(cmake_eigen_demo src/main.cpp)
target_link_libraries(cmake_eigen_demo PRIVATE Eigen3::Eigen)