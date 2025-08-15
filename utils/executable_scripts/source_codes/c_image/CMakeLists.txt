cmake_minimum_required(VERSION 3.17)
project(c_image C)

set(CMAKE_C_STANDARD 11)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -fomit-frame-pointer -march=native -flto")

#link_directories(/home/xlazarik/gsl/)
#link_directories(/home/xlazarik/gsl/include)
#include_directories(/home/xlazarik/gsl/include/)
add_executable(c_image main.c)
#add_library(lgsl)
#add_library(lgslcblas)
#add_library(lm)
#link_libraries(lgsl lgslcblas lm)

find_package(GSL REQUIRED)    # See below (2)
target_link_libraries(c_image GSL::gsl GSL::gslcblas)