cmake_minimum_required(VERSION 3.9)
project(v2d LANGUAGES CUDA CXX)
project(v2d)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS})

add_executable(v2d main.cu)
target_link_libraries( v2d ${OpenCV_LIBS} ${Boost_LIBRARIES})
