cmake_minimum_required(VERSION 3.9)
project(b2h LANGUAGES CUDA CXX)
project(b2h)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(b2h main.cu)
target_link_libraries( b2h ${OpenCV_LIBS} )
