cmake_minimum_required(VERSION 3.7)
project(hand_recognition)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )

set(SOURCE_FILES main.cpp)
add_executable(hand_recognition ${SOURCE_FILES})
target_link_libraries( hand_recognition ${OpenCV_LIBS} )