cmake_minimum_required(VERSION 3.10)
project(IntermediateProjectCV)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(IntermediateProjectCV
        features_detection_TEST.cpp)

target_link_libraries(IntermediateProjectCV ${OpenCV_LIBS})


