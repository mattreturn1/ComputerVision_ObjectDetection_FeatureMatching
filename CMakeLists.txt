cmake_minimum_required(VERSION 3.10)
project(IntermediateProjectCV)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib") #TODO change with VM path
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
        src/preprocessing.cpp
        src/utils.cpp
        src/ModelsDetector.cpp
        src/TestsDetector.cpp
        src/Output.cpp
        src/metrics.cpp
        src/main.cpp
)

add_executable(IntermediateProjectCV ${SOURCES})

target_link_libraries(IntermediateProjectCV ${OpenCV_LIBS})
