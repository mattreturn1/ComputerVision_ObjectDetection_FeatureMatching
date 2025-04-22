cmake_minimum_required(VERSION 3.10)
project(IntermediateProjectCV)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Percorso per OpenCV — personalizza se necessario
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# Includi la cartella headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Aggiungi tutti i file sorgente
set(SOURCES
        src/ObjectDetector.cpp
        src/FeatureFactory.cpp
        src/main.cpp
        src/Preprocessing.cpp
        src/Filter.cpp
        src/Output.cpp
        include/Output.hpp
)

# Crea l'eseguibile
add_executable(IntermediateProjectCV ${SOURCES})

# Linka OpenCV
target_link_libraries(IntermediateProjectCV ${OpenCV_LIBS})
