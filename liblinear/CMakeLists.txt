
cmake_minimum_required(VERSION 2.8) 
project(liblinear)
#find_package(OpenCV REQUIRED)
#set(CMAKE_CXX_FLAGS "-g")
#set(NAME_SRC src/main.cpp)
#set(NAME_HEADERS include/header.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/blas/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/blas/)
#link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
add_executable(lin-train train.cpp linear.cpp tron.cpp)
add_executable(lin-predict predict.cpp linear.cpp tron.cpp)
#target_link_libraries(svm-predict ${OpenCV_LIBS} armadillo)
target_link_libraries(lin-train armadillo blas.a)
target_link_libraries(lin-predict armadillo blas.a)


