
cmake_minimum_required(VERSION 2.8) 
project(name)
#find_package(OpenCV REQUIRED)
#set(CMAKE_CXX_FLAGS "-g")
#set(NAME_SRC src/main.cpp)
#set(NAME_HEADERS include/header.h)
include_directories("./svm/")
#link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
add_executable(svm-predict ./svm/svm-predict.cpp ./svm/svm.cpp)
add_executable(svm-train ./svm/svm-train.cpp ./svm/svm.cpp)
#target_link_libraries(svm-predict ${OpenCV_LIBS} armadillo)
target_link_libraries(svm-predict armadillo)


