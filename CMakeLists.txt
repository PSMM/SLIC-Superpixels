cmake_minimum_required(VERSION 2.8.3)
project(SLIC-demo)


find_package(OpenCV 3 REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)


add_executable( test_slic_v2
    test_slic_v2.cpp
    slic.cpp
    )


target_link_libraries( test_slic_v2  ${OpenCV_LIBRARIES} )
