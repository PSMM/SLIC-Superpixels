cmake_minimum_required(VERSION 3.10)

project(SLIC-Superpixels)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file (GLOB MY_HEADER_FILES "${PROJECT_SOURCE_DIR}/*.h")
file (GLOB MY_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cpp")

add_executable(SLIC-Superpixels ${MY_SOURCE_FILES} ${MY_HEADER_FILES})

# Link libraries
target_link_libraries(SLIC-Superpixels ${OpenCV_LIBS})

# Check if using MSVC compiler
if(MSVC)
    # Enable parallel build
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    # Set CMake flag to use static runtime library
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # Alternatively, set compiler flags to explicitly specify /MT and /MTd
    # Set /MT, /MTd for all targets
    foreach(flag_var
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endif()
