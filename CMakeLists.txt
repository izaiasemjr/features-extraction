cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pipeline-features-keypoints)

find_package(PCL 1.8 REQUIRED)

set (CMAKE_CXX_STANDARD 14)
#list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

set(PATH_SOURCE ./local_include)
set(PATH_SOURCE ./local_include/*)


#set(${PATH_SOURCE}/fe atures.hpp  ${PATH_SOURCE}/keypoints.hpp
#    ${PATH_SOURCE}/utils.hpp)

include_directories(${PCL_INCLUDE_DIRS} ${PATH_SOURCE}/ )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (${PROJECT_NAME} main.cpp ${SOURCES})
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})
