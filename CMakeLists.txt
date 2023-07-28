cmake_minimum_required(VERSION 3.16)
project(Rectangle_POC)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)
# Enable the target igl::imgui
igl_include(imgui)

include(FetchContent)
FetchContent_Declare(
  tinyad
  GIT_REPOSITORY https://github.com/EliasJadon/TinyAD.git
  GIT_TAG main
)
FetchContent_MakeAvailable(tinyad)

# Add your project files
file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_compile_definitions(${PROJECT_NAME} PUBLIC SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl::imgui TinyAD)
# tinyad needs C++17 (not sure if that means this project does too)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
