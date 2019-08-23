cmake_minimum_required(VERSION 3.9)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "")

set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(buddhabrot CXX CUDA)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.14/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(CONANFILE conanfile.txt
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

set(CMAKE_CXX_STANDARD 20)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O3 -use_fast_math -lineinfo -g --ptxas-options=-v")

if (MSVC)
set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS -utf-8 -permissive- -EHs-c- -D_HAS_EXCEPTIONS=0 -GR- -GS- -Zc:inline -arch:AVX2 -MP -Z7")
string(APPEND CMAKE_CXX_FLAGS " -W4 -WX -wd4100 -wd4996 -wd4505 -w44062")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -Zo -JMC -MDd") # /Oy- ?
set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob3 /UNDEBUG /GL -fp:fast")
link_libraries(optimized -DEBUG -OPT:REF -OPT:ICF -LTCG:INCREMENTAL)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
set(CMAKE_CXX_FLAGS "-march=native -fno-exceptions -fno-rtti -fvisibility=hidden")
string(APPEND CMAKE_CXX_FLAGS " -Werror -Wall -Wextra -Wno-unused-parameter -Wconversion -Wcast-qual -Wsign-conversion -Wfloat-conversion -Wmultiple-inheritance -Werror=virtual-inheritance -Wshadow=local")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -UNDEBUG -ffast-math")
add_link_options(-Wl,--as-needed,--no-undefined,--gc-sections -fsanitize=leak)
endif()

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS src/*)
add_executable(buddhabrot ${SRCS} conanfile.txt)
conan_target_link_libraries(buddhabrot)
#target_include_directories(projectname SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/include)
#target_link_directories(projectname PRIVATE ${CMAKE_BINARY_DIR}/lib)
#target_link_libraries(projectname debug libd optimized lib)
