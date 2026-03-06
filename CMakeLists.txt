cmake_minimum_required(VERSION 3.10)

# Make sure this matches ./NAM/version.h!
project(NAM VERSION 0.4.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use libc++ on macOS, system default on Linux
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	include_directories(SYSTEM /usr/local/include)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	link_libraries(stdc++fs)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
	message(FATAL_ERROR "Unrecognized Platform!")
endif()

set(NAM_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies")
include_directories(SYSTEM "${NAM_DEPS_PATH}/eigen")

# Per-operation tuning config: path to auto-generated header from nam_autotune
set(NAM_TUNING_CONFIG "" CACHE FILEPATH "Path to auto-generated tuning config header (from nam_autotune)")
if(NAM_TUNING_CONFIG)
  get_filename_component(NAM_TUNING_CONFIG_DIR "${NAM_TUNING_CONFIG}" DIRECTORY)
  add_compile_definitions(NAM_TUNING_CONFIG)
  # Add the config directory to include path so #include "nam_tuning_config.h" works
  include_directories("${NAM_TUNING_CONFIG_DIR}")
  # Also force-include the config header so macros are set before any source compilation
  add_compile_options("-include" "${NAM_TUNING_CONFIG}")
endif()

add_subdirectory(tools)

#file(MAKE_DIRECTORY build/tools)

#add_custom_target(copy_tools ALL
#	${CMAKE_COMMAND} -E copy "$<TARGET_FILE:tools>" tools/
#	DEPENDS tools
#)
