# NAM Benchmark for Daisy Seed
# Project Name
TARGET = NAMDaisy

# Library Locations
LIBDAISY_DIR = ../../../libDaisy
DAISYSP_DIR = ../../../DaisySP

# Enable FatFS for SD card access
USE_FATFS = 1

# Sources - Main application and FatFS loader
CPP_SOURCES = \
	NAMDaisy.cpp \
	get_dsp_fatfs.cpp

# NAM library sources
CPP_SOURCES += \
	../NAM/activations.cpp \
	../NAM/conv1d.cpp \
	../NAM/convnet.cpp \
	../NAM/dsp.cpp \
	../NAM/get_dsp.cpp \
	../NAM/lstm.cpp \
	../NAM/ring_buffer.cpp \
	../NAM/util.cpp \
	../NAM/wavenet.cpp

# Include paths
# Note: NAM headers use #include "json.hpp" so we need nlohmann in the path
C_INCLUDES += \
	-I.. \
	-I../NAM \
	-I../Dependencies/eigen \
	-I../Dependencies/nlohmann

# Use float samples (saves memory compared to double)
C_DEFS += -DNAM_SAMPLE_FLOAT

# Use C++17 for std::optional support
CPP_STANDARD = -std=gnu++17

APP_TYPE = BOOT_QSPI

# Core location and generic Makefile
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# Enable C++ exceptions AFTER include (core Makefile sets -fno-exceptions)
# NAM library uses std::runtime_error for error handling
CPPFLAGS += -fexceptions

# Custom clean targets
# clean-local: Only clean project files (fast rebuild)
# clean-all:   Clean everything including NAM library (full rebuild)
# clean:       Alias for clean-all (default behavior)

# Project-specific object files (fast to rebuild)
LOCAL_OBJS = $(BUILD_DIR)/NAMDaisy.o $(BUILD_DIR)/get_dsp_fatfs.o

# NAM library object files (slow to rebuild)
NAM_OBJS = $(BUILD_DIR)/activations.o $(BUILD_DIR)/conv1d.o $(BUILD_DIR)/convnet.o \
           $(BUILD_DIR)/dsp.o $(BUILD_DIR)/get_dsp.o $(BUILD_DIR)/lstm.o \
           $(BUILD_DIR)/ring_buffer.o $(BUILD_DIR)/util.o $(BUILD_DIR)/wavenet.o

.PHONY: clean-local clean-all

clean-local:
	rm -f $(LOCAL_OBJS)
	rm -f $(BUILD_DIR)/NAMDaisy.d $(BUILD_DIR)/get_dsp_fatfs.d
	rm -f $(BUILD_DIR)/NAMDaisy.lst $(BUILD_DIR)/get_dsp_fatfs.lst
	rm -f $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).bin $(BUILD_DIR)/$(TARGET).hex
	rm -f $(BUILD_DIR)/$(TARGET).map

clean-all: clean
