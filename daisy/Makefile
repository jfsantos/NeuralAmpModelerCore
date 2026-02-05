# NAM Benchmark for Daisy Seed
# Project Name
TARGET = NAMDaisy

# Library Locations
LIBDAISY_DIR = ../../../libDaisy
DAISYSP_DIR = ../../../DaisySP

# Enable FatFS for SD card access
USE_FATFS = 1

# Optimization flags for ARM Cortex-M7
# Override the default -O2 with more aggressive optimizations
OPT = -O3

# Additional optimization flags (appended after core Makefile is included)
# -ffast-math: Allow aggressive FP optimizations (safe for audio DSP)
# -funroll-loops: Unroll small loops for better pipelining
# -ftree-vectorize: Enable auto-vectorization (implicit with -O3 but explicit)
# -fmove-loop-invariants: Re-enable loop invariant motion (disabled by libDaisy core)
NAM_OPT_FLAGS = -ffast-math -funroll-loops -ftree-vectorize -fmove-loop-invariants

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
	../NAM/dtcm.cpp \
	../NAM/get_dsp.cpp \
	../NAM/lstm.cpp \
	../NAM/profiling.cpp \
	../NAM/ring_buffer.cpp \
	../NAM/util.cpp \
	../NAM/wavenet.cpp

# CMSIS-DSP sources for optimized matrix operations on ARM Cortex-M
# These provide hardware-optimized implementations of matrix multiply and vector add
CMSIS_DSP_DIR = $(LIBDAISY_DIR)/Drivers/CMSIS/DSP/Source
C_SOURCES += \
	$(CMSIS_DSP_DIR)/MatrixFunctions/arm_mat_mult_f32.c \
	$(CMSIS_DSP_DIR)/MatrixFunctions/arm_mat_init_f32.c \
	$(CMSIS_DSP_DIR)/BasicMathFunctions/arm_add_f32.c

# Include paths
# Note: NAM headers use #include "json.hpp" so we need nlohmann in the path
C_INCLUDES += \
	-I.. \
	-I../NAM \
	-I../Dependencies/eigen \
	-I../Dependencies/nlohmann

# Use float samples (saves memory compared to double)
C_DEFS += -DNAM_SAMPLE_FLOAT

# ARM Cortex-M7 architecture flag
C_DEFS += -D__ARM_ARCH_7EM__

# Enable profiling (comment out to disable)
C_DEFS += -DNAM_PROFILING

# Matrix multiplication optimization options (uncomment ONE):
# Option 1: CMSIS-DSP (auto-enabled on ARM, uses arm_mat_mult_f32)
# Option 2: Inline GEMM (hand-written fused multiply-accumulate)
# Option 3: Eigen fallback (disable both for default Eigen behavior)
#
# Using inline GEMM (Option 2) - optimized for small matrices:
C_DEFS += -DNAM_DISABLE_CMSIS_DSP -DNAM_USE_INLINE_GEMM
#
# Uncomment for Option 1 (CMSIS-DSP):
# (no defines needed - CMSIS-DSP is enabled by default on ARM)
#
# Uncomment for Option 3 (pure Eigen):
# C_DEFS += -DNAM_DISABLE_CMSIS_DSP

# Use C++17 for std::optional support
CPP_STANDARD = -std=gnu++17

APP_TYPE = BOOT_QSPI

# Core location and generic Makefile
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core

include $(SYSTEM_FILES_DIR)/Makefile

# Enable C++ exceptions AFTER include (core Makefile sets -fno-exceptions)
# NAM library uses std::runtime_error for error handling
CPPFLAGS += -fexceptions

# Apply NAM optimization flags
CPPFLAGS += $(NAM_OPT_FLAGS)
CFLAGS += $(NAM_OPT_FLAGS)

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
