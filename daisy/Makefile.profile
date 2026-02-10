# NAM Profiling Tool for Daisy Pod
# Build with: make -f Makefile.profile
# Flash with: make -f Makefile.profile program-dfu

TARGET = NAMProfile

# Library Locations
LIBDAISY_DIR = ../../../libDaisy
DAISYSP_DIR = ../../../DaisySP

# Enable FatFS for SD card access
USE_FATFS = 1

OPT = -O3
NAM_OPT_FLAGS = -ffast-math -funroll-loops -ftree-vectorize -fmove-loop-invariants

# Sources - Profiling application and FatFS loader
CPP_SOURCES = \
	NAMProfile.cpp \
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
CMSIS_DSP_DIR = $(LIBDAISY_DIR)/Drivers/CMSIS/DSP/Source
C_SOURCES += \
	$(CMSIS_DSP_DIR)/MatrixFunctions/arm_mat_mult_f32.c \
	$(CMSIS_DSP_DIR)/MatrixFunctions/arm_mat_init_f32.c \
	$(CMSIS_DSP_DIR)/BasicMathFunctions/arm_add_f32.c

# Include paths
C_INCLUDES += \
	-I.. \
	-I../NAM \
	-I../Dependencies/eigen \
	-I../Dependencies/nlohmann

# Use float samples (same as real-time pedal)
C_DEFS += -DNAM_SAMPLE_FLOAT

# ARM Cortex-M7 architecture flag
C_DEFS += -D__ARM_ARCH_7EM__

# Enable profiling for detailed per-operation breakdown
C_DEFS += -DNAM_PROFILING

# Matrix multiplication: same as pedal build for comparable results
C_DEFS += -DNAM_DISABLE_CMSIS_DSP -DNAM_USE_INLINE_GEMM

# Use C++17 for std::optional support
CPP_STANDARD = -std=gnu++17

APP_TYPE = BOOT_QSPI

# Enable float formatting in printf/vsnprintf (newlib-nano disables it by default)
LDFLAGS = -u _printf_float

# Core location and generic Makefile
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core

include $(SYSTEM_FILES_DIR)/Makefile

# Enable C++ exceptions AFTER include (core Makefile sets -fno-exceptions)
CPPFLAGS += -fexceptions

# Apply NAM optimization flags
CPPFLAGS += $(NAM_OPT_FLAGS)
CFLAGS += $(NAM_OPT_FLAGS)
