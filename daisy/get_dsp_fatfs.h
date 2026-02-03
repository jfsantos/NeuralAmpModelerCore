#pragma once

#include <memory>
#include <string>

#include "NAM/dsp.h"

namespace nam
{

/// \brief Error codes for get_dsp_fatfs
enum class LoadError
{
    OK = 0,
    FILE_OPEN_FAILED,
    FILE_EMPTY,
    FILE_READ_FAILED,
    JSON_PARSE_FAILED,
    MODEL_CREATE_FAILED
};

/// \brief Get the last error from get_dsp_fatfs
/// \return The error code from the last call to get_dsp_fatfs
LoadError get_dsp_fatfs_last_error();

/// \brief Get the FatFS result code from the last file operation
/// \return The FRESULT code (cast to int) from the last failed file operation
int get_dsp_fatfs_last_fresult();

/// \brief Get a human-readable error message
/// \param error The error code
/// \return A string describing the error
const char* get_dsp_fatfs_error_string(LoadError error);

/// \brief Load a NAM model from an SD card file using FatFS
///
/// This function is a FatFS-based alternative to get_dsp() that works on
/// embedded systems without std::filesystem support.
///
/// \param filename Path to the .nam file on the SD card (e.g., "model.nam")
/// \return Unique pointer to a DSP object, or nullptr if loading failed
std::unique_ptr<DSP> get_dsp_fatfs(const char* filename);

/// \brief Load a NAM model from an SD card file using FatFS
///
/// This overload also returns the parsed model data.
///
/// \param filename Path to the .nam file on the SD card
/// \param returnedConfig Output parameter that will be filled with the model data
/// \return Unique pointer to a DSP object, or nullptr if loading failed
std::unique_ptr<DSP> get_dsp_fatfs(const char* filename, dspData& returnedConfig);

} // namespace nam
