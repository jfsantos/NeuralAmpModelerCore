#include "get_dsp_fatfs.h"

#include <cstring>
#include <vector>

#include "daisy_core.h"
#include "fatfs.h"
#include "json.hpp"
#include "NAM/get_dsp.h"

namespace nam
{

// Buffer size for reading file in chunks
static constexpr size_t READ_CHUNK_SIZE = 512;

// DMA-accessible buffer for f_read operations
// SDMMC peripheral can ONLY access D1 AXI SRAM (0x24000000 range)
// NOT D2 SRAM (.sram1_bss at 0x30000000)
// Regular static variables go to .bss which is in D1 SRAM
static char __attribute__((aligned(4))) s_readBuffer[READ_CHUNK_SIZE];

// FIL structure must also be in D1 RAM (not on stack, not in D2)
static FIL __attribute__((aligned(4))) s_file;

// Error tracking
static LoadError s_lastError = LoadError::OK;
static int s_lastFresult = 0;

LoadError get_dsp_fatfs_last_error()
{
    return s_lastError;
}

int get_dsp_fatfs_last_fresult()
{
    return s_lastFresult;
}

const char* get_dsp_fatfs_error_string(LoadError error)
{
    switch (error)
    {
        case LoadError::OK:
            return "OK";
        case LoadError::FILE_OPEN_FAILED:
            return "Failed to open file";
        case LoadError::FILE_EMPTY:
            return "File is empty";
        case LoadError::FILE_READ_FAILED:
            return "Failed to read file";
        case LoadError::JSON_PARSE_FAILED:
            return "Failed to parse JSON";
        case LoadError::MODEL_CREATE_FAILED:
            return "Failed to create model";
        default:
            return "Unknown error";
    }
}

std::unique_ptr<DSP> get_dsp_fatfs(const char* filename)
{
    dspData temp;
    return get_dsp_fatfs(filename, temp);
}

std::unique_ptr<DSP> get_dsp_fatfs(const char* filename, dspData& returnedConfig)
{
    // Use static DMA-accessible FIL structure
    FIL& file = s_file;
    FRESULT result;

    // Reset error state
    s_lastError = LoadError::OK;
    s_lastFresult = 0;

    // Open the file
    result = f_open(&file, filename, FA_READ);
    if (result != FR_OK)
    {
        s_lastError = LoadError::FILE_OPEN_FAILED;
        s_lastFresult = static_cast<int>(result);
        return nullptr;
    }

    // Get file size
    FSIZE_t fileSize = f_size(&file);
    if (fileSize == 0)
    {
        s_lastError = LoadError::FILE_EMPTY;
        f_close(&file);
        return nullptr;
    }

    // Allocate buffer for file contents
    std::vector<char> buffer(fileSize + 1);

    // Read the entire file using DMA-accessible chunk buffer
    UINT bytesRead = 0;
    UINT totalBytesRead = 0;
    char* bufferPtr = buffer.data();

    while (totalBytesRead < fileSize)
    {
        size_t toRead = fileSize - totalBytesRead;
        if (toRead > READ_CHUNK_SIZE)
        {
            toRead = READ_CHUNK_SIZE;
        }

        result = f_read(&file, s_readBuffer, toRead, &bytesRead);
        if (result != FR_OK)
        {
            s_lastError = LoadError::FILE_READ_FAILED;
            s_lastFresult = static_cast<int>(result);
            f_close(&file);
            return nullptr;
        }

        // Copy from DMA buffer to main buffer
        memcpy(bufferPtr + totalBytesRead, s_readBuffer, bytesRead);
        totalBytesRead += bytesRead;

        if (bytesRead == 0)
        {
            break;
        }
    }

    f_close(&file);

    // Null-terminate the buffer
    buffer[totalBytesRead] = '\0';

    // Parse the JSON
    nlohmann::json j;
    try
    {
        j = nlohmann::json::parse(buffer.data(), buffer.data() + totalBytesRead);
    }
    catch (const nlohmann::json::parse_error&)
    {
        s_lastError = LoadError::JSON_PARSE_FAILED;
        return nullptr;
    }

    // Use the existing get_dsp overload that takes a JSON object
    // This populates returnedConfig and creates the DSP object
    std::unique_ptr<DSP> dsp;
    try
    {
        dsp = get_dsp(j, returnedConfig);
    }
    catch (const std::bad_alloc&)
    {
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }
    catch (const std::exception&)
    {
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }

    if (!dsp)
    {
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }

    return dsp;
}

} // namespace nam
