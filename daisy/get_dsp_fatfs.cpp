#include "get_dsp_fatfs.h"

#include <cstring>
#include <new>
#include <vector>

#include "daisy_core.h"
#include "dev/sdram.h"
#include "fatfs.h"
#include "json.hpp"
#include "NAM/get_dsp.h"

// =============================================================================
// SDRAM Arena Allocator
//
// During JSON parsing of large .nam files, the file buffer (~900 KB) and
// nlohmann::json DOM (~1-2 MB) far exceed the STM32H750's 512 KB internal
// SRAM heap. We use a bump allocator over a static SDRAM buffer to handle
// these temporary allocations.
//
// The arena is enabled ONLY during file reading and JSON parsing. It is
// disabled before get_dsp() so that the model itself allocates from the
// regular heap (fast internal SRAM). After get_dsp() copies what it needs,
// the JSON DOM is destroyed (delete calls are no-ops for arena memory) and
// the arena is reset.
//
// This is safe because:
//   - Model loading is single-threaded
//   - The arena is only active for a bounded scope
//   - All arena memory is freed at once (no fragmentation)
//   - Fallback to malloc if arena is full or disabled
// =============================================================================

static constexpr size_t SDRAM_ARENA_SIZE = 4 * 1024 * 1024; // 4 MB

static char DSY_SDRAM_BSS s_sdramArena[SDRAM_ARENA_SIZE];
static size_t s_arenaOffset = 0;
static bool s_arenaActive = false;

// Usage tracking — survives arena_reset() so caller can read after loading
static size_t s_arenaPeakUsed = 0;
static size_t s_arenaOverflowCount = 0;

// Phase timing (microseconds) — survives across calls so caller can read after loading
static uint32_t s_timingSdReadUs = 0;
static uint32_t s_timingJsonParseUs = 0;
static uint32_t s_timingModelCreateUs = 0;

static inline bool is_arena_ptr(void* p)
{
    return p >= s_sdramArena && p < s_sdramArena + SDRAM_ARENA_SIZE;
}

static void* arena_alloc(size_t size)
{
    // Align to 8 bytes for safety
    size = (size + 7) & ~7;
    if (s_arenaOffset + size > SDRAM_ARENA_SIZE)
    {
        // Arena full — fall back to regular heap
        s_arenaOverflowCount++;
        return malloc(size);
    }
    void* ptr = s_sdramArena + s_arenaOffset;
    s_arenaOffset += size;
    if (s_arenaOffset > s_arenaPeakUsed)
        s_arenaPeakUsed = s_arenaOffset;
    return ptr;
}

static void arena_reset()
{
    s_arenaOffset = 0;
}

// =============================================================================
// Global operator new/delete overrides
//
// When the arena is active, all allocations go to SDRAM. When inactive,
// they go through malloc/free as normal. The delete overrides check if the
// pointer is in the arena range — if so, it's a no-op (bump allocator).
// =============================================================================

void* operator new(size_t size)
{
    if (s_arenaActive)
    {
        return arena_alloc(size);
    }
    void* p = malloc(size);
    if (!p)
        throw std::bad_alloc();
    return p;
}

void* operator new[](size_t size)
{
    if (s_arenaActive)
    {
        return arena_alloc(size);
    }
    void* p = malloc(size);
    if (!p)
        throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept
{
    if (!p)
        return;
    if (is_arena_ptr(p))
        return; // Bump allocator — no individual free
    free(p);
}

void operator delete[](void* p) noexcept
{
    if (!p)
        return;
    if (is_arena_ptr(p))
        return;
    free(p);
}

void operator delete(void* p, size_t) noexcept
{
    if (!p)
        return;
    if (is_arena_ptr(p))
        return;
    free(p);
}

void operator delete[](void* p, size_t) noexcept
{
    if (!p)
        return;
    if (is_arena_ptr(p))
        return;
    free(p);
}

// RAII guard that activates the arena on construction and deactivates + resets
// on destruction (or explicit call). Ensures the arena is always cleaned up.
struct ArenaGuard
{
    ArenaGuard()
    {
        s_arenaActive = true;
        s_arenaOffset = 0;
        s_arenaPeakUsed = 0;
        s_arenaOverflowCount = 0;
    }
    ~ArenaGuard() { release(); }

    // Deactivate the arena but don't reset yet — caller may still hold
    // arena-backed objects that need to be destroyed first.
    void deactivate() { s_arenaActive = false; }

    // Full release: deactivate + reset
    void release()
    {
        s_arenaActive = false;
        arena_reset();
    }

    ArenaGuard(const ArenaGuard&) = delete;
    ArenaGuard& operator=(const ArenaGuard&) = delete;
};

// =============================================================================
// FatFS Model Loader
// =============================================================================

namespace nam
{

// Buffer size for reading file in chunks
static constexpr size_t READ_CHUNK_SIZE = 512;

// DMA-accessible buffer for f_read operations
// SDMMC peripheral can ONLY access D1 AXI SRAM (0x24000000 range)
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

size_t get_dsp_fatfs_arena_peak()
{
    return s_arenaPeakUsed;
}

size_t get_dsp_fatfs_arena_overflows()
{
    return s_arenaOverflowCount;
}

LoadTiming get_dsp_fatfs_timing()
{
    return {s_timingSdReadUs, s_timingJsonParseUs, s_timingModelCreateUs};
}

// Convert DWT cycle count to microseconds.
// SystemCoreClock is defined by CMSIS and set by libDaisy (480 MHz on STM32H750).
static inline uint32_t cycles_to_us(uint32_t cycles)
{
    return cycles / (SystemCoreClock / 1000000);
}

// Internal helper: reads file from SD card and parses JSON.
// File buffer and JSON DOM are allocated in the SDRAM arena.
// On success, j contains parsed JSON and arena is deactivated (but not reset).
// On failure, returns false and sets s_lastError.
static bool load_and_parse_json(const char* filename, nlohmann::json& j, ArenaGuard& arena)
{
    FIL& file = s_file;
    FRESULT result;

    // Open the file
    uint32_t phase_start = DWT->CYCCNT;
    result = f_open(&file, filename, FA_READ);
    if (result != FR_OK)
    {
        s_lastError = LoadError::FILE_OPEN_FAILED;
        s_lastFresult = static_cast<int>(result);
        return false;
    }

    FSIZE_t fileSize = f_size(&file);
    if (fileSize == 0)
    {
        s_lastError = LoadError::FILE_EMPTY;
        f_close(&file);
        return false;
    }

    // Allocate file buffer from SDRAM arena (first allocation, guaranteed contiguous)
    char* fileBuffer = static_cast<char*>(arena_alloc(fileSize + 1));
    if (!fileBuffer)
    {
        s_lastError = LoadError::FILE_READ_FAILED;
        f_close(&file);
        return false;
    }

    // Read the entire file using DMA-accessible chunk buffer
    UINT bytesRead = 0;
    UINT totalBytesRead = 0;

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
            return false;
        }

        memcpy(fileBuffer + totalBytesRead, s_readBuffer, bytesRead);
        totalBytesRead += bytesRead;

        if (bytesRead == 0)
            break;
    }

    f_close(&file);
    fileBuffer[totalBytesRead] = '\0';
    s_timingSdReadUs = cycles_to_us(DWT->CYCCNT - phase_start);

    // Parse JSON — DOM nodes allocated from SDRAM arena via operator new
    phase_start = DWT->CYCCNT;
    try
    {
        j = nlohmann::json::parse(fileBuffer, fileBuffer + totalBytesRead);
    }
    catch (const nlohmann::json::parse_error&)
    {
        s_timingJsonParseUs = cycles_to_us(DWT->CYCCNT - phase_start);
        s_lastError = LoadError::JSON_PARSE_FAILED;
        return false;
    }
    s_timingJsonParseUs = cycles_to_us(DWT->CYCCNT - phase_start);

    // Deactivate arena BEFORE get_dsp — model allocations go to regular heap.
    // Arena is not reset yet; j still references arena memory.
    arena.deactivate();
    return true;
}

// Create model from parsed JSON. Handles error tracking and timing.
// Uses the efficient single-arg get_dsp() — only 1 copy of weights on heap.
static std::unique_ptr<DSP> create_model(const nlohmann::json& j)
{
    uint32_t phase_start = DWT->CYCCNT;
    std::unique_ptr<DSP> dsp;
    try
    {
        dsp = get_dsp(j);
    }
    catch (const std::bad_alloc&)
    {
        s_timingModelCreateUs = cycles_to_us(DWT->CYCCNT - phase_start);
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }
    catch (const std::exception&)
    {
        s_timingModelCreateUs = cycles_to_us(DWT->CYCCNT - phase_start);
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }
    s_timingModelCreateUs = cycles_to_us(DWT->CYCCNT - phase_start);

    if (!dsp)
    {
        s_lastError = LoadError::MODEL_CREATE_FAILED;
    }
    return dsp;
}

// Create model and populate returnedConfig. Uses get_dsp(j, returnedConfig)
// which needs 2 copies of weights (one for returnedConfig, one for model).
static std::unique_ptr<DSP> create_model_with_config(const nlohmann::json& j, dspData& returnedConfig)
{
    uint32_t phase_start = DWT->CYCCNT;
    std::unique_ptr<DSP> dsp;
    try
    {
        dsp = get_dsp(j, returnedConfig);
    }
    catch (const std::bad_alloc&)
    {
        s_timingModelCreateUs = cycles_to_us(DWT->CYCCNT - phase_start);
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }
    catch (const std::exception&)
    {
        s_timingModelCreateUs = cycles_to_us(DWT->CYCCNT - phase_start);
        s_lastError = LoadError::MODEL_CREATE_FAILED;
        return nullptr;
    }
    s_timingModelCreateUs = cycles_to_us(DWT->CYCCNT - phase_start);

    if (!dsp)
    {
        s_lastError = LoadError::MODEL_CREATE_FAILED;
    }
    return dsp;
}

std::unique_ptr<DSP> get_dsp_fatfs(const char* filename)
{
    s_lastError = LoadError::OK;
    s_lastFresult = 0;
    s_timingSdReadUs = 0;
    s_timingJsonParseUs = 0;
    s_timingModelCreateUs = 0;

    std::unique_ptr<DSP> dsp;
    {
        // ---- Arena scope: file buffer + JSON parsing in SDRAM ----
        ArenaGuard arena;

        // Allocate the JSON object itself from the arena (arena is active).
        // All DOM nodes created during parsing also go to the arena.
        // We deliberately skip the destructor — arena reset reclaims everything
        // at once, which is much faster than walking the DOM node-by-node.
        nlohmann::json* j = new nlohmann::json();

        if (!load_and_parse_json(filename, *j, arena))
            return nullptr;  // arena cleanup handles j

        // Create model — only 1 copy of weights on heap (efficient path)
        dsp = create_model(*j);

        // Do NOT delete j — all its memory (the object + DOM nodes) is in the
        // arena. ArenaGuard destructor resets the arena, reclaiming everything.
    }

    return dsp;
}

std::unique_ptr<DSP> get_dsp_fatfs(const char* filename, dspData& returnedConfig)
{
    s_lastError = LoadError::OK;
    s_lastFresult = 0;
    s_timingSdReadUs = 0;
    s_timingJsonParseUs = 0;
    s_timingModelCreateUs = 0;

    std::unique_ptr<DSP> dsp;
    {
        // ---- Arena scope: file buffer + JSON parsing in SDRAM ----
        ArenaGuard arena;

        // Allocate JSON from arena — skip destructor, arena reset reclaims all.
        nlohmann::json* j = new nlohmann::json();

        if (!load_and_parse_json(filename, *j, arena))
            return nullptr;

        // Create model and populate returnedConfig (2 copies of weights)
        dsp = create_model_with_config(*j, returnedConfig);

        // Do NOT delete j — arena reset handles everything.
    }

    return dsp;
}

} // namespace nam
