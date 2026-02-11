// NAM Profiling Tool for Daisy Pod
//
// Non-realtime benchmark of all .nam models found on the SD card.
// Scans folders in the root of the SD card (non-recursive) for .nam files.
// For each model, processes 2 seconds of pseudo-random audio and
// reports wall-clock time, realtime ratio, and profiling breakdown.
//
// Output is printed over USB serial AND written to profiling.log
// inside each folder.
//
// LED1 = blue while profiling, green when done, red on SD error.

#include <cstdio>
#include <cstring>
#include <malloc.h>

#include "daisy_pod.h"
#include "fatfs.h"

#include "NAM/dsp.h"
#include "NAM/dtcm.h"
#include "NAM/activations.h"
#include "NAM/profiling.h"
#include "get_dsp_fatfs.h"

using namespace daisy;

// Uncomment to disable D-Cache for profiling comparison
// #define NAM_PROFILE_NO_DCACHE

// Uncomment to zero all model weights after loading.
// Useful for isolating numerical issues (NaN/Inf propagation from trained weights)
// vs structural performance differences.
#define NAM_PROFILE_ZERO_WEIGHTS

// Configuration
static constexpr size_t AUDIO_BUFFER_SIZE = 48;
static constexpr float SAMPLE_RATE = 48000.0f;
static constexpr size_t NUM_SECONDS = 2;
static constexpr size_t NUM_BUFFERS = (size_t)(SAMPLE_RATE / AUDIO_BUFFER_SIZE) * NUM_SECONDS;

// Folder management
static constexpr size_t MAX_FOLDERS = 16;
static constexpr size_t MAX_FOLDER_NAME_LEN = 64;
static char s_folders[MAX_FOLDERS][MAX_FOLDER_NAME_LEN];
static size_t s_numFolders = 0;

// Per-folder .nam file list (full paths like "/folder/model.nam")
static constexpr size_t MAX_NAM_FILES = 64;
static constexpr size_t MAX_PATH_LEN = 128;
static char s_namFiles[MAX_NAM_FILES][MAX_PATH_LEN];
static size_t s_numNamFiles = 0;

// Hardware
static DaisyPod hw;
static SdmmcHandler sd;
static FatFSInterface fsi;

// Directory reading - must be in D1 SRAM for SDMMC DMA access
static DIR __attribute__((aligned(4))) s_dir;
static FILINFO __attribute__((aligned(4))) s_fno;

// Log file - must be aligned for DMA access
static FIL __attribute__((aligned(4))) s_logFile;
static bool s_logFileOpen = false;

// Audio buffers
static NAM_SAMPLE inputBuffer[AUDIO_BUFFER_SIZE];
static NAM_SAMPLE outputBuffer[AUDIO_BUFFER_SIZE];
static NAM_SAMPLE* inputPtr = inputBuffer;
static NAM_SAMPLE* outputPtr = outputBuffer;

// Shared format buffer
static char printBuffer[256];

// ---------------------------------------------------------------------------
// Output helpers: print to serial + log file (if open)
// ---------------------------------------------------------------------------

void PrintLine(const char* msg)
{
  hw.seed.PrintLine(msg);
  if (s_logFileOpen)
  {
    UINT bw;
    f_write(&s_logFile, msg, strlen(msg), &bw);
    f_write(&s_logFile, "\n", 1, &bw);
  }
}

void Printf(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsnprintf(printBuffer, sizeof(printBuffer), fmt, args);
  va_end(args);
  PrintLine(printBuffer);
}

// Print profiling breakdown through our dual-output Printf.
// Replaces nam::profiling::print_results() which only goes to printf/serial.
#ifdef NAM_PROFILING
void PrintProfilingResults()
{
  const auto& t = nam::profiling::g_timings;
  uint32_t total = t.total();

  Printf("  Profiling breakdown:");
  Printf("  %-12s %8s %6s", "Category", "Time(ms)", "%");
  Printf("  %-12s %8s %6s", "--------", "--------", "----");

  auto print_row = [total](const char* name, uint32_t us) {
    if (us > 0)
    {
      uint32_t pct = total > 0 ? (us * 100 / total) : 0;
      Printf("  %-12s %8.1f %5lu%%", name, us / 1000.0f, (unsigned long)pct);
    }
  };

  print_row("Conv1D", t.conv1d);
  print_row("InputMixin", t.input_mixin);
  print_row("Layer1x1", t.layer1x1);
  print_row("Head1x1", t.head1x1);
  print_row("Rechannel", t.rechannel);
  print_row("Conv1x1", t.conv1x1);
  print_row("Activation", t.activation);
  print_row("FiLM", t.film);
  print_row("Copies", t.copies);
  print_row("SetZero", t.setzero);
  print_row("RingBuf", t.ringbuf);
  print_row("Condition", t.condition);
  print_row("LSTM", t.lstm);
  print_row("Other", t.other);

  Printf("  %-12s %8s %6s", "--------", "--------", "----");
  Printf("  %-12s %8.1f %5s", "Total", total / 1000.0f, "100%");
}
#else
void PrintProfilingResults() {}
#endif

// ---------------------------------------------------------------------------
// Log file management
// ---------------------------------------------------------------------------

bool OpenLogFile(const char* folder)
{
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "/%s/profiling.log", folder);

  FRESULT result = f_open(&s_logFile, path, FA_WRITE | FA_CREATE_ALWAYS);
  if (result != FR_OK)
  {
    Printf("  WARNING: Could not create %s (err=%d)", path, (int)result);
    s_logFileOpen = false;
    return false;
  }

  s_logFileOpen = true;
  return true;
}

void SyncLogFile()
{
  if (s_logFileOpen)
  {
    f_sync(&s_logFile);
  }
}

void CloseLogFile()
{
  if (s_logFileOpen)
  {
    f_close(&s_logFile);
    s_logFileOpen = false;
  }
}

// ---------------------------------------------------------------------------
// SD card and file scanning
// ---------------------------------------------------------------------------

bool InitSDCard()
{
  System::Delay(1000);

  SdmmcHandler::Config sd_cfg;
  sd_cfg.Defaults();
  sd.Init(sd_cfg);

  System::Delay(500);
  fsi.Init(FatFSInterface::Config::MEDIA_SD);
  System::Delay(100);

  FRESULT result = FR_DISK_ERR;
  const int maxAttempts = 5;

  for (int attempt = 1; attempt <= maxAttempts; attempt++)
  {
    result = f_mount(&fsi.GetSDFileSystem(), "/", 1);
    if (result == FR_OK)
    {
      break;
    }
    System::Delay(100 * attempt);
  }

  return result == FR_OK;
}

bool IsNamFile(const char* filename)
{
  size_t len = strlen(filename);
  if (len < 4)
    return false;

  if ((filename[0] == '.') || (filename[0] == '_'))
    return false;

  const char* ext = filename + len - 4;
  return (ext[0] == '.' && (ext[1] == 'n' || ext[1] == 'N') && (ext[2] == 'a' || ext[2] == 'A')
          && (ext[3] == 'm' || ext[3] == 'M'));
}

// Scan root of SD card for folders
void ScanFolders()
{
  DIR& dir = s_dir;
  FILINFO& fno = s_fno;
  s_numFolders = 0;

  FRESULT result = f_opendir(&dir, "/");
  if (result != FR_OK)
  {
    PrintLine("ERROR: Failed to open root directory");
    return;
  }

  while (s_numFolders < MAX_FOLDERS)
  {
    result = f_readdir(&dir, &fno);
    if (result != FR_OK || fno.fname[0] == 0)
      break;

    if (!(fno.fattrib & AM_DIR))
      continue;

    if (fno.fname[0] == '.' || fno.fname[0] == '_')
      continue;

    strncpy(s_folders[s_numFolders], fno.fname, MAX_FOLDER_NAME_LEN - 1);
    s_folders[s_numFolders][MAX_FOLDER_NAME_LEN - 1] = '\0';
    Printf("  [%d] %s/", (int)s_numFolders, s_folders[s_numFolders]);
    s_numFolders++;
  }

  f_closedir(&dir);
  Printf("Found %d folder(s)", (int)s_numFolders);
}

// Scan a single folder for .nam files, storing full paths in s_namFiles
size_t ScanNamFiles(const char* folder)
{
  DIR& dir = s_dir;
  FILINFO& fno = s_fno;
  s_numNamFiles = 0;

  char folderPath[MAX_PATH_LEN];
  snprintf(folderPath, sizeof(folderPath), "/%s", folder);

  FRESULT result = f_opendir(&dir, folderPath);
  if (result != FR_OK)
  {
    Printf("  ERROR: Failed to open /%s", folder);
    return 0;
  }

  while (s_numNamFiles < MAX_NAM_FILES)
  {
    result = f_readdir(&dir, &fno);
    if (result != FR_OK || fno.fname[0] == 0)
      break;

    if (fno.fattrib & AM_DIR)
      continue;

    if (!IsNamFile(fno.fname))
      continue;

    snprintf(s_namFiles[s_numNamFiles], MAX_PATH_LEN, "/%s/%s", folder, fno.fname);
    Printf("    [%d] %s", (int)s_numNamFiles, fno.fname);
    s_numNamFiles++;
  }

  f_closedir(&dir);
  Printf("  Found %d .nam file(s) in /%s", (int)s_numNamFiles, folder);
  return s_numNamFiles;
}

// ---------------------------------------------------------------------------
// Profiling
// ---------------------------------------------------------------------------

void FillInputNoise()
{
  uint32_t rng = 0xDEADBEEF;
  for (size_t i = 0; i < AUDIO_BUFFER_SIZE; i++)
  {
    // LCG pseudo-random, scaled to small guitar-level signal
    rng = rng * 1664525u + 1013904223u;
    inputBuffer[i] = ((float)(rng >> 16) / 32768.0f - 1.0f) * 0.1f;
  }
}

void ProfileModel(int index)
{
  // Extract just the filename from the full path for display
  const char* fullPath = s_namFiles[index];
  const char* filename = strrchr(fullPath, '/');
  filename = filename ? filename + 1 : fullPath;

  Printf("");
  Printf("--- [%d/%d] %s ---", index + 1, (int)s_numNamFiles, filename);

  // Heap diagnostics before loading
  {
    struct mallinfo mi = mallinfo();
    Printf("  Heap before load: used=%lu free=%lu (arena=%lu)",
           (unsigned long)mi.uordblks, (unsigned long)mi.fordblks, (unsigned long)mi.arena);
  }

  // Print progress marker and flush BEFORE loading, so if the board crashes
  // during loading the log will show which model caused it.
  Printf("  Loading...");
  SyncLogFile();

  // Load model
  std::unique_ptr<nam::DSP> model = nam::get_dsp_fatfs(fullPath);

  // Report phase timings (available even on failure)
  {
    nam::LoadTiming timing = nam::get_dsp_fatfs_timing();
    Printf("  Load phases: SD read %lu ms, JSON parse %lu ms, model create %lu ms",
           (unsigned long)(timing.sd_read_us / 1000),
           (unsigned long)(timing.json_parse_us / 1000),
           (unsigned long)(timing.model_create_us / 1000));
  }

  if (!model)
  {
    Printf("  SKIP: %s", nam::get_dsp_fatfs_error_string(nam::get_dsp_fatfs_last_error()));
    SyncLogFile();
    return;
  }

  // Report SDRAM arena usage from JSON parsing
  {
    size_t peak = nam::get_dsp_fatfs_arena_peak();
    size_t overflows = nam::get_dsp_fatfs_arena_overflows();
    if (peak > 0)
    {
      Printf("  SDRAM arena: %lu KB peak", (unsigned long)(peak / 1024));
      if (overflows > 0)
      {
        Printf("  SDRAM arena: %lu allocation(s) overflowed to heap", (unsigned long)overflows);
      }
    }
    else
    {
      Printf("  SDRAM arena: not used");
    }
  }

  // Copy weights to DTCM (copy_weights_to_dtcm resets the allocator internally)
  size_t weight_count = model->get_total_weight_count();
  if (weight_count > 0)
  {
    if (model->copy_weights_to_dtcm())
    {
      Printf("  DTCM: %lu floats (%lu KB)", (unsigned long)nam::dtcm::get_weights_used(),
             (unsigned long)(nam::dtcm::get_weights_used() * sizeof(float) / 1024));
    }
    else
    {
      size_t used = nam::dtcm::get_weights_used();
      Printf("  DTCM: failed (%lu/%lu used, %lu max) - using main RAM",
             (unsigned long)used,
             (unsigned long)weight_count,
             (unsigned long)nam::dtcm::WEIGHT_BUFFER_SIZE);
    }
  }
  else
  {
    Printf("  DTCM: not supported for this model type");
  }

#ifdef NAM_PROFILE_ZERO_WEIGHTS
  // Zero all weights to isolate numerical issues from structural performance.
  // If speed difference disappears with zeroed weights, the issue is numerical
  // (e.g., NaN/Inf propagation from trained weight values).
  model->zero_weights();
  Printf("  ZERO_WEIGHTS: all weights zeroed (%lu floats)", (unsigned long)weight_count);
#endif

  // Reset and prewarm
  model->ResetAndPrewarm(SAMPLE_RATE, AUDIO_BUFFER_SIZE);

  // Fill input buffer with noise
  FillInputNoise();

  // Reset profiling counters
  nam::profiling::reset();

  // Flush before benchmark so crash during processing is distinguishable from loading stall
  Printf("  Benchmarking...");
  SyncLogFile();

  // Benchmark: process NUM_SECONDS of audio
  // Accumulate cycle counts per-buffer into a 64-bit total to handle DWT wrap.
  // DWT->CYCCNT wraps every ~8.95s at 480 MHz. Each buffer processes in <<8.95s,
  // so per-buffer unsigned subtraction is always correct. The 64-bit accumulator
  // handles arbitrarily long total processing times.
  uint64_t total_cycles = 0;
  uint32_t prev_cyc = nam::profiling::get_cycles();
  for (size_t i = 0; i < NUM_BUFFERS; i++)
  {
    model->process(&inputPtr, &outputPtr, AUDIO_BUFFER_SIZE);
    uint32_t now_cyc = nam::profiling::get_cycles();
    total_cycles += (uint32_t)(now_cyc - prev_cyc);
    prev_cyc = now_cyc;
  }

  uint32_t elapsed_us = (uint32_t)(total_cycles / nam::profiling::cycles_per_us());
  float elapsed_ms = elapsed_us / 1000.0f;
  float audio_ms = NUM_SECONDS * 1000.0f;
  float ratio = elapsed_ms / audio_ms;

  Printf("  Buffers: %lu x %lu samples", (unsigned long)NUM_BUFFERS, (unsigned long)AUDIO_BUFFER_SIZE);
  Printf("  Time: %.1f ms for %d s audio", elapsed_ms, (int)NUM_SECONDS);

  if (ratio < 1.0f)
  {
    Printf("  Result: %.2fx realtime - PASS", ratio);
  }
  else
  {
    Printf("  Result: %.2fx realtime - TOO SLOW", ratio);
  }

  // Check output buffer for numerical issues
  {
    int nans = 0, infs = 0, subnormals = 0;
    for (size_t i = 0; i < AUDIO_BUFFER_SIZE; i++)
    {
      float v = outputBuffer[i];
      uint32_t bits;
      __builtin_memcpy(&bits, &v, sizeof(bits));
      uint32_t exp = (bits >> 23) & 0xFF;
      uint32_t mant = bits & 0x7FFFFF;
      if (exp == 0xFF && mant != 0) nans++;
      else if (exp == 0xFF && mant == 0) infs++;
      else if (exp == 0 && mant != 0) subnormals++;
    }
    if (nans || infs || subnormals)
    {
      Printf("  WARNING: output has %d NaN, %d Inf, %d subnormal", nans, infs, subnormals);
    }
    // Also verify FPSCR is still set
    uint32_t fpc = __get_FPSCR();
    if (!((fpc >> 24) & 1))
    {
      Printf("  WARNING: FPSCR FZ bit cleared! (0x%08lX)", (unsigned long)fpc);
    }
  }

  // Print detailed profiling breakdown to both serial and log
  PrintProfilingResults();

  // Release model before loading the next one
  model.reset();

  // Heap diagnostics after model destruction
  {
    struct mallinfo mi = mallinfo();
    Printf("  Heap after free: used=%lu free=%lu",
           (unsigned long)mi.uordblks, (unsigned long)mi.fordblks);
  }

  // Flush log after each model in case of crash
  SyncLogFile();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void)
{
  hw.Init();
#ifdef NAM_PROFILE_NO_DCACHE
  SCB_DisableDCache();
#endif

  // Enable Flush-to-Zero (FZ) and Default-NaN (DN) on the FPU.
  // Without this, subnormal floats trigger the slow exception path on Cortex-M7,
  // causing 100-1000x slowdowns in models that produce near-zero intermediates.
  // Set both FPSCR (current context) and FPDSCR (default for new contexts/ISRs).
  uint32_t fpscr = __get_FPSCR();
  fpscr |= (1U << 24) | (1U << 25);  // FZ | DN
  __set_FPSCR(fpscr);
  FPU->FPDSCR |= (1U << 24) | (1U << 25);

  // Enable DWT cycle counter for profiling (may already be enabled by libDaisy,
  // but ensure it's available for NAM_PROFILING timing macros).
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  hw.seed.StartLog(true);
  System::Delay(100);

  // Verify FPSCR was actually set
  uint32_t fpscr_check = __get_FPSCR();

  PrintLine("");
  PrintLine("=================================");
  PrintLine("NAM Profiling Tool for Daisy Pod");
  Printf("  FPSCR: 0x%08lX (FZ=%d, DN=%d)",
         (unsigned long)fpscr_check,
         (int)((fpscr_check >> 24) & 1),
         (int)((fpscr_check >> 25) & 1));
#ifdef NAM_PROFILE_NO_DCACHE
  PrintLine("  D-Cache: DISABLED");
#else
  PrintLine("  D-Cache: enabled");
#endif
#ifdef NAM_PROFILING
  PrintLine("  Profiling: ENABLED (per-op breakdown)");
#else
  PrintLine("  Profiling: disabled (no breakdown)");
#endif
#ifdef NAM_PROFILE_ZERO_WEIGHTS
  PrintLine("  ZERO WEIGHTS: ENABLED (all weights zeroed after load)");
#endif
  PrintLine("=================================");

  // Initialize SD card
  PrintLine("Initializing SD card...");
  if (!InitSDCard())
  {
    PrintLine("ERROR: SD card mount failed");
    hw.led1.Set(1.0f, 0.0f, 0.0f);
    hw.UpdateLeds();
    for (;;)
    {
      System::Delay(100);
    }
  }
  PrintLine("SD card ready.");
  PrintLine("");

  // Enable fast tanh approximation (same as real-time pedal)
  nam::activations::Activation::enable_fast_tanh();


  // Scan root for folders
  PrintLine("Scanning for folders...");
  ScanFolders();

  if (s_numFolders == 0)
  {
    PrintLine("No folders found on SD card.");
    hw.led1.Set(1.0f, 0.8f, 0.0f);
    hw.UpdateLeds();
    for (;;)
    {
      System::Delay(100);
    }
  }

  PrintLine("");
  Printf("Profiling config: %d s audio, %lu-sample buffers",
         (int)NUM_SECONDS, (unsigned long)AUDIO_BUFFER_SIZE);

  hw.led1.Set(0.0f, 0.0f, 1.0f);
  hw.UpdateLeds();

  size_t totalModels = 0;

  // Process each folder
  for (size_t f = 0; f < s_numFolders; f++)
  {
    PrintLine("");
    Printf("=== Folder: /%s ===", s_folders[f]);

    size_t count = ScanNamFiles(s_folders[f]);
    if (count == 0)
    {
      Printf("  No .nam files, skipping.");
      continue;
    }

    // Open log file for this folder
    OpenLogFile(s_folders[f]);

    // Write header to log (log file is now open, so Printf goes to both)
    Printf("NAM Profiling: /%s", s_folders[f]);
    Printf("Config: %d s audio, %lu-sample buffers",
           (int)NUM_SECONDS, (unsigned long)AUDIO_BUFFER_SIZE);

    // Profile each model in this folder
    for (size_t i = 0; i < s_numNamFiles; i++)
    {
      ProfileModel((int)i);
    }

    totalModels += s_numNamFiles;
    CloseLogFile();
    Printf("  Log written to /%s/profiling.log", s_folders[f]);
  }

  PrintLine("");
  PrintLine("=================================");
  Printf("Profiling complete. %lu model(s) across %lu folder(s).",
         (unsigned long)totalModels, (unsigned long)s_numFolders);
  PrintLine("=================================");

  hw.led1.Set(0.0f, 1.0f, 0.0f);
  hw.UpdateLeds();

  for (;;)
  {
    System::Delay(1000);
  }
}
