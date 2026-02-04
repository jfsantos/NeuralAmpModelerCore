// NAM Benchmark for Daisy Pod
//
// This example loads all Neural Amp Modeler (.nam) files from the SD card
// and runs a benchmark on each one by processing silence buffers.
//
// Usage:
// 1. Place .nam model files on the SD card root
// 2. Flash this program to the Daisy Pod
// 3. Connect via USB serial (115200 baud) to see benchmark results

#include <cstdio>
#include <cstring>

#include "daisy_pod.h"
#include "fatfs.h"

#include "NAM/dsp.h"
#include "NAM/dtcm.h"
#include "NAM/activations.h"
#include "get_dsp_fatfs.h"

using namespace daisy;

// Configuration
static constexpr size_t AUDIO_BUFFER_SIZE = 48;
static constexpr float SAMPLE_RATE = 48000.0f;
static constexpr size_t BENCHMARK_DURATION_SECONDS = 2;
static constexpr size_t NUM_BENCHMARK_BUFFERS =
  (static_cast<size_t>(SAMPLE_RATE) / AUDIO_BUFFER_SIZE) * BENCHMARK_DURATION_SECONDS;

// Hardware
static DaisyPod hw;
static SdmmcHandler sd;
static FatFSInterface fsi;

// Directory reading structures - must be in D1 SRAM for SDMMC DMA access
static DIR __attribute__((aligned(4))) s_dir;
static FILINFO __attribute__((aligned(4))) s_fno;

// Audio buffers (using float since we define NAM_SAMPLE_FLOAT)
static NAM_SAMPLE inputBuffer[AUDIO_BUFFER_SIZE];
static NAM_SAMPLE outputBuffer[AUDIO_BUFFER_SIZE];
static NAM_SAMPLE* inputPtr = inputBuffer;
static NAM_SAMPLE* outputPtr = outputBuffer;

// USB serial output helper
static char printBuffer[256];

void PrintLine(const char* msg)
{
  hw.seed.PrintLine(msg);
}

void Printf(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsnprintf(printBuffer, sizeof(printBuffer), fmt, args);
  va_end(args);
  hw.seed.PrintLine(printBuffer);
}

bool InitSDCard()
{
  // Extra delay for QSPI boot settling
  System::Delay(1000);

  SdmmcHandler::Config sd_cfg;
  sd_cfg.Defaults();
  sd.Init(sd_cfg);

  System::Delay(500);
  fsi.Init(FatFSInterface::Config::MEDIA_SD);
  System::Delay(100);

  // Try mounting with retries
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

void RunBenchmark(std::unique_ptr<nam::DSP>& model, const char* filename)
{
  Printf("Benchmarking: %s", filename);

  // Reset the model with our sample rate and buffer size
  model->Reset(SAMPLE_RATE, AUDIO_BUFFER_SIZE);

  // Clear input buffer (silence)
  for (size_t i = 0; i < AUDIO_BUFFER_SIZE; i++)
  {
    inputBuffer[i] = 0.0f;
  }

  // Warm up the model
  for (size_t i = 0; i < 10; i++)
  {
    model->process(&inputPtr, &outputPtr, AUDIO_BUFFER_SIZE);
  }

  // Run the benchmark
  uint32_t startTime = System::GetUs();

  for (size_t i = 0; i < NUM_BENCHMARK_BUFFERS; i++)
  {
    model->process(&inputPtr, &outputPtr, AUDIO_BUFFER_SIZE);
  }

  uint32_t endTime = System::GetUs();
  uint32_t benchmarkTimeUs = endTime - startTime;

  // Calculate and print results
  // Note: nano specs doesn't support %f, so we use integer math
  uint32_t totalTimeMs = benchmarkTimeUs / 1000;
  uint32_t totalTimeFrac = (benchmarkTimeUs % 1000) / 10; // 2 decimal places
  size_t totalSamples = NUM_BENCHMARK_BUFFERS * AUDIO_BUFFER_SIZE;
  uint32_t audioTimeMs = (totalSamples * 1000) / (uint32_t)SAMPLE_RATE;

  // Real-time factor as percentage (100 = 1.0x, 200 = 2.0x, etc.)
  uint32_t realTimePercent = (audioTimeMs * 100) / totalTimeMs;

  Printf("  Time: %lu.%02lu ms | Audio: %lu ms | RT: %lu.%02lux | %s", (unsigned long)totalTimeMs,
         (unsigned long)totalTimeFrac, (unsigned long)audioTimeMs, (unsigned long)(realTimePercent / 100),
         (unsigned long)(realTimePercent % 100), realTimePercent >= 100 ? "OK" : "SLOW");
}

// Check if filename ends with .nam (case insensitive)
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

int main(void)
{
  // Initialize hardware
  hw.Init();
  hw.seed.StartLog(true);
  System::Delay(100);

  PrintLine("");
  PrintLine("=================================");
  PrintLine("NAM Benchmark for Daisy Pod");
  PrintLine("=================================");

  // Initialize SD card
  PrintLine("Initializing SD card...");
  if (!InitSDCard())
  {
    PrintLine("ERROR: SD card mount failed");
    hw.led1.Set(1.0f, 0.0f, 0.0f); // Red LED
    hw.UpdateLeds();
    for (;;)
    {
      System::Delay(100);
    }
  }
  PrintLine("SD card ready.");
  PrintLine("");

  // Enable fast tanh approximation for better performance
  nam::activations::Activation::enable_fast_tanh();

  // Open root directory (using static structures for DMA access)
  DIR& dir = s_dir;
  FILINFO& fno = s_fno;
  FRESULT result = f_opendir(&dir, "/");

  if (result != FR_OK)
  {
    PrintLine("ERROR: Failed to open root directory");
    hw.led1.Set(1.0f, 0.0f, 0.0f); // Red LED
    hw.UpdateLeds();
    for (;;)
    {
      System::Delay(100);
    }
  }

  // Count and process .nam files
  int fileCount = 0;
  int successCount = 0;
  int failCount = 0;

  PrintLine("Scanning for .nam files...");
  PrintLine("");

  // Iterate through directory
  while (true)
  {
    result = f_readdir(&dir, &fno);

    if (result != FR_OK || fno.fname[0] == 0)
    {
      break; // Error or end of directory
    }

    // Skip directories
    if (fno.fattrib & AM_DIR)
    {
      continue;
    }

    // Check if it's a .nam file
    if (!IsNamFile(fno.fname))
    {
      continue;
    }

    fileCount++;

    // Load the model
    std::unique_ptr<nam::DSP> model = nam::get_dsp_fatfs(fno.fname);

    if (model == nullptr)
    {
      Printf("FAILED: %s - %s", fno.fname, nam::get_dsp_fatfs_error_string(nam::get_dsp_fatfs_last_error()));
      failCount++;
      continue;
    }

    // Copy weights to DTCM for faster access
    nam::dtcm::reset_weight_allocator();
    if (model->copy_weights_to_dtcm())
    {
      Printf("  DTCM: %lu floats (%lu KB)", (unsigned long)nam::dtcm::get_weights_used(),
             (unsigned long)(nam::dtcm::get_weights_used() * sizeof(float) / 1024));
    }
    else
    {
      Printf("  DTCM: Model too large (%lu floats needed, %lu available)",
             (unsigned long)model->get_total_weight_count(), (unsigned long)nam::dtcm::get_weights_available());
    }

    // Run benchmark
    RunBenchmark(model, fno.fname);
    successCount++;

    // Release model memory before loading next
    model.reset();
  }

  f_closedir(&dir);

  // Print summary
  PrintLine("");
  PrintLine("=================================");
  Printf("Total files: %d", fileCount);
  Printf("Successful:  %d", successCount);
  Printf("Failed:      %d", failCount);
  PrintLine("=================================");

  // Set LED based on results
  if (fileCount == 0)
  {
    PrintLine("No .nam files found on SD card");
    hw.led1.Set(1.0f, 1.0f, 0.0f); // Yellow LED
  }
  else if (failCount == 0)
  {
    hw.led1.Set(0.0f, 1.0f, 0.0f); // Green LED
  }
  else
  {
    hw.led1.Set(1.0f, 0.5f, 0.0f); // Orange LED
  }
  hw.UpdateLeds();

  PrintLine("");
  PrintLine("Benchmark complete.");

  // Main loop - idle
  for (;;)
  {
    System::Delay(100);
  }
}
