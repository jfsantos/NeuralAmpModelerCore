// NAM Guitar Pedal for Daisy Pod
//
// Real-time Neural Amp Modeler effect pedal.
// Loads .nam model files from the SD card and processes live audio.
//
// Controls:
//   SW_1:    Toggle effect bypass (on/off)
//   POT1:    Input volume
//   POT2:    Output volume
//   Encoder: Select .nam model (CW = next, CCW = previous, wraps around)
//
// LEDs:
//   Green:  Effect active
//   Red:    Bypassed
//   Blue:   Loading model
//   Yellow: No .nam files found

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

// File management
static constexpr size_t MAX_NAM_FILES = 32;
static constexpr size_t MAX_FILENAME_LEN = 64;
static char s_namFiles[MAX_NAM_FILES][MAX_FILENAME_LEN];
static size_t s_numNamFiles = 0;

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

// Model state
static std::unique_ptr<nam::DSP> s_modelOwner;
static volatile nam::DSP* s_activeModel = nullptr;

// Inter-context communication (audio callback <-> main loop)
static volatile int s_requestedModelIndex = 0;
static volatile int s_currentModelIndex = 0;
static volatile bool s_modelLoadRequested = false;
static volatile bool s_modelLoading = false;

// Control state (written in main loop, read by audio callback)
static volatile bool s_effectEnabled = true;
static volatile float s_inputGain = 1.0f;
static volatile float s_outputGain = 1.0f;

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

// Scan SD card root for .nam files and store filenames
void ScanNamFiles()
{
  DIR& dir = s_dir;
  FILINFO& fno = s_fno;
  s_numNamFiles = 0;

  FRESULT result = f_opendir(&dir, "/");
  if (result != FR_OK)
  {
    PrintLine("ERROR: Failed to open root directory");
    return;
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

    strncpy(s_namFiles[s_numNamFiles], fno.fname, MAX_FILENAME_LEN - 1);
    s_namFiles[s_numNamFiles][MAX_FILENAME_LEN - 1] = '\0';
    Printf("  [%d] %s", (int)s_numNamFiles, s_namFiles[s_numNamFiles]);
    s_numNamFiles++;
  }

  f_closedir(&dir);
  Printf("Found %d .nam file(s)", (int)s_numNamFiles);
}

// Load a model by index from the filename array
bool LoadModel(int index)
{
  if (index < 0 || index >= (int)s_numNamFiles)
    return false;

  s_modelLoading = true;
  Printf("Loading: %s [%d/%d]", s_namFiles[index], index + 1, (int)s_numNamFiles);

  std::unique_ptr<nam::DSP> newModel = nam::get_dsp_fatfs(s_namFiles[index]);
  if (!newModel)
  {
    Printf("FAILED: %s", nam::get_dsp_fatfs_error_string(nam::get_dsp_fatfs_last_error()));
    s_modelLoading = false;
    return false;
  }

  // Deactivate current model while we set up the new one
  s_activeModel = nullptr;
  __asm volatile("" ::: "memory");

  // Copy weights to DTCM for faster access
  nam::dtcm::reset_weight_allocator();
  if (newModel->copy_weights_to_dtcm())
  {
    Printf("  DTCM: %lu floats (%lu KB)", (unsigned long)nam::dtcm::get_weights_used(),
           (unsigned long)(nam::dtcm::get_weights_used() * sizeof(float) / 1024));
  }
  else
  {
    Printf("  DTCM: Model too large (%lu floats needed, %lu available)",
           (unsigned long)newModel->get_total_weight_count(),
           (unsigned long)nam::dtcm::get_weights_available());
  }

  // Reset and prewarm
  newModel->ResetAndPrewarm(SAMPLE_RATE, AUDIO_BUFFER_SIZE);

  // Swap ownership and activate
  s_modelOwner = std::move(newModel);
  __asm volatile("" ::: "memory");
  s_activeModel = s_modelOwner.get();

  s_currentModelIndex = index;
  s_modelLoading = false;

  Printf("Loaded: %s", s_namFiles[index]);
  return true;
}

// Update LEDs based on current state
void UpdateLeds()
{
  if (s_numNamFiles == 0)
  {
    // No files found: yellow
    hw.led1.Set(1.0f, 0.8f, 0.0f);
    hw.led2.Set(0.0f, 0.0f, 0.0f);
  }
  else if (s_modelLoading)
  {
    // Loading: blue
    hw.led1.Set(0.0f, 0.0f, 1.0f);
    hw.led2.Set(0.0f, 0.0f, 1.0f);
  }
  else if (s_effectEnabled && s_activeModel != nullptr)
  {
    // Effect active: green, LED2 brightness tracks output gain
    hw.led1.Set(0.0f, 1.0f, 0.0f);
    hw.led2.Set(0.0f, s_outputGain, 0.0f);
  }
  else
  {
    // Bypassed: dim red
    hw.led1.Set(0.3f, 0.0f, 0.0f);
    hw.led2.Set(0.0f, 0.0f, 0.0f);
  }

  hw.UpdateLeds();
}

// Real-time audio callback (interleaving: in/out are L,R,L,R,... size = total samples)
static void AudioCallback(AudioHandle::InterleavingInputBuffer  in,
                           AudioHandle::InterleavingOutputBuffer out,
                           size_t                                size)
{
  nam::DSP* model = const_cast<nam::DSP*>(s_activeModel);
  float inGain = s_inputGain;
  float outGain = s_outputGain;
  size_t numFrames = size / 2;

  if (s_effectEnabled && model != nullptr)
  {
    // Extract left channel with input gain
    for (size_t i = 0; i < numFrames; i++)
    {
      inputBuffer[i] = in[i * 2] * inGain;
    }

    // Process through NAM model (mono)
    model->process(&inputPtr, &outputPtr, numFrames);

    // Write output to both channels with output gain
    for (size_t i = 0; i < numFrames; i++)
    {
      float sample = outputBuffer[i] * outGain;
      out[i * 2]     = sample;
      out[i * 2 + 1] = sample;
    }
  }
  else
  {
    // Bypass: pass left input through to both channels
    for (size_t i = 0; i < size; i += 2)
    {
      out[i]     = in[i];
      out[i + 1] = in[i];
    }
  }
}

int main(void)
{
  hw.Init();
  hw.seed.StartLog(true);
  System::Delay(100);

  PrintLine("");
  PrintLine("=================================");
  PrintLine("NAM Guitar Pedal for Daisy Pod");
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

  // Enable fast tanh approximation for better performance
  nam::activations::Activation::enable_fast_tanh();

  // Scan for .nam files
  PrintLine("Scanning for .nam files...");
  ScanNamFiles();
  PrintLine("");

  // Load first model if available
  if (s_numNamFiles > 0)
  {
    if (!LoadModel(0))
    {
      PrintLine("WARNING: First model failed to load");
    }
  }
  else
  {
    PrintLine("No .nam files found - running in bypass mode");
  }

  // Start audio
  hw.SetAudioBlockSize(AUDIO_BUFFER_SIZE);
  hw.StartAdc();
  hw.StartAudio(AudioCallback);
  PrintLine("Audio started.");

  // Main loop: process controls and handle model switching
  for (;;)
  {
    hw.ProcessAllControls();

    // SW_1: toggle bypass
    if (hw.button1.RisingEdge())
    {
      s_effectEnabled = !s_effectEnabled;
    }

    // Knobs: input and output gain (Value() since ProcessAllControls already called Process())
    s_inputGain = hw.knob1.Value();
    s_outputGain = hw.knob2.Value();

    // Encoder: model selection
    int32_t enc = hw.encoder.Increment();
    if (enc != 0 && s_numNamFiles > 0 && !s_modelLoading)
    {
      int newIndex = s_requestedModelIndex + enc;
      if (newIndex < 0)
        newIndex = (int)s_numNamFiles - 1;
      else if (newIndex >= (int)s_numNamFiles)
        newIndex = 0;

      s_requestedModelIndex = newIndex;
      s_modelLoadRequested = true;
    }

    // Update LEDs
    UpdateLeds();

    // Handle model switching
    if (s_modelLoadRequested)
    {
      s_modelLoadRequested = false;
      int targetIndex = s_requestedModelIndex;

      if (targetIndex != s_currentModelIndex)
      {
        if (!LoadModel(targetIndex))
        {
          // Load failed: revert to current model
          s_requestedModelIndex = s_currentModelIndex;
          Printf("Load failed, keeping current model [%d]", (int)s_currentModelIndex + 1);
        }
      }
    }

    System::Delay(1);
  }
}
