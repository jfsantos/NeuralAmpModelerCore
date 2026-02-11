# WaveNet Architecture Profiling Guide

Systematic profiling of WaveNet architectures on the Daisy board (STM32H750) to understand how architecture parameters affect compute time.

## Prerequisites

- Python 3.10+ with `numpy` installed
- PA2T package: `pip install -e PA2T_ProjectA2Modeling/`
- Optional: `pip install scikit-learn matplotlib` (for regression analysis and plots)
- Daisy toolchain (arm-none-eabi-gcc, dfu-util)
- SD card formatted as FAT32
- USB serial monitor (e.g., `screen`, `minicom`, or the Daisy web serial tool)

## Pipeline Overview

```
[1. Generate models]  →  [2. Copy to SD]  →  [3. Profile on Daisy]  →  [4. Parse + Analyze]
     (host)                 (manual)            (firmware)                 (host)
```

## Step 1: Generate Models

From the project root (`NeuralAmpModelerCore/`):

```bash
python scripts/generate_profiling_suite.py \
    -n 55 \
    -o profiling_models/ \
    --include-over-dtcm 5 \
    --oversample 1000
```

This produces:
- `profiling_models/*.nam` — model files with varied architectures
- `profiling_models/manifest.csv` — architecture parameters for each model

The generator uses a search space wider than `SearchSpace.daisy()` (channels/bottleneck up to 16, kernel_size up to 8, dilations up to 20) so that models span the full 0–25k+ weight range. Models are selected via **stratified sampling** across log-scale weight buckets for even coverage.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-n` | 50 | Target number of models (total, including over-DTCM) |
| `-o` | `profiling_models/` | Output directory |
| `--max-weights` | 24576 | DTCM weight limit |
| `--include-over-dtcm` | 0 | Number of extra models above DTCM limit (to test heap fallback performance cliff) |
| `--oversample` | 500 | Candidate pool size (increase if you're not hitting your target count) |
| `--buckets` | 8 | Number of weight-count buckets for stratified sampling |
| `--seed` | 42 | Base random seed for reproducibility |

### What gets varied

| Parameter | Range | Effect on compute |
|-----------|-------|-------------------|
| `channels` | 1, 2, 4, 8, 16 | Quadratic in GEMM ops |
| `bottleneck` | 1, 2, 4, 8, 16 | Conv1D output size |
| `kernel_size` | 2, 3, 4, 5, 6, 7, 8 | Conv1D weight count |
| `num_dilations` | 1–20 | Number of layers (linear) |
| `gating_mode` | none, gated, blended | Doubles conv output when gated/blended |
| `activation` | 5+ types | Minor compute differences |
| `num_films` | 0–8 | Extra per-layer FiLM overhead |
| `layer1x1` | true/false | Extra 1x1 conv per layer |
| `head1x1` | true/false | Extra 1x1 conv per layer |

### Typical output

With `--oversample 1000 --buckets 10`, expect models ranging from ~30 to ~20,000+ weights within DTCM, plus over-DTCM models up to ~40,000+ weights. Very small architectures (< 50 weights) have few possible configurations, so the lowest buckets may have fewer models than requested.

## Step 2: Copy Models to SD Card

1. Mount the SD card on your computer
2. Create a folder at the root (e.g., `profiling/`)
3. Copy all `.nam` files from `profiling_models/` into that folder
4. Eject the SD card and insert it into the Daisy board

**Important:** The profiler scans folders in the SD card root. Each folder gets its own `profiling.log`. If you have more than 64 models, split them across multiple folders (max 16 folders, 64 files per folder).

## Step 3: Run Profiling on Daisy

### Build and flash the profiler firmware

```bash
cd daisy
make -f Makefile.profile clean
make -f Makefile.profile program-dfu
```

### Monitor progress

Connect a USB serial monitor at 115200 baud to watch progress:

```bash
screen /dev/tty.usbmodem* 115200
```

The LED indicates status:
- **Blue** — profiling in progress
- **Green** — done
- **Red** — SD card error

Each model takes a few seconds. For 50 models, expect ~5–10 minutes total.

### What the profiler does

For each `.nam` file it finds:
1. Loads the model from SD card
2. Copies weights to DTCM (fast RAM) if they fit (< 24,576 floats)
3. Zeroes all weights (isolates structural performance from numerical issues)
4. Processes 2 seconds of pseudo-random audio in 48-sample buffers
5. Reports wall-clock time, realtime factor, and per-operation timing breakdown

Models above the DTCM limit will show "DTCM: copy failed" and run from heap memory (slower due to cache misses). This is intentional for `--include-over-dtcm` models.

Output goes to both USB serial and `profiling.log` inside each folder on the SD card.

## Step 4: Retrieve Results

1. Power off the Daisy and remove the SD card
2. Mount it on your computer
3. Copy `profiling.log` from each folder back to your host machine

For example, if your SD card folder was `profiling/`:
```bash
cp /Volumes/SDCARD/profiling/profiling.log profiling_results/
```

If you used multiple folders, concatenate the logs or run the parser once per folder.

## Step 5: Parse Results

Join the profiling log with the manifest to create a single results CSV:

```bash
python scripts/parse_profiling_results.py \
    profiling_results/profiling.log \
    profiling_models/manifest.csv \
    -o results.csv
```

This produces `results.csv` with columns for both architecture features and timing data.

### Output columns

**Architecture** (from manifest): `filename`, `channels`, `bottleneck`, `kernel_size`, `num_dilations`, `gating_mode`, `activation`, `num_films`, `layer1x1`, `head1x1`, `has_condition_dsp`, `weight_count`, `over_dtcm`

**Timing** (from profiling.log): `time_ms`, `realtime_factor`, `pass_fail`, `dtcm_floats`, `dtcm_failed`

**Per-op breakdown** (ms): `prof_Conv1D`, `prof_InputMixin`, `prof_Layer1x1`, `prof_Head1x1`, `prof_Rechannel`, `prof_Conv1x1`, `prof_Activation`, `prof_FiLM`, `prof_Copies`, `prof_SetZero`, `prof_RingBuf`, `prof_Condition`, `prof_LSTM`, `prof_Other`, `prof_Total`

## Step 6: Analyze

```bash
python scripts/analyze_profiling.py results.csv
```

Add `--plot` to save PNG plots (requires matplotlib):

```bash
python scripts/analyze_profiling.py results.csv --plot
```

### What the analysis outputs

1. **Summary statistics** — min/max/mean/median for timing and memory
2. **Correlation matrix** — which features correlate most with total time
3. **Linear regression** — `time_ms ~ features` with R², RMSE, coefficients (requires scikit-learn)
4. **Standardized feature importance** — which parameters matter most
5. **Prediction formula** — equation to estimate time from architecture params
6. **`results_sorted.csv`** — full results sorted by realtime factor (fastest first)

### Regression features

The regression model uses these derived features as predictors of `time_ms` (total wall-clock time to process 2 seconds of audio):

| Feature | Formula |
|---------|---------|
| `compute_proxy` | `channels * effective_bottleneck * kernel_size * num_dilations` |
| `film_overhead` | `num_films * channels * num_dilations` |
| `layer1x1_compute` | `bottleneck * channels * num_dilations` (0 if inactive) |
| `head1x1_compute` | `bottleneck * channels * num_dilations` (0 if inactive) |
| `input_mixin_compute` | `effective_bottleneck * num_dilations` |
| `num_dilations` | raw count (captures per-layer fixed overhead) |
| `weight_count` | total weights (proxy for memory/cache pressure) |

Where `effective_bottleneck = bottleneck * gated_multiplier` (2 for gated/blended, 1 for none).

### Plots (with `--plot`)

- `scatter_features.png` — time vs compute_proxy, weight_count, channels, num_dilations
- `profiling_breakdown.png` — stacked bar chart of per-op timing for top 15 models
- `realtime_distribution.png` — histogram of realtime factors

## Quick Reference

```bash
# Generate ~50 models spanning the full DTCM range + 5 over-DTCM models
python scripts/generate_profiling_suite.py -n 55 -o profiling_models/ --include-over-dtcm 5 --oversample 1000

# ... copy *.nam to SD card folder, flash & run Daisy profiler, copy back profiling.log ...

# Parse and analyze
python scripts/parse_profiling_results.py profiling.log profiling_models/manifest.csv -o results.csv
python scripts/analyze_profiling.py results.csv --plot
```

## Troubleshooting

**Not enough models generated** — The stratified sampler may under-fill buckets at the extremes (very small or very large models). Increase `--oversample` (e.g., 1000 or 2000) to give the sampler more candidates to choose from.

**Models missing from results.csv** — The parser joins on exact filename. Verify the `.nam` filenames on the SD card match `manifest.csv`. FatFS may truncate long filenames.

**DTCM copy failed** — Model weights exceed 24,576 floats (96 KB). The model still runs but from heap memory (slower). This is expected for `--include-over-dtcm` models and useful for measuring the DTCM vs heap performance cliff.

**Over-DTCM models crash the board** — Very large models can exhaust heap memory. If this happens, reduce the upper bound by lowering `--max-weights` or avoid `--include-over-dtcm`. The default 2x DTCM limit (49,152 weights) is generally safe.

**R² is low** — With only ~50 data points, try reducing the number of regression features or check for outlier models that crashed or had numerical issues (NaN/Inf warnings in the log).
