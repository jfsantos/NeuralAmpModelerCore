#!/usr/bin/env python3
"""
Generate a suite of WaveNet .nam models for systematic profiling on Daisy.

Uses PA2T's SearchSpace and generate_random_a2_config() to produce models
with varied architecture parameters via Latin Hypercube Sampling. Uses a
wider-than-daisy search space to span the full DTCM weight range (0-25k+),
with stratified sampling by weight bucket for even coverage.

Usage:
    python scripts/generate_profiling_suite.py -n 50 -o profiling_models/
    python scripts/generate_profiling_suite.py -n 60 -o profiling_models/ --include-over-dtcm 10
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Add PA2T to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_pa2t_dir = _project_root / "PA2T_ProjectA2Modeling"
if str(_pa2t_dir) not in sys.path:
    sys.path.insert(0, str(_pa2t_dir))

from p2at_project_a2.generate_configs import (
    SearchSpace,
    generate_random_a2_config,
)
from p2at_project_a2.model_tools import get_num_weights
from p2at_project_a2._data_model import NamFile

# DTCM limit on STM32H750
DTCM_WEIGHT_LIMIT = 24576

# Daisy heap memory budget (KB).  The STM32H750 has 512 KB of internal SRAM.
# Reserve headroom for stack, FatFS buffers, audio buffers, etc.
DEFAULT_MAX_MEMORY_KB = 400


# =============================================================================
# Memory estimation — Python port of tools/memory_usage.cpp
#
# Computes total RAM (weights + buffers) a WaveNet model needs at runtime,
# given the architecture parameters and the audio buffer size (M).
# =============================================================================

def _conv1x1_mem(in_ch: int, out_ch: int, bias: bool, groups: int, M: int) -> tuple[int, int]:
    """Return (weight_floats, buffer_floats) for a Conv1x1."""
    depthwise = (groups == in_ch == out_ch)
    w = in_ch if depthwise else out_ch * in_ch
    if bias:
        w += out_ch
    b = out_ch * M  # _output
    return w, b


def _conv1d_mem(in_ch: int, out_ch: int, kernel_size: int, bias: bool,
                dilation: int, groups: int, M: int) -> tuple[int, int]:
    """Return (weight_floats, buffer_floats) for a Conv1D."""
    depthwise = (groups == in_ch == out_ch)
    w = kernel_size * in_ch if depthwise else kernel_size * out_ch * in_ch
    if bias:
        w += out_ch
    max_lookback = (kernel_size - 1) * dilation if kernel_size > 0 else 0
    ring_storage = 2 * max_lookback + M
    b = in_ch * ring_storage  # ring buffer
    b += out_ch * M           # _output
    return w, b


def _film_mem(condition_dim: int, input_dim: int, active: bool, shift: bool,
              groups: int, M: int) -> tuple[int, int]:
    """Return (weight_floats, buffer_floats) for a FiLM module."""
    if not active:
        return 0, 0
    scale_shift_dim = 2 * input_dim if shift else input_dim
    w, b = _conv1x1_mem(condition_dim, scale_shift_dim, True, groups, M)
    b += input_dim * M  # _output
    return w, b


def _wavenet_layer_mem(
    condition_size: int, channels: int, bottleneck: int, kernel_size: int,
    dilation: int, gated: bool, groups_input: int, groups_input_mixin: int,
    layer1x1_active: bool, layer1x1_groups: int,
    head1x1_active: bool, head1x1_out_channels: int, head1x1_groups: int,
    film_params: dict,  # keys: film_name -> (active, shift, groups)
    M: int,
) -> tuple[int, int]:
    """Return (weight_floats, buffer_floats) for one WaveNet _Layer."""
    conv_out = 2 * bottleneck if gated else bottleneck
    tw, tb = 0, 0

    def add(wb):
        nonlocal tw, tb
        tw += wb[0]; tb += wb[1]

    # _conv
    add(_conv1d_mem(channels, conv_out, kernel_size, True, dilation, groups_input, M))
    # _input_mixin
    add(_conv1x1_mem(condition_size, conv_out, False, groups_input_mixin, M))
    # _layer1x1
    if layer1x1_active:
        add(_conv1x1_mem(bottleneck, channels, True, layer1x1_groups, M))
    # _head1x1
    if head1x1_active:
        add(_conv1x1_mem(bottleneck, head1x1_out_channels, True, head1x1_groups, M))

    # Internal buffers: _z, _output_next_layer, _output_head
    tb += conv_out * M
    tb += channels * M
    head_out = head1x1_out_channels if head1x1_active else bottleneck
    tb += head_out * M

    # FiLM modules
    film_targets = {
        "conv_pre_film": channels,
        "conv_post_film": conv_out,
        "input_mixin_pre_film": condition_size,
        "input_mixin_post_film": conv_out,
        "activation_pre_film": conv_out,
        "activation_post_film": bottleneck,
    }
    for name, dim in film_targets.items():
        if name in film_params:
            a, s, g = film_params[name]
            add(_film_mem(condition_size, dim, a, s, g, M))
    if layer1x1_active and "layer1x1_post_film" in film_params:
        a, s, g = film_params["layer1x1_post_film"]
        add(_film_mem(condition_size, channels, a, s, g, M))
    if head1x1_active and "head1x1_post_film" in film_params:
        a, s, g = film_params["head1x1_post_film"]
        add(_film_mem(condition_size, head1x1_out_channels, a, s, g, M))

    return tw, tb


def _estimate_wavenet_memory(nam: "NamFile", buffer_size: int = 48) -> tuple[int, int]:
    """
    Estimate weight and buffer floats for a WaveNet NamFile on Daisy.

    Port of tools/memory_usage.cpp logic.

    Returns (weight_floats, buffer_floats).  weight_floats is the storage
    count (full matrices including grouped-conv zeros) — this is what the
    DTCM weight buffer must hold at runtime.
    """
    M = buffer_size
    layer_cfg = nam.config.layers[0]

    channels = layer_cfg.channels
    bottleneck = layer_cfg.bottleneck
    kernel_size = layer_cfg.kernel_size
    dilations = layer_cfg.dilations
    condition_size = layer_cfg.condition_size
    head_size = layer_cfg.head_size
    head_bias = layer_cfg.head_bias
    input_size = layer_cfg.input_size
    groups_input = layer_cfg.groups_input
    groups_input_mixin = layer_cfg.groups_input_mixin

    # Gating
    gating_mode = layer_cfg.gating_mode
    if isinstance(gating_mode, list):
        gating_mode = gating_mode[0]
    gated = gating_mode.lower() in ("gated", "blended")

    # layer1x1
    layer1x1_active = layer_cfg.layer1x1.active
    layer1x1_groups = layer_cfg.layer1x1.groups

    # head1x1
    head1x1_active = layer_cfg.head1x1.active
    head1x1_out_channels = layer_cfg.head1x1.out_channels
    head1x1_groups = layer_cfg.head1x1.groups

    # FiLM params
    film_params = {}
    for name in ("conv_pre_film", "conv_post_film", "input_mixin_pre_film",
                 "input_mixin_post_film", "activation_pre_film",
                 "activation_post_film", "layer1x1_post_film", "head1x1_post_film"):
        fc = getattr(layer_cfg, name, None)
        if fc:
            film_params[name] = (fc.active, fc.shift, fc.groups)

    tw, tb = 0, 0

    def add(wb):
        nonlocal tw, tb
        tw += wb[0]; tb += wb[1]

    # WaveNet top-level: _condition_input + _condition_output
    in_channels = 1  # condition_dim = in_channels (always 1 for our models)
    tb += in_channels * M  # _condition_input
    tb += in_channels * M  # _condition_output (no condition_dsp)

    # _rechannel: Conv1x1(input_size -> channels, bias=false)
    add(_conv1x1_mem(input_size, channels, False, 1, M))

    # Per-layer
    for dilation in dilations:
        add(_wavenet_layer_mem(
            condition_size, channels, bottleneck, kernel_size, dilation,
            gated, groups_input, groups_input_mixin,
            layer1x1_active, layer1x1_groups,
            head1x1_active, head1x1_out_channels, head1x1_groups,
            film_params, M,
        ))

    # _head_rechannel
    head_output_size = head1x1_out_channels if head1x1_active else bottleneck
    add(_conv1x1_mem(head_output_size, head_size, head_bias, 1, M))

    # LayerArray buffers: _layer_outputs + _head_inputs
    tb += channels * M
    tb += head_output_size * M

    # _head_scale
    tw += 1

    return tw, tb


def estimate_memory_bytes(nam: "NamFile", buffer_size: int = 48) -> int:
    """
    Estimate total runtime RAM (bytes) for a WaveNet NamFile on Daisy.

    Port of tools/memory_usage.cpp logic.  Returns weight_bytes + buffer_bytes.
    """
    tw, tb = _estimate_wavenet_memory(nam, buffer_size)
    return (tw + tb) * 4  # sizeof(float)


def estimate_storage_weight_count(nam: "NamFile", buffer_size: int = 48) -> int:
    """
    Estimate the number of floats needed in the DTCM weight buffer.

    This is the *storage* count (full matrices including grouped-conv zeros),
    NOT the JSON/logical weight count from get_num_weights().  The DTCM buffer
    must hold the full matrices, so this is what determines DTCM fitness.
    """
    tw, _ = _estimate_wavenet_memory(nam, buffer_size)
    return tw


def profiling_search_space() -> SearchSpace:
    """
    Search space for profiling: wider than daisy() to produce models spanning
    the full 0-25k+ weight range.

    Key differences from SearchSpace.daisy():
      - channels up to 16 (was 8)
      - bottleneck up to 16 (was 8)
      - kernel_size up to 8 (was 5)
      - num_dilations up to 20 (was 9)
      - condition_dsp disabled (keeps condition_size=1, same as daisy pedal)
    """
    return SearchSpace(
        channel_options=[1, 2, 4, 8, 16],
        bottleneck_options=[1, 2, 4, 8, 16],
        kernel_size_options=[2, 3, 4, 5, 6, 7, 8],
        num_dilations_options=list(range(1, 21)),
        gating_mode_options=["gated", "blended", "none"],
        film_options=[(False, False), (True, False), (True, True)],
        layer1x1_options=[True, False],
        head1x1_options=[True, False],
        condition_dsp_options=[False],
        condition_dsp_head_size_options=[1, 2, 4],
    )


def extract_architecture_params(nam: NamFile) -> dict:
    """Extract key architecture parameters from a NamFile for the manifest."""
    layer = nam.config.layers[0]

    channels = layer.channels
    bottleneck = layer.bottleneck
    kernel_size = layer.kernel_size
    num_dilations = len(layer.dilations)

    # gating_mode — can be str or list
    gating_mode = layer.gating_mode
    if isinstance(gating_mode, list):
        gating_mode = gating_mode[0]
    gating_mode = gating_mode.lower()

    # activation
    activation = layer.activation
    if isinstance(activation, list):
        activation = activation[0]
    if hasattr(activation, "type"):
        activation_name = activation.type
    elif isinstance(activation, str):
        activation_name = activation
    else:
        activation_name = str(activation)

    # Count active FiLM modules
    film_names = [
        "conv_pre_film", "conv_post_film",
        "input_mixin_pre_film", "input_mixin_post_film",
        "activation_pre_film", "activation_post_film",
        "layer1x1_post_film", "head1x1_post_film",
    ]
    num_films = 0
    for fn in film_names:
        film_cfg = getattr(layer, fn, None)
        if film_cfg and film_cfg.active:
            num_films += 1

    layer1x1 = layer.layer1x1.active if layer.layer1x1 else False
    head1x1 = layer.head1x1.active if layer.head1x1 else False
    head1x1_out_channels = layer.head1x1.out_channels if (layer.head1x1 and layer.head1x1.active) else 0
    has_condition_dsp = nam.config.condition_dsp is not None

    # secondary_activation (only meaningful when gating_mode != "none")
    secondary_activation = layer.secondary_activation
    if isinstance(secondary_activation, list):
        secondary_activation = secondary_activation[0] if secondary_activation else ""
    if hasattr(secondary_activation, "type"):
        secondary_activation_name = secondary_activation.type
    elif isinstance(secondary_activation, str):
        secondary_activation_name = secondary_activation
    else:
        secondary_activation_name = str(secondary_activation)

    return {
        "channels": channels,
        "bottleneck": bottleneck,
        "kernel_size": kernel_size,
        "num_dilations": num_dilations,
        "gating_mode": gating_mode,
        "activation": activation_name,
        "secondary_activation": secondary_activation_name,
        "num_films": num_films,
        "layer1x1": layer1x1,
        "head1x1": head1x1,
        "head1x1_out_channels": head1x1_out_channels,
        "has_condition_dsp": has_condition_dsp,
    }


def make_filename(params: dict, weight_count: int, index: int) -> str:
    """
    Create a short, descriptive filename encoding key parameters.
    Kept short for FatFS compatibility (< 64 chars).
    """
    gating_short = {
        "none": "ng",
        "gated": "gt",
        "blended": "bl",
    }.get(params["gating_mode"], "unk")

    name = (
        f"c{params['channels']}"
        f"_b{params['bottleneck']}"
        f"_k{params['kernel_size']}"
        f"_d{params['num_dilations']}"
        f"_{gating_short}"
        f"_{index:03d}"
        f".nam"
    )
    return name


def make_weight_buckets(max_weights: int, n_buckets: int) -> list[tuple[int, int]]:
    """
    Create weight-count buckets with log-scale spacing.

    Log-scale gives finer resolution at the small end (where most models land)
    and wider buckets at the large end.
    """
    # Use log-scale boundaries from 1 to max_weights
    log_min = 0  # log(1) = 0
    log_max = math.log(max_weights)
    boundaries = [0]
    for i in range(1, n_buckets):
        boundaries.append(int(math.exp(log_min + (log_max - log_min) * i / n_buckets)))
    boundaries.append(max_weights)

    # Deduplicate and sort
    boundaries = sorted(set(boundaries))

    buckets = []
    for i in range(len(boundaries) - 1):
        buckets.append((boundaries[i], boundaries[i + 1]))
    return buckets


def generate_candidate_pool(
    search_space: SearchSpace,
    max_weight_limit: int,
    n_candidates: int,
    base_seed: int,
    max_memory_kb: int = 0,
    buffer_size: int = 48,
) -> list[dict]:
    """Generate a large pool of candidate models.

    If max_memory_kb > 0, models whose estimated runtime RAM exceeds that
    budget are silently discarded.
    """
    candidates = []
    memory_rejected = 0
    max_memory_bytes = max_memory_kb * 1024

    for i in range(n_candidates):
        seed = base_seed + i
        try:
            nam = generate_random_a2_config(
                max_weights=max_weight_limit,
                seed=seed,
                search_space=search_space,
            )
        except ValueError:
            continue

        weight_count = get_num_weights(nam)
        storage_count = estimate_storage_weight_count(nam, buffer_size)
        mem_bytes = estimate_memory_bytes(nam, buffer_size)

        if max_memory_bytes > 0 and mem_bytes > max_memory_bytes:
            memory_rejected += 1
            continue

        params = extract_architecture_params(nam)

        candidates.append({
            "nam": nam,
            "weight_count": weight_count,
            "storage_weight_count": storage_count,
            "memory_bytes": mem_bytes,
            "params": params,
            "seed": seed,
        })

    if memory_rejected > 0:
        print(f"  Memory filter: rejected {memory_rejected} candidates "
              f"exceeding {max_memory_kb} KB")

    return candidates


def stratified_select(
    candidates: list[dict],
    buckets: list[tuple[int, int]],
    per_bucket: int,
) -> list[dict]:
    """
    Select up to per_bucket models from each weight bucket.

    Within each bucket, picks models spread across the weight range
    (sorted by weight, then evenly spaced).
    """
    selected = []

    for lo, hi in buckets:
        in_bucket = [c for c in candidates if lo <= c["weight_count"] < hi]
        if not in_bucket:
            continue

        # Sort by weight count for even spacing
        in_bucket.sort(key=lambda c: c["weight_count"])

        if len(in_bucket) <= per_bucket:
            selected.extend(in_bucket)
        else:
            # Evenly spaced indices
            step = len(in_bucket) / per_bucket
            indices = [int(i * step) for i in range(per_bucket)]
            selected.extend(in_bucket[idx] for idx in indices)

    return selected


def generate_suite(
    n_target: int,
    output_dir: Path,
    max_weights: int,
    max_memory_kb: int,
    n_over_dtcm: int,
    n_oversample: int,
    n_buckets: int,
    base_seed: int,
) -> None:
    """Generate models with stratified weight coverage."""
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = profiling_search_space()

    # --- Phase 1: Generate models within DTCM limit ---
    n_within = n_target - n_over_dtcm
    print(f"Target: {n_within} models within DTCM ({max_weights} weights)")
    if max_memory_kb > 0:
        print(f"Memory budget: {max_memory_kb} KB")
    if n_over_dtcm > 0:
        print(f"      + {n_over_dtcm} models above DTCM (heap fallback)")
    print(f"Generating {n_oversample} candidates...")
    print()

    within_candidates = generate_candidate_pool(
        search_space, max_weights, n_oversample, base_seed,
        max_memory_kb=max_memory_kb,
    )
    print(f"  Pool: {len(within_candidates)} candidates within {max_weights} weights")

    if not within_candidates:
        print("ERROR: No candidates generated. Check search space and weight limit.")
        sys.exit(1)

    # Stratified selection
    buckets = make_weight_buckets(max_weights, n_buckets)
    per_bucket = max(1, n_within // len(buckets) + 1)

    print(f"  Buckets ({len(buckets)}):")
    for lo, hi in buckets:
        count = sum(1 for c in within_candidates if lo <= c["weight_count"] < hi)
        print(f"    {lo:>6} - {hi:>6}: {count:>4} candidates")

    models = stratified_select(within_candidates, buckets, per_bucket)

    # Trim to target if we overshot
    if len(models) > n_within:
        models = models[:n_within]

    print(f"\n  Selected {len(models)} models (stratified across {len(buckets)} buckets)")

    # --- Phase 2: Generate models above DTCM limit (heap fallback) ---
    if n_over_dtcm > 0:
        # Models between DTCM limit and 2x DTCM limit
        over_limit = max_weights * 2
        print(f"\nGenerating over-DTCM candidates ({max_weights} - {over_limit} weights)...")

        over_candidates = generate_candidate_pool(
            search_space, over_limit, n_oversample, base_seed + n_oversample,
            max_memory_kb=max_memory_kb,
        )
        # Filter to only those above DTCM limit
        over_candidates = [c for c in over_candidates if c["weight_count"] >= max_weights]
        print(f"  Pool: {len(over_candidates)} candidates above DTCM limit")

        if over_candidates:
            # Sort and evenly space
            over_candidates.sort(key=lambda c: c["weight_count"])
            if len(over_candidates) <= n_over_dtcm:
                over_selected = over_candidates
            else:
                step = len(over_candidates) / n_over_dtcm
                indices = [int(i * step) for i in range(n_over_dtcm)]
                over_selected = [over_candidates[idx] for idx in indices]

            models.extend(over_selected)
            print(f"  Selected {len(over_selected)} over-DTCM models")
        else:
            print(f"  WARNING: No candidates above DTCM limit found")

    # Sort final selection by weight count
    models.sort(key=lambda c: c["weight_count"])

    print(f"\nTotal models: {len(models)}")

    # --- Write output ---
    manifest_rows = []

    for idx, entry in enumerate(models):
        nam = entry["nam"]
        params = entry["params"]
        weight_count = entry["weight_count"]

        filename = make_filename(params, weight_count, idx)
        filepath = output_dir / filename

        nam_json = nam.model_dump()
        with open(filepath, "w") as f:
            json.dump(nam_json, f)

        storage_count = entry["storage_weight_count"]
        # DTCM fitness is determined by storage count (full matrices),
        # not JSON weight count (which divides by groups)
        over_dtcm = storage_count >= DTCM_WEIGHT_LIMIT
        mem_bytes = entry["memory_bytes"]
        mem_kb = mem_bytes / 1024

        manifest_rows.append({
            "filename": filename,
            "channels": params["channels"],
            "bottleneck": params["bottleneck"],
            "kernel_size": params["kernel_size"],
            "num_dilations": params["num_dilations"],
            "gating_mode": params["gating_mode"],
            "activation": params["activation"],
            "secondary_activation": params["secondary_activation"],
            "num_films": params["num_films"],
            "layer1x1": params["layer1x1"],
            "head1x1": params["head1x1"],
            "head1x1_out_channels": params["head1x1_out_channels"],
            "has_condition_dsp": params["has_condition_dsp"],
            "weight_count": weight_count,
            "storage_weight_count": storage_count,
            "memory_kb": round(mem_kb, 1),
            "over_dtcm": over_dtcm,
            "seed": entry["seed"],
        })

        dtcm_tag = " [OVER DTCM]" if over_dtcm else ""
        storage_note = f", storage={storage_count}" if storage_count != weight_count else ""
        print(f"  [{idx + 1}/{len(models)}] {filename} ({weight_count} weights{storage_note}, {mem_kb:.0f} KB){dtcm_tag}")

    # Write manifest.csv
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "filename", "channels", "bottleneck", "kernel_size", "num_dilations",
        "gating_mode", "activation", "secondary_activation",
        "num_films", "layer1x1", "head1x1", "head1x1_out_channels",
        "has_condition_dsp", "weight_count", "storage_weight_count", "memory_kb",
        "over_dtcm", "seed",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest written to {manifest_path}")
    print(f"Models written to {output_dir}/")

    # Summary statistics
    weights = [r["weight_count"] for r in manifest_rows]
    mems = [r["memory_kb"] for r in manifest_rows]
    within = [w for w in weights if w < DTCM_WEIGHT_LIMIT]
    over = [w for w in weights if w >= DTCM_WEIGHT_LIMIT]
    print(f"\nWeight count stats:")
    print(f"  Min:  {min(weights):>8}")
    print(f"  Max:  {max(weights):>8}")
    print(f"  Mean: {sum(weights) / len(weights):>8.0f}")
    print(f"  Within DTCM: {len(within)}")
    print(f"  Over DTCM:   {len(over)}")
    print(f"\nMemory stats:")
    print(f"  Min:  {min(mems):>8.1f} KB")
    print(f"  Max:  {max(mems):>8.1f} KB")
    print(f"  Mean: {sum(mems) / len(mems):>8.1f} KB")


def regenerate_manifest(directory: Path, buffer_size: int = 48) -> None:
    """Re-read existing .nam files and write a fresh manifest.csv.

    This is useful when extract_architecture_params() has been updated
    (e.g., new columns added) and you want to regenerate the manifest
    without re-running the full generation + profiling pipeline.
    """
    nam_files = sorted(directory.glob("*.nam"))
    if not nam_files:
        print(f"ERROR: No .nam files found in {directory}")
        sys.exit(1)

    print(f"Found {len(nam_files)} .nam files in {directory}")

    manifest_rows = []
    for idx, nam_path in enumerate(nam_files):
        filename = nam_path.name
        try:
            with open(nam_path) as f:
                data = json.load(f)
            nam = NamFile(**data)
        except Exception as e:
            print(f"  WARNING: Failed to parse {filename}: {e}")
            continue

        params = extract_architecture_params(nam)
        weight_count = get_num_weights(nam)
        storage_count = estimate_storage_weight_count(nam, buffer_size)
        mem_bytes = estimate_memory_bytes(nam, buffer_size)
        mem_kb = mem_bytes / 1024
        over_dtcm = storage_count >= DTCM_WEIGHT_LIMIT

        manifest_rows.append({
            "filename": filename,
            "channels": params["channels"],
            "bottleneck": params["bottleneck"],
            "kernel_size": params["kernel_size"],
            "num_dilations": params["num_dilations"],
            "gating_mode": params["gating_mode"],
            "activation": params["activation"],
            "secondary_activation": params["secondary_activation"],
            "num_films": params["num_films"],
            "layer1x1": params["layer1x1"],
            "head1x1": params["head1x1"],
            "head1x1_out_channels": params["head1x1_out_channels"],
            "has_condition_dsp": params["has_condition_dsp"],
            "weight_count": weight_count,
            "storage_weight_count": storage_count,
            "memory_kb": round(mem_kb, 1),
            "over_dtcm": over_dtcm,
            "seed": "",
        })

        dtcm_tag = " [OVER DTCM]" if over_dtcm else ""
        print(f"  [{idx + 1}/{len(nam_files)}] {filename} ({weight_count} weights, {mem_kb:.0f} KB){dtcm_tag}")

    manifest_path = directory / "manifest.csv"
    fieldnames = [
        "filename", "channels", "bottleneck", "kernel_size", "num_dilations",
        "gating_mode", "activation", "secondary_activation",
        "num_films", "layer1x1", "head1x1", "head1x1_out_channels",
        "has_condition_dsp", "weight_count", "storage_weight_count", "memory_kb",
        "over_dtcm", "seed",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest written to {manifest_path} ({len(manifest_rows)} models)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate WaveNet .nam models for Daisy profiling"
    )
    parser.add_argument(
        "-n", "--num-models", type=int, default=50,
        help="Target number of models (default: 50)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="profiling_models",
        help="Output directory for .nam files and manifest.csv",
    )
    parser.add_argument(
        "--max-weights", type=int, default=DTCM_WEIGHT_LIMIT,
        help=f"DTCM weight limit (default: {DTCM_WEIGHT_LIMIT})",
    )
    parser.add_argument(
        "--max-memory", type=int, default=DEFAULT_MAX_MEMORY_KB, metavar="KB",
        help=f"Max runtime RAM in KB (default: {DEFAULT_MAX_MEMORY_KB}). "
             "Models exceeding this are filtered out. Set to 0 to disable.",
    )
    parser.add_argument(
        "--include-over-dtcm", type=int, default=0, metavar="N",
        help="Also generate N models above DTCM limit (heap fallback test)",
    )
    parser.add_argument(
        "--oversample", type=int, default=500,
        help="Candidate pool size per phase (default: 500)",
    )
    parser.add_argument(
        "--buckets", type=int, default=8,
        help="Number of weight-count buckets for stratification (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--regenerate-manifest", type=str, default=None, metavar="DIR",
        help="Re-read .nam files from DIR and write a fresh manifest.csv "
             "(skips model generation). Use after updating extract_architecture_params().",
    )

    args = parser.parse_args()

    if args.regenerate_manifest:
        regenerate_manifest(Path(args.regenerate_manifest))
    else:
        generate_suite(
            n_target=args.num_models,
            output_dir=Path(args.output_dir),
            max_weights=args.max_weights,
            max_memory_kb=args.max_memory,
            n_over_dtcm=args.include_over_dtcm,
            n_oversample=args.oversample,
            n_buckets=args.buckets,
            base_seed=args.seed,
        )


if __name__ == "__main__":
    main()
