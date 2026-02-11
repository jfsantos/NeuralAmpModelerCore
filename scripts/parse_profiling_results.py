#!/usr/bin/env python3
"""
Parse profiling.log from NAMProfile firmware and join with manifest.csv.

Extracts per-model timing, DTCM usage, realtime factor, and per-operation
profiling breakdown from the log file written by the Daisy profiler.

Usage:
    python scripts/parse_profiling_results.py profiling/profiling.log profiling_models/manifest.csv -o results.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# All profiling categories emitted by NAMProfile.cpp (PrintProfilingResults)
PROFILING_CATEGORIES = [
    "Conv1D", "InputMixin", "Layer1x1", "Head1x1", "Rechannel",
    "Conv1x1", "Activation", "FiLM", "Copies", "SetZero",
    "RingBuf", "Condition", "LSTM", "Other",
]


def parse_profiling_log(log_path: Path) -> list[dict]:
    """
    Parse a profiling.log file into a list of model result dicts.

    Expected format from NAMProfile.cpp:

        --- [1/N] model_name.nam ---
          DTCM: 12345 floats (49 KB)
          ...
          Buffers: 2000 x 48 samples
          Time: 1500.0 ms for 2 s audio
          Result: 0.75x realtime - PASS
          ...
          Profiling breakdown:
            ...
            Conv1D           1200.0    80%
            InputMixin        150.0    10%
            ...
    """
    text = log_path.read_text()
    results = []

    # Split on model headers: "--- [index/total] filename ---"
    model_pattern = re.compile(
        r"^---\s+\[\d+/\d+\]\s+(.+?)\s+---\s*$",
        re.MULTILINE,
    )

    splits = list(model_pattern.finditer(text))

    for i, match in enumerate(splits):
        filename = match.group(1).strip()

        # Get the block of text for this model
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        block = text[start:end]

        result = {"filename": filename}

        # Check for SKIP
        if re.search(r"SKIP:", block):
            result["status"] = "skip"
            results.append(result)
            continue

        # Load phase timings (ms)
        # Format: "  Load phases: SD read 123 ms, JSON parse 456 ms, model create 789 ms"
        phase_match = re.search(
            r"Load phases:\s+SD read\s+(\d+)\s+ms,\s+JSON parse\s+(\d+)\s+ms,\s+model create\s+(\d+)\s+ms",
            block,
        )
        if phase_match:
            result["load_sd_read_ms"] = int(phase_match.group(1))
            result["load_json_parse_ms"] = int(phase_match.group(2))
            result["load_model_create_ms"] = int(phase_match.group(3))

        # DTCM weight count
        dtcm_match = re.search(r"DTCM:\s+(\d+)\s+floats", block)
        if dtcm_match:
            result["dtcm_floats"] = int(dtcm_match.group(1))

        # DTCM copy failed
        dtcm_fail = re.search(r"DTCM: copy failed", block)
        if dtcm_fail:
            result["dtcm_failed"] = True

        # Time and audio duration
        time_match = re.search(r"Time:\s+([\d.]+)\s+ms\s+for\s+(\d+)\s+s\s+audio", block)
        if time_match:
            result["time_ms"] = float(time_match.group(1))
            result["audio_seconds"] = int(time_match.group(2))

        # Realtime factor and pass/fail
        result_match = re.search(r"Result:\s+([\d.]+)x\s+realtime\s+-\s+(\S+)", block)
        if result_match:
            result["realtime_factor"] = float(result_match.group(1))
            result["pass_fail"] = result_match.group(2)

        # Per-category profiling breakdown
        # Format: "  Category    1234.5   80%"
        for cat in PROFILING_CATEGORIES:
            cat_pattern = re.compile(
                rf"^\s+{re.escape(cat)}\s+([\d.]+)\s+\d+%",
                re.MULTILINE,
            )
            cat_match = cat_pattern.search(block)
            if cat_match:
                result[f"prof_{cat}"] = float(cat_match.group(1))

        # Total profiling time
        total_match = re.search(r"^\s+Total\s+([\d.]+)\s+100%", block, re.MULTILINE)
        if total_match:
            result["prof_Total"] = float(total_match.group(1))

        result["status"] = "ok"
        results.append(result)

    return results


def load_manifest(manifest_path: Path) -> dict:
    """Load manifest.csv into a dict keyed by filename."""
    manifest = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["filename"]] = row
    return manifest


def join_results(
    profiling_results: list[dict],
    manifest: dict,
) -> list[dict]:
    """
    Join profiling results with manifest data on filename.

    Returns combined rows with architecture features + performance data.
    """
    joined = []

    for result in profiling_results:
        filename = result["filename"]

        if result.get("status") == "skip":
            print(f"  SKIP: {filename}")
            continue

        # Look up in manifest
        manifest_entry = manifest.get(filename)
        if manifest_entry is None:
            print(f"  WARNING: {filename} not found in manifest, including without architecture data")

        row = {}

        # Add manifest columns first
        if manifest_entry:
            row.update(manifest_entry)
        else:
            row["filename"] = filename

        # Add profiling columns
        row["time_ms"] = result.get("time_ms", "")
        row["realtime_factor"] = result.get("realtime_factor", "")
        row["pass_fail"] = result.get("pass_fail", "")
        row["dtcm_floats"] = result.get("dtcm_floats", "")
        row["dtcm_failed"] = result.get("dtcm_failed", False)

        # Add load phase timings
        row["load_sd_read_ms"] = result.get("load_sd_read_ms", "")
        row["load_json_parse_ms"] = result.get("load_json_parse_ms", "")
        row["load_model_create_ms"] = result.get("load_model_create_ms", "")

        # Add per-category profiling times
        for cat in PROFILING_CATEGORIES:
            row[f"prof_{cat}"] = result.get(f"prof_{cat}", "")
        row["prof_Total"] = result.get("prof_Total", "")

        joined.append(row)

    return joined


def write_results_csv(rows: list[dict], output_path: Path) -> None:
    """Write joined results to CSV."""
    if not rows:
        print("No results to write.")
        return

    # Build fieldnames: manifest columns, then profiling columns
    manifest_fields = [
        "filename", "channels", "bottleneck", "kernel_size", "num_dilations",
        "gating_mode", "activation", "secondary_activation",
        "num_films", "layer1x1", "head1x1", "head1x1_out_channels",
        "has_condition_dsp", "weight_count", "storage_weight_count",
        "memory_kb", "over_dtcm", "seed",
    ]
    profiling_fields = [
        "time_ms", "realtime_factor", "pass_fail",
        "dtcm_floats", "dtcm_failed",
        "load_sd_read_ms", "load_json_parse_ms", "load_model_create_ms",
    ]
    prof_cat_fields = [f"prof_{cat}" for cat in PROFILING_CATEGORIES]
    prof_cat_fields.append("prof_Total")

    fieldnames = manifest_fields + profiling_fields + prof_cat_fields

    # Only include fields that exist in at least one row
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = [f for f in fieldnames if f in all_keys]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {output_path} ({len(rows)} models)")


def main():
    parser = argparse.ArgumentParser(
        description="Parse NAMProfile profiling.log and join with manifest.csv"
    )
    parser.add_argument(
        "log_file", type=str, nargs="+",
        help="Path(s) to profiling.log from Daisy SD card",
    )
    parser.add_argument(
        "-m", "--manifest", type=str, nargs="+", required=True,
        help="Path(s) to manifest.csv from generate_profiling_suite.py",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results.csv",
        help="Output CSV path (default: results.csv)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Parse all log files
    results = []
    for lf in args.log_file:
        log_path = Path(lf)
        if not log_path.exists():
            print(f"ERROR: Log file not found: {log_path}")
            sys.exit(1)
        print(f"Parsing {log_path}...")
        entries = parse_profiling_log(log_path)
        print(f"  Found {len(entries)} model entries")
        results.extend(entries)

    # Load and merge all manifests
    manifest = {}
    for mf in args.manifest:
        manifest_path = Path(mf)
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}")
            sys.exit(1)
        print(f"Loading manifest {manifest_path}...")
        m = load_manifest(manifest_path)
        print(f"  Found {len(m)} models in manifest")
        manifest.update(m)

    print("Joining results...")
    joined = join_results(results, manifest)

    write_results_csv(joined, output_path)

    # Print summary
    times = [r["time_ms"] for r in joined if r.get("time_ms")]
    if times:
        times = [float(t) for t in times]
        print(f"\nTiming summary:")
        print(f"  Models profiled: {len(times)}")
        print(f"  Min time: {min(times):.1f} ms")
        print(f"  Max time: {max(times):.1f} ms")
        print(f"  Mean time: {sum(times) / len(times):.1f} ms")

        pass_count = sum(1 for r in joined if r.get("pass_fail") == "PASS")
        fail_count = sum(1 for r in joined if r.get("pass_fail") == "TOO SLOW")
        print(f"  PASS: {pass_count}, TOO SLOW: {fail_count}")


if __name__ == "__main__":
    main()
