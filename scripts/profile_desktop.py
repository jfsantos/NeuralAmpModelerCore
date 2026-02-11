#!/usr/bin/env python3
"""
Run desktop profiling of all .nam models in a directory using benchmodel_bufsize.

Produces a profiling.log in the same format as the Daisy NAMProfile firmware,
so results can be parsed by parse_profiling_results.py and analyzed with
analyze_profiling.py.

Usage:
    python scripts/profile_desktop.py profiling_models/ -o desktop_results/profiling.log
    python scripts/profile_desktop.py profiling_models/ -o desktop_results/profiling.log --buffer-size 48

Then parse and analyze:
    python scripts/parse_profiling_results.py desktop_results/profiling.log profiling_models/manifest.csv -o desktop_results.csv
    python scripts/analyze_profiling.py desktop_results.csv
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_BUFFER_SIZE = 48
DEFAULT_NUM_ITERATIONS = 5
DEFAULT_BENCHMODEL = "build_test/tools/benchmodel_bufsize"


def find_nam_files(directory: Path) -> list[Path]:
    """Find all .nam files in a directory, sorted by name."""
    files = sorted(directory.glob("*.nam"))
    return files


def run_benchmark(
    benchmodel: str,
    model_path: Path,
    buffer_size: int,
    num_iterations: int,
) -> dict | None:
    """Run benchmodel_bufsize on a single model and parse the output.

    Returns a dict with timing and profiling data, or None on failure.
    """
    try:
        result = subprocess.run(
            [benchmodel, str(model_path), str(buffer_size), str(num_iterations)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per model
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except FileNotFoundError:
        return {"error": f"benchmodel not found: {benchmodel}"}

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return {"error": stderr or f"exit code {result.returncode}"}

    output = result.stdout

    parsed = {}

    # First line: buffer_size,avg_microseconds
    lines = output.strip().split("\n")
    if not lines:
        return {"error": "empty output"}

    # Use split instead of regex — float() handles scientific notation (e.g. 1.23e+06)
    parts = lines[0].strip().split(",")
    if len(parts) == 2:
        try:
            parsed["time_ms"] = float(parts[1]) / 1000.0
        except ValueError:
            pass

    # Parse profiling breakdown (same format as Daisy)
    for line in lines[1:]:
        # Match: "Category    1234.5    30%"
        cat_match = re.match(r"\s*(\w+)\s+([\d.]+)\s+(\d+)%", line)
        if cat_match:
            category = cat_match.group(1)
            time_ms = float(cat_match.group(2))
            if category != "Total":
                parsed[f"prof_{category}"] = time_ms
            else:
                parsed["prof_Total"] = time_ms

    return parsed


def format_daisy_log(
    filename: str,
    index: int,
    total: int,
    result: dict | None,
    buffer_size: int,
    num_seconds: int = 2,
) -> str:
    """Format benchmark result in the same format as NAMProfile.cpp output."""
    lines = []
    lines.append(f"--- [{index + 1}/{total}] {filename} ---")

    if result is None or "error" in result:
        error = result.get("error", "unknown error") if result else "unknown error"
        lines.append(f"  SKIP: {error}")
        return "\n".join(lines)

    time_ms = result.get("time_ms", 0)
    audio_ms = num_seconds * 1000
    num_buffers = int(48000 / buffer_size) * num_seconds
    ratio = time_ms / audio_ms if audio_ms > 0 else 0

    lines.append(f"  Buffers: {num_buffers} x {buffer_size} samples")
    lines.append(f"  Time: {time_ms:.1f} ms for {num_seconds} s audio")

    if ratio < 1.0:
        lines.append(f"  Result: {ratio:.2f}x realtime - PASS")
    else:
        lines.append(f"  Result: {ratio:.2f}x realtime - TOO SLOW")

    # Profiling breakdown
    categories = [
        "Conv1D", "InputMixin", "Layer1x1", "Head1x1", "Rechannel",
        "Conv1x1", "Activation", "FiLM", "Copies", "SetZero",
        "RingBuf", "Condition", "LSTM", "Other",
    ]
    prof_total = result.get("prof_Total", 0)

    has_profiling = any(f"prof_{cat}" in result for cat in categories)
    if has_profiling:
        lines.append("  Profiling breakdown:")
        lines.append(f"  {'Category':<12s} {'Time(ms)':>8s} {'%':>6s}")
        lines.append(f"  {'--------':<12s} {'--------':>8s} {'----':>6s}")

        for cat in categories:
            key = f"prof_{cat}"
            if key in result and result[key] > 0:
                val = result[key]
                pct = int(val * 100 / prof_total) if prof_total > 0 else 0
                lines.append(f"  {cat:<12s} {val:>8.1f} {pct:>5d}%")

        lines.append(f"  {'--------':<12s} {'--------':>8s} {'----':>6s}")
        lines.append(f"  {'Total':<12s} {prof_total:>8.1f} {'100%':>5s}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run desktop profiling of .nam models"
    )
    parser.add_argument(
        "model_dir", type=str,
        help="Directory containing .nam files",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="profiling.log",
        help="Output log file (default: profiling.log)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE,
        help=f"Audio buffer size in samples (default: {DEFAULT_BUFFER_SIZE})",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_NUM_ITERATIONS,
        help=f"Number of timing iterations per model (default: {DEFAULT_NUM_ITERATIONS})",
    )
    parser.add_argument(
        "--benchmodel", type=str, default=DEFAULT_BENCHMODEL,
        help=f"Path to benchmodel_bufsize binary (default: {DEFAULT_BENCHMODEL})",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    if not model_dir.is_dir():
        print(f"ERROR: Not a directory: {model_dir}")
        sys.exit(1)

    nam_files = find_nam_files(model_dir)
    if not nam_files:
        print(f"No .nam files found in {model_dir}")
        sys.exit(1)

    print(f"Desktop profiling: {len(nam_files)} models in {model_dir}")
    print(f"Config: buffer_size={args.buffer_size}, iterations={args.iterations}")
    print(f"Binary: {args.benchmodel}")
    print()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines = []
    log_lines.append(f"NAM Desktop Profiling: {model_dir}")
    log_lines.append(f"Config: 2 s audio, {args.buffer_size}-sample buffers, "
                     f"{args.iterations} iterations")

    pass_count = 0
    fail_count = 0
    skip_count = 0
    times = []

    for i, model_path in enumerate(nam_files):
        filename = model_path.name
        print(f"  [{i + 1}/{len(nam_files)}] {filename}...", end=" ", flush=True)

        result = run_benchmark(
            args.benchmodel, model_path, args.buffer_size, args.iterations,
        )

        if result and "error" not in result:
            time_ms = result.get("time_ms", 0)
            ratio = time_ms / 2000.0
            status = "PASS" if ratio < 1.0 else "TOO SLOW"
            print(f"{time_ms:.1f} ms ({ratio:.2f}x) - {status}")
            times.append(time_ms)
            if ratio < 1.0:
                pass_count += 1
            else:
                fail_count += 1
        else:
            error = result.get("error", "unknown") if result else "unknown"
            print(f"SKIP: {error}")
            skip_count += 1

        entry = format_daisy_log(
            filename, i, len(nam_files), result, args.buffer_size,
        )
        log_lines.append("")
        log_lines.append(entry)

    # Write log file
    with open(output_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nLog written to {output_path}")
    print(f"\nSummary:")
    print(f"  Models: {len(nam_files)}")
    print(f"  PASS: {pass_count}, TOO SLOW: {fail_count}, SKIP: {skip_count}")
    if times:
        print(f"  Min time:  {min(times):.1f} ms")
        print(f"  Max time:  {max(times):.1f} ms")
        print(f"  Mean time: {sum(times) / len(times):.1f} ms")


if __name__ == "__main__":
    main()
