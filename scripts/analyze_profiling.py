#!/usr/bin/env python3
"""
Analyze profiling results to build predictive models of compute time.

Reads results.csv (output of parse_profiling_results.py), computes derived
features, runs linear regression, and outputs feature importance and a
prediction formula.

Usage:
    python scripts/analyze_profiling.py results.csv
    python scripts/analyze_profiling.py results.csv --plot  # save plots as PNG
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from scipy.linalg import LinAlgWarning
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import cross_val_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(path: Path) -> list[dict]:
    """Load results.csv and convert numeric fields."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in [
                "channels", "bottleneck", "kernel_size", "num_dilations",
                "num_films", "weight_count", "time_ms", "realtime_factor",
                "dtcm_floats", "head1x1_out_channels",
            ]:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass

            # Convert bool fields
            for key in ["layer1x1", "head1x1", "has_condition_dsp", "dtcm_failed"]:
                if key in row:
                    row[key] = row[key] in ("True", "true", "1", True)

            # Convert profiling category times
            for key in row:
                if key.startswith("prof_") and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass

            rows.append(row)
    return rows


def compute_derived_features(rows: list[dict]) -> list[dict]:
    """Add derived features to each row."""

    from collections import Counter

    # Discover all primary activation types present in the data
    activation_types = sorted(set(
        r.get("activation", "") for r in rows
        if r.get("activation") and isinstance(r.get("activation"), str)
    ))

    # Pick the most common activation as reference category (dropped from one-hot)
    if activation_types:
        act_counts = Counter(r.get("activation", "") for r in rows if r.get("activation"))
        reference_activation = act_counts.most_common(1)[0][0]
        onehot_activations = [a for a in activation_types if a != reference_activation]
    else:
        reference_activation = None
        onehot_activations = []

    # Discover secondary activation types (only relevant for gated/blended modes)
    secondary_types = sorted(set(
        r.get("secondary_activation", "") for r in rows
        if r.get("secondary_activation") and isinstance(r.get("secondary_activation"), str)
    ))

    if secondary_types:
        sec_counts = Counter(
            r.get("secondary_activation", "") for r in rows
            if r.get("secondary_activation")
        )
        reference_secondary = sec_counts.most_common(1)[0][0]
        onehot_secondary = [a for a in secondary_types if a != reference_secondary]
    else:
        reference_secondary = None
        onehot_secondary = []

    for row in rows:
        channels = row.get("channels", 0)
        bottleneck = row.get("bottleneck", 0)
        kernel_size = row.get("kernel_size", 0)
        num_dilations = row.get("num_dilations", 0)
        num_films = row.get("num_films", 0)
        gating_mode = row.get("gating_mode", "none")

        if not all(isinstance(v, (int, float)) for v in [channels, bottleneck, kernel_size, num_dilations]):
            continue

        # Gating multiplier: doubles conv output channels for gated/blended
        gated_multiplier = 2 if gating_mode in ("gated", "blended") else 1
        row["gated_multiplier"] = gated_multiplier

        # Effective bottleneck (what the conv actually outputs)
        effective_bottleneck = bottleneck * gated_multiplier
        row["effective_bottleneck"] = effective_bottleneck

        # Compute proxy: rough FLOP estimate for Conv1D per layer
        # Conv1D: channels * effective_bottleneck * kernel_size (per dilation)
        # Total across all dilations
        row["compute_proxy"] = channels * effective_bottleneck * kernel_size * num_dilations

        # FiLM overhead estimate: each FiLM is a 1x1 conv on condition_size -> feature_dim
        row["film_overhead"] = num_films * channels * num_dilations

        # layer1x1 contribution: bottleneck * channels per dilation
        row["layer1x1_compute"] = (bottleneck * channels * num_dilations) if row.get("layer1x1") else 0

        # head1x1 contribution: Conv1x1(bottleneck → head1x1_out_channels) per dilation
        head1x1_out = row.get("head1x1_out_channels", 0)
        row["head1x1_compute"] = (bottleneck * head1x1_out * num_dilations) if row.get("head1x1") else 0

        # Input mixin: condition_size * effective_bottleneck per dilation
        # condition_size is always 1 without condition_dsp
        row["input_mixin_compute"] = effective_bottleneck * num_dilations

        # channels^2 term (dominates rechannel and various 1x1 convs)
        row["channels_sq"] = channels * channels

        # One-hot encode primary activation type (reference category is dropped)
        act = row.get("activation", "")
        for a in onehot_activations:
            row[f"act_{a}"] = 1.0 if act == a else 0.0

        # One-hot encode secondary activation (only meaningful for gated/blended)
        sec_act = row.get("secondary_activation", "")
        for a in onehot_secondary:
            row[f"sec_{a}"] = 1.0 if sec_act == a else 0.0

    if reference_activation:
        print(f"  Primary activation one-hot: {len(onehot_activations)} columns "
              f"(reference: {reference_activation})")
    if reference_secondary:
        print(f"  Secondary activation one-hot: {len(onehot_secondary)} columns "
              f"(reference: {reference_secondary})")

    return rows


def print_summary(rows: list[dict]) -> None:
    """Print summary statistics."""
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Filter to rows with valid timing data
    valid = [r for r in rows if isinstance(r.get("time_ms"), (int, float))]
    if not valid:
        print("No valid timing data found.")
        return

    times = [r["time_ms"] for r in valid]
    weights = [r["weight_count"] for r in valid if isinstance(r.get("weight_count"), (int, float))]
    rt_factors = [r["realtime_factor"] for r in valid if isinstance(r.get("realtime_factor"), (int, float))]

    print(f"\nModels with timing data: {len(valid)}")

    print(f"\nProcessing time (ms for 2s audio):")
    print(f"  Min:    {min(times):8.1f}")
    print(f"  Max:    {max(times):8.1f}")
    print(f"  Mean:   {np.mean(times):8.1f}")
    print(f"  Median: {np.median(times):8.1f}")
    print(f"  Std:    {np.std(times):8.1f}")

    if rt_factors:
        print(f"\nRealtime factor (< 1.0 = realtime capable):")
        print(f"  Min:    {min(rt_factors):8.3f}")
        print(f"  Max:    {max(rt_factors):8.3f}")
        print(f"  Mean:   {np.mean(rt_factors):8.3f}")
        print(f"  Median: {np.median(rt_factors):8.3f}")
        pass_count = sum(1 for r in rt_factors if r < 1.0)
        print(f"  Realtime capable: {pass_count}/{len(rt_factors)}")

    if weights:
        print(f"\nWeight count:")
        print(f"  Min:    {min(weights):8.0f}")
        print(f"  Max:    {max(weights):8.0f}")
        print(f"  Mean:   {np.mean(weights):8.0f}")

    # Distribution of key architecture params
    print(f"\nArchitecture distribution:")
    for param in ["channels", "bottleneck", "kernel_size", "num_dilations", "gating_mode", "activation", "secondary_activation"]:
        values = [r.get(param) for r in valid if r.get(param) is not None]
        if values:
            from collections import Counter
            counts = Counter(values)
            dist = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: str(x[0])))
            print(f"  {param}: {dist}")


def print_correlation(rows: list[dict]) -> None:
    """Print correlation of features with total processing time."""
    print("\n" + "=" * 70)
    print("CORRELATION WITH TOTAL TIME")
    print("=" * 70)

    valid = [r for r in rows if isinstance(r.get("time_ms"), (int, float))]
    if len(valid) < 3:
        print("Not enough data for correlation analysis.")
        return

    times = np.array([r["time_ms"] for r in valid])

    # Discover activation one-hot columns from data
    act_columns = sorted(set(
        k for r in valid for k in r if k.startswith("act_")
    ))
    sec_columns = sorted(set(
        k for r in valid for k in r if k.startswith("sec_")
    ))

    features = [
        "channels", "bottleneck", "kernel_size", "num_dilations",
        "gated_multiplier", "head1x1_out_channels",
        "weight_count", "num_films",
        "compute_proxy", "film_overhead",
        "effective_bottleneck", "layer1x1_compute", "head1x1_compute",
        "input_mixin_compute", "channels_sq",
    ] + act_columns + sec_columns

    print(f"\n{'Feature':<25s} {'Correlation':>12s}")
    print("-" * 40)

    correlations = []
    for feat in features:
        values = [r.get(feat, 0) for r in valid]
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        vals = np.array(values, dtype=float)
        if np.std(vals) == 0:
            continue
        corr = np.corrcoef(vals, times)[0, 1]
        correlations.append((feat, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, corr in correlations:
        print(f"  {feat:<23s} {corr:+.4f}")


FEATURE_NAMES = [
    "channels",
    "bottleneck",
    "kernel_size",
    "num_dilations",
    "gated_multiplier",
    "head1x1_out_channels",
    "num_films",
]


def _build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (X, y, feature_names) from rows with valid timing data.

    Automatically discovers act_* one-hot columns from the data and appends
    them to the base FEATURE_NAMES.
    """
    valid = [r for r in rows if isinstance(r.get("time_ms"), (int, float))]

    # Discover activation one-hot columns from data
    act_columns = sorted(set(
        k for r in valid for k in r if k.startswith("act_")
    ))
    sec_columns = sorted(set(
        k for r in valid for k in r if k.startswith("sec_")
    ))
    all_features = FEATURE_NAMES + act_columns + sec_columns

    X_rows = []
    y_list = []
    for r in valid:
        feat_vals = []
        skip = False
        for fn in all_features:
            v = r.get(fn, 0)
            if not isinstance(v, (int, float)):
                skip = True
                break
            feat_vals.append(float(v))
        if skip:
            continue
        X_rows.append(feat_vals)
        y_list.append(r["time_ms"])

    return np.array(X_rows), np.array(y_list), all_features


def run_regression(rows: list[dict]) -> None:
    """Run linear regression to predict processing time from features."""
    if not HAS_SKLEARN:
        print("\n[scikit-learn not installed — skipping regression]")
        print("  Install with: pip install scikit-learn")
        return

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION MODEL")
    print("=" * 70)

    X, y, feature_names = _build_feature_matrix(rows)

    if len(X) < 5:
        print("Not enough data for regression (need >= 5 models).")
        return

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = model.score(X, y)
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    # Cross-validated R² (leave-one-out if < 20 samples, 5-fold otherwise)
    cv = min(5, len(X))
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")

    print(f"\nModel: time_ms ~ intercept + sum(coef_i * feature_i)")
    print(f"  R^2 (train): {r2:.4f}")
    print(f"  R^2 (CV {cv}-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  RMSE: {rmse:.1f} ms")
    print(f"  MAE:  {mae:.1f} ms")
    print(f"  N:    {len(X)}, features: {len(feature_names)}")

    print(f"\nCoefficients:")
    print(f"  {'Feature':<25s} {'Coef':>12s} {'Interpretation':>30s}")
    print("  " + "-" * 70)
    print(f"  {'(intercept)':<25s} {model.intercept_:>12.4f} {'base overhead ms':>30s}")

    for name, coef in sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<25s} {coef:>12.6f}")

    # Feature importance via standardized coefficients
    print(f"\nStandardized feature importance (absolute):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_std = LinearRegression()
    model_std.fit(X_scaled, y)

    importances = list(zip(feature_names, np.abs(model_std.coef_)))
    importances.sort(key=lambda x: x[1], reverse=True)

    total_imp = sum(v for _, v in importances)
    print(f"  {'Feature':<25s} {'Importance':>12s} {'% of total':>12s}")
    print("  " + "-" * 52)
    for name, imp in importances:
        pct = 100 * imp / total_imp if total_imp > 0 else 0
        print(f"  {name:<25s} {imp:>12.2f} {pct:>11.1f}%")

    # Prediction formula in terms of raw features
    print(f"\nPrediction formula:")
    terms = [f"{model.intercept_:.2f}"]
    for name, coef in zip(feature_names, model.coef_):
        if abs(coef) > 1e-8:
            terms.append(f"{coef:+.6f} * {name}")
    formula = "\n    ".join(terms)
    print(f"  time_ms = {formula}")


def run_interaction_regression(rows: list[dict]) -> None:
    """Run regression with interaction and squared terms using Ridge (L2)."""
    if not HAS_SKLEARN:
        return

    print("\n" + "=" * 70)
    print("INTERACTION REGRESSION MODEL (Ridge, pairwise interactions)")
    print("=" * 70)

    X_raw, y, base_names = _build_feature_matrix(rows)

    if len(X_raw) < 10:
        print("Not enough data for interaction model (need >= 10 models).")
        return

    # Generate interaction terms (no squared terms — reduces collinearity)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    poly_names = poly.get_feature_names_out(base_names)

    # Ridge with built-in cross-validation over a range of alphas
    alphas = np.logspace(-2, 6, 50)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ill-conditioned.*", category=LinAlgWarning)
        model = RidgeCV(alphas=alphas, cv=min(5, len(X_poly)))
        model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    r2 = model.score(X_poly, y)
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    # Cross-validated R²
    cv = min(5, len(X_poly))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ill-conditioned.*", category=LinAlgWarning)
        cv_scores = cross_val_score(
            RidgeCV(alphas=alphas, cv=cv), X_poly, y, cv=cv, scoring="r2",
        )

    print(f"\nModel: time_ms ~ Ridge(degree=2 interaction features)")
    print(f"  R^2 (train): {r2:.4f}")
    print(f"  R^2 (CV {cv}-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  RMSE: {rmse:.1f} ms")
    print(f"  MAE:  {mae:.1f} ms")
    print(f"  N:    {len(X_poly)}, features: {X_poly.shape[1]} (from {len(base_names)} base)")
    print(f"  Best alpha: {model.alpha_:.2f}")

    # Feature importance via standardized coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ill-conditioned.*", category=LinAlgWarning)
        model_std = RidgeCV(alphas=alphas, cv=cv)
        model_std.fit(X_scaled, y)

    importances = list(zip(poly_names, np.abs(model_std.coef_)))
    importances.sort(key=lambda x: x[1], reverse=True)

    total_imp = sum(v for _, v in importances)
    print(f"\nTop 15 features by standardized importance:")
    print(f"  {'Feature':<40s} {'Importance':>12s} {'% of total':>12s}")
    print("  " + "-" * 66)
    for name, imp in importances[:15]:
        pct = 100 * imp / total_imp if total_imp > 0 else 0
        print(f"  {name:<40s} {imp:>12.2f} {pct:>11.1f}%")

    # Print the top non-zero coefficients (raw, not standardized)
    coef_pairs = list(zip(poly_names, model.coef_))
    coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 15 coefficients (raw):")
    print(f"  {'Feature':<40s} {'Coef':>14s}")
    print("  " + "-" * 56)
    print(f"  {'(intercept)':<40s} {model.intercept_:>14.4f}")
    for name, coef in coef_pairs[:15]:
        if abs(coef) > 1e-10:
            print(f"  {name:<40s} {coef:>14.6f}")


def save_sorted_csv(rows: list[dict], output_dir: Path) -> None:
    """Save results sorted by realtime factor."""
    valid = [r for r in rows if isinstance(r.get("realtime_factor"), (int, float))]
    valid.sort(key=lambda r: r["realtime_factor"])

    sorted_path = output_dir / "results_sorted.csv"

    if not valid:
        return

    # Use all keys from first row
    fieldnames = list(valid[0].keys())

    with open(sorted_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)

    print(f"\nSorted results written to {sorted_path}")


def make_plots(rows: list[dict], output_dir: Path) -> None:
    """Generate analysis plots saved as PNG files."""
    if not HAS_MATPLOTLIB:
        print("\n[matplotlib not installed — skipping plots]")
        print("  Install with: pip install matplotlib")
        return

    valid = [r for r in rows if isinstance(r.get("time_ms"), (int, float))]
    if not valid:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.array([r["time_ms"] for r in valid])

    # Plot 1: Scatter plot per model feature
    features = FEATURE_NAMES
    ncols = 3
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        values = [r.get(feat, 0) for r in valid]
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        vals = np.array(values, dtype=float)
        ax.scatter(vals, times, alpha=0.6, s=30)
        ax.set_xlabel(feat)
        ax.set_ylabel("Time (ms)")

        if np.std(vals) > 0:
            z = np.polyfit(vals, times, 1)
            p = np.poly1d(z)
            x_line = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5)
            corr = np.corrcoef(vals, times)[0, 1]
            ax.set_title(f"{feat} (r={corr:.3f})")
        else:
            ax.set_title(feat)

    # Hide unused subplots
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Processing Time vs Model Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_features.png", dpi=150)
    print(f"  Saved {output_dir / 'scatter_features.png'}")
    plt.close(fig)

    # Plot 1b: Box plot of time by activation type
    activations = [r.get("activation", "") for r in valid]
    unique_acts = sorted(set(a for a in activations if a))
    if len(unique_acts) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        act_groups = {a: [] for a in unique_acts}
        for r in valid:
            a = r.get("activation", "")
            if a in act_groups:
                act_groups[a].append(r["time_ms"])

        # Sort by median time
        sorted_acts = sorted(unique_acts, key=lambda a: np.median(act_groups[a]))
        box_data = [act_groups[a] for a in sorted_acts]
        bp = ax.boxplot(box_data, labels=sorted_acts, vert=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xlabel("Activation Type")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Processing Time by Activation Type")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "activation_boxplot.png", dpi=150)
        print(f"  Saved {output_dir / 'activation_boxplot.png'}")
        plt.close(fig)

    # Plot 1c: Box plot of time by secondary activation type (gated/blended only)
    gated_valid = [r for r in valid if r.get("gating_mode") in ("gated", "blended")]
    if gated_valid:
        sec_activations = [r.get("secondary_activation", "") for r in gated_valid]
        unique_sec_acts = sorted(set(a for a in sec_activations if a))
        if len(unique_sec_acts) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sec_groups = {a: [] for a in unique_sec_acts}
            for r in gated_valid:
                a = r.get("secondary_activation", "")
                if a in sec_groups:
                    sec_groups[a].append(r["time_ms"])

            sorted_sec = sorted(unique_sec_acts, key=lambda a: np.median(sec_groups[a]))
            box_data = [sec_groups[a] for a in sorted_sec]
            bp = ax.boxplot(box_data, labels=sorted_sec, vert=True, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightyellow")
            ax.set_xlabel("Secondary (Gating) Activation Type")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Processing Time by Secondary Activation (Gated/Blended Only)")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output_dir / "secondary_activation_boxplot.png", dpi=150)
            print(f"  Saved {output_dir / 'secondary_activation_boxplot.png'}")
            plt.close(fig)

    # Plot 2: Profiling breakdown stacked bar (top 10 by time)
    sorted_valid = sorted(valid, key=lambda r: r.get("time_ms", 0), reverse=True)[:15]
    categories = [
        "Conv1D", "InputMixin", "Layer1x1", "Head1x1", "Rechannel",
        "Conv1x1", "Activation", "FiLM", "Copies", "Other",
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    filenames = [r["filename"][:20] for r in sorted_valid]
    bottoms = np.zeros(len(sorted_valid))

    for cat in categories:
        key = f"prof_{cat}"
        vals = np.array([float(r.get(key, 0) or 0) for r in sorted_valid])
        if vals.sum() > 0:
            ax.barh(range(len(sorted_valid)), vals, left=bottoms, label=cat)
            bottoms += vals

    ax.set_yticks(range(len(sorted_valid)))
    ax.set_yticklabels(filenames, fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Profiling Breakdown (Top 15 by Total Time)")
    ax.legend(loc="lower right", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "profiling_breakdown.png", dpi=150)
    print(f"  Saved {output_dir / 'profiling_breakdown.png'}")
    plt.close(fig)

    # Plot 3: Realtime factor distribution
    rt_factors = [r["realtime_factor"] for r in valid if isinstance(r.get("realtime_factor"), (int, float))]
    if rt_factors:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rt_factors, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(x=1.0, color="red", linestyle="--", label="Realtime threshold")
        ax.set_xlabel("Realtime Factor")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Realtime Factor")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "realtime_distribution.png", dpi=150)
        print(f"  Saved {output_dir / 'realtime_distribution.png'}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze WaveNet profiling results"
    )
    parser.add_argument(
        "results_csv", type=str,
        help="Path to results.csv (from parse_profiling_results.py)",
    )
    parser.add_argument(
        "--no-interaction", action="store_true",
        help="Skip the interaction/polynomial regression model",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate PNG plots (requires matplotlib)",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Directory for plot PNGs (default: same dir as results.csv)",
    )

    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading {results_path}...")
    rows = load_results(results_path)
    print(f"  Loaded {len(rows)} models")

    rows = compute_derived_features(rows)

    print_summary(rows)
    print_correlation(rows)
    run_regression(rows)
    if not args.no_interaction:
        run_interaction_regression(rows)

    # Save sorted CSV
    output_dir = results_path.parent
    save_sorted_csv(rows, output_dir)

    # Generate plots
    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir
        print(f"\nGenerating plots in {plot_dir}...")
        make_plots(rows, plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
