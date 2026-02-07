#!/usr/bin/env python3
"""
Script to build comparison figures for paper.

Creates a figure with original image and counterfactuals from each method side by side.

Usage:
    python scripts/build_comparison_figure.py --sweep outputs/sweep_20260201_140051 \
        --pathology Edema --sample 0 --direction increase

    python scripts/build_comparison_figure.py --sweep outputs/sweep_20260201_140051 \
        --pathology Atelectasis --sample 5 --direction decrease

    # With pixelwise difference row:
    python scripts/build_comparison_figure.py --sweep outputs/sweep_20260201_140051 \
        --pathology Edema --sample 0 --direction increase --show-diff
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image


# Display names for methods
METHOD_DISPLAY_NAMES = {
    "graph_fixed": "Graph",
    "text_fixed": "Text",
    "hybrid_fixed": "Hybrid",
}


def load_sweep_config(sweep_dir: Path) -> dict:
    """Load sweep configuration to get methods list."""
    config_path = sweep_dir / "sweep_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def get_methods_from_sweep(sweep_dir: Path) -> list[str]:
    """Get list of methods from sweep directory."""
    config = load_sweep_config(sweep_dir)
    methods_str = config.get("methods", "")
    return methods_str.split()


def get_image_path(
    sweep_dir: Path,
    method: str,
    pathology: str,
    sample_idx: int,
    image_type: str,
) -> Path:
    """
    Get path to an image.

    Args:
        sweep_dir: Path to sweep directory
        method: Method name (e.g., "graph_fixed")
        pathology: Pathology name (e.g., "Edema")
        sample_idx: Sample index number
        image_type: One of "original", "reconstructed", "increase", "decrease"

    Returns:
        Path to the image file
    """
    return sweep_dir / method / pathology / image_type / f"sample_{sample_idx:04d}.png"


def build_comparison_figure(
    sweep_dir: Path,
    pathology: str,
    sample_idx: int,
    direction: str = "increase",
    output_path: Optional[Path] = None,
    figsize: tuple[float, float] = (16, 4),
    dpi: int = 150,
    show_diff: bool = False,
    auto_scale_diff: bool = False,
) -> Path:
    """
    Build a comparison figure with original and counterfactuals from each method.

    When show_diff is enabled, the original image is on the left spanning both rows,
    with counterfactuals on the right (top row) and their diffs below (bottom row).

    Args:
        sweep_dir: Path to sweep directory
        pathology: Target pathology
        sample_idx: Sample index
        direction: "increase" or "decrease"
        output_path: Optional output path, defaults to sweep_dir/plots/
        figsize: Figure size in inches
        dpi: Output DPI
        show_diff: If True, add second row with pixelwise differences
        auto_scale_diff: If True, scale diff colormap to actual max diff (makes subtle diffs visible)

    Returns:
        Path to saved figure
    """
    methods = get_methods_from_sweep(sweep_dir)

    # Load original image (use first method's original since they're all the same)
    orig_path = get_image_path(sweep_dir, methods[0], pathology, sample_idx, "original")
    if not orig_path.exists():
        raise FileNotFoundError(f"Original image not found: {orig_path}")
    orig_img = Image.open(orig_path)

    # Load counterfactuals from each method
    cf_images = []
    cf_labels = []
    for method in methods:
        cf_path = get_image_path(sweep_dir, method, pathology, sample_idx, direction)
        if not cf_path.exists():
            raise FileNotFoundError(f"Counterfactual not found: {cf_path}")
        cf_images.append(Image.open(cf_path))
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        cf_labels.append(display_name)

    n_methods = len(methods)

    if show_diff:
        # Layout with diff: Original on left spanning 2 rows, methods + diffs on right
        # Grid: 2 rows x (n_methods + 1) cols, with original spanning both rows in col 0
        n_cols = n_methods + 1
        n_rows = 2

        actual_figsize = (figsize[0], figsize[1] * n_rows)
        fig = plt.figure(figsize=actual_figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig)

        # Original image: spans both rows in column 0
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_orig.imshow(orig_img, cmap="gray")
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        for spine in ax_orig.spines.values():
            spine.set_visible(False)
        ax_orig.set_title("Original", fontsize=14, fontweight="bold")

        # Counterfactuals: top row, columns 1 to n_methods
        orig_array = np.array(orig_img.convert("L"), dtype=np.float32)

        # Pre-compute diffs
        cf_arrays = [np.array(img.convert("L"), dtype=np.float32) for img in cf_images]
        diffs = [np.abs(orig_array - arr) for arr in cf_arrays]

        for i, (img, label, diff) in enumerate(zip(cf_images, cf_labels, diffs)):
            col = i + 1  # Offset by 1 for original column

            # Top row: counterfactual image
            ax_cf = fig.add_subplot(gs[0, col])
            ax_cf.imshow(img, cmap="gray")
            ax_cf.set_xticks([])
            ax_cf.set_yticks([])
            for spine in ax_cf.spines.values():
                spine.set_visible(False)
            ax_cf.set_title(label, fontsize=14, fontweight="bold")

            # Label for first counterfactual column
            if i == 0:
                ax_cf.set_ylabel("Counterfactual", fontsize=12, fontweight="bold")

            # Bottom row: pixelwise difference
            ax_diff = fig.add_subplot(gs[1, col])

            if auto_scale_diff:
                # Per-method scaling: each diff normalized to its own max
                diff_vmax = max(diff.max(), 1.0)  # Avoid division by zero
            else:
                diff_vmax = 255

            ax_diff.imshow(diff, vmin=0, vmax=diff_vmax)
            ax_diff.set_xticks([])
            ax_diff.set_yticks([])
            for spine in ax_diff.spines.values():
                spine.set_visible(False)

            # Label for first diff column
            if i == 0:
                ax_diff.set_ylabel("Difference", fontsize=12, fontweight="bold")

    else:
        # Simple layout without diff: Original first, then methods in a single row
        n_cols = n_methods + 1
        actual_figsize = figsize

        fig, axes = plt.subplots(1, n_cols, figsize=actual_figsize)
        if n_cols == 1:
            axes = [axes]

        # Original image first
        axes[0].imshow(orig_img, cmap="gray")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for spine in axes[0].spines.values():
            spine.set_visible(False)
        axes[0].set_xlabel("Original", fontsize=14, fontweight="bold")

        # Counterfactuals
        for i, (img, label) in enumerate(zip(cf_images, cf_labels)):
            ax = axes[i + 1]
            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(label, fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save figure
    if output_path is None:
        plots_dir = sweep_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        suffix = "_diff" if show_diff else ""
        filename = f"comparison_{pathology}_{sample_idx:04d}_{direction}{suffix}.png"
        output_path = plots_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build comparison figure for paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="Path to sweep directory",
    )
    parser.add_argument(
        "--pathology",
        type=str,
        required=True,
        help="Target pathology (e.g., Edema, Atelectasis)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        required=True,
        help="Sample index number",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["increase", "decrease"],
        default="increase",
        help="Counterfactual direction (default: increase)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: sweep_dir/plots/comparison_*.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[16, 4],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 16 4)",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Add second row with pixelwise differences",
    )
    parser.add_argument(
        "--auto-scale-diff",
        action="store_true",
        help="Auto-scale diff colormap to actual max diff (makes subtle diffs visible)",
    )

    args = parser.parse_args()

    build_comparison_figure(
        sweep_dir=args.sweep,
        pathology=args.pathology,
        sample_idx=args.sample,
        direction=args.direction,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        show_diff=args.show_diff,
        auto_scale_diff=args.auto_scale_diff,
    )


if __name__ == "__main__":
    main()
