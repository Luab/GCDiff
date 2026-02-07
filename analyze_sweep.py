#!/usr/bin/env python3
"""
Analyze sweep results comparing graph vs text counterfactual generation.

Extracts key metrics, computes derived statistics, and generates comparison tables.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(results_path: Path) -> Dict:
    """Load a single results.json file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def detect_classifier_format(eval_results: Dict) -> Tuple[bool, List[str]]:
    """
    Detect if results use multi-classifier format and return classifier names.
    
    Returns:
        Tuple of (is_multi_classifier, classifier_names)
        - For multi-classifier: (True, ['densenet121-res224-all', 'jfhealthcare', ...])
        - For single classifier (old format): (False, ['default'])
    """
    classifier_data = eval_results.get('classifier', {})
    
    # Check if 'aggregated' exists at top level (old single-classifier format)
    if 'aggregated' in classifier_data:
        return False, ['default']
    
    # Multi-classifier: keys are model names (filter out non-dict entries)
    classifier_names = [k for k, v in classifier_data.items() if isinstance(v, dict)]
    if classifier_names:
        return True, classifier_names
    
    # Fallback: empty or unknown format
    return False, ['default']


def load_sweep_results(sweep_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load all results from a sweep directory.
    
    Dynamically detects method directories (e.g., 'graph', 'graph_fixed', 'text_v2', etc.)
    by looking for subdirectories containing pathology folders with results.json files.
    
    Returns:
        Dict of {method_name: {pathology: results_dict}}
    """
    all_results = {}
    
    # Scan sweep directory for method subdirectories
    for method_dir in sweep_dir.iterdir():
        if not method_dir.is_dir():
            continue
        
        # Skip common non-method directories
        if method_dir.name in ('plots', 'logs', '__pycache__'):
            continue
        
        # Check if this directory contains pathology subdirs with results.json
        method_results = {}
        for pathology_dir in method_dir.iterdir():
            if pathology_dir.is_dir():
                results_file = pathology_dir / "results.json"
                if results_file.exists():
                    method_results[pathology_dir.name] = load_results(results_file)
        
        # Only add if we found valid results
        if method_results:
            all_results[method_dir.name] = method_results
    
    return all_results


def _get_classifier_aggregated(classifier_data: Dict, classifier_name: Optional[str], is_multi: bool) -> Dict:
    """Helper to get aggregated metrics from classifier data based on format."""
    if is_multi and classifier_name and classifier_name != 'default':
        # Multi-classifier format: classifier_data[model_name]['aggregated']
        return classifier_data.get(classifier_name, {}).get('aggregated', {})
    else:
        # Single classifier format: classifier_data['aggregated']
        return classifier_data.get('aggregated', {})


def recompute_flip_metrics(
    classifier_data: Dict,
    target_pathology: str,
    threshold: float,
    direction: Optional[str] = None,
) -> Dict:
    """
    Recompute flip metrics from per-image predictions with a custom threshold.
    
    Args:
        classifier_data: The classifier results dict (contains 'per_image' with raw predictions)
        target_pathology: The target pathology being edited
        threshold: Confidence threshold for significant change
        direction: 'increase' or 'decrease' or None (any direction)
    
    Returns:
        Dict with recomputed metrics matching the aggregated format
    """
    per_image = classifier_data.get('per_image', [])
    if not per_image:
        return {}
    
    n_images = len(per_image)
    target_flips = 0
    intended_flips = 0
    target_deltas = []
    non_target_preserved_counts = []
    perfect_preserved = 0
    
    # Get list of all pathologies from first image
    first_img = per_image[0]
    all_pathologies = list(first_img.get('confidence_deltas', {}).keys())
    n_non_target = len([p for p in all_pathologies if p != target_pathology])
    
    for img in per_image:
        deltas = img.get('confidence_deltas', {})
        
        if target_pathology not in deltas:
            continue
            
        target_delta = deltas[target_pathology]
        target_deltas.append(target_delta)
        
        # Check if target flipped (any direction)
        if abs(target_delta) > threshold:
            target_flips += 1
            
            # Check intended direction
            if direction == 'increase' and target_delta > threshold:
                intended_flips += 1
            elif direction == 'decrease' and target_delta < -threshold:
                intended_flips += 1
            elif direction is None:
                intended_flips += 1  # Any flip counts as intended if no direction specified
        
        # Count non-target preservation
        non_target_preserved = 0
        for p, delta in deltas.items():
            if p != target_pathology:
                if abs(delta) <= threshold:
                    non_target_preserved += 1
        
        non_target_preserved_counts.append(non_target_preserved)
        if n_non_target > 0 and non_target_preserved == n_non_target:
            perfect_preserved += 1
    
    mean_preservation = (
        np.mean(non_target_preserved_counts) / n_non_target 
        if n_non_target > 0 and non_target_preserved_counts 
        else 1.0
    )
    
    return {
        'target_metrics': {
            'flip_rate': target_flips / n_images if n_images > 0 else 0,
            'intended_flip_rate': intended_flips / n_images if n_images > 0 else 0,
            'mean_confidence_delta': float(np.mean(target_deltas)) if target_deltas else 0,
        },
        'non_target_preservation': {
            'mean_rate': mean_preservation,
            'perfect_preservation_rate': perfect_preserved / n_images if n_images > 0 else 0,
        },
    }


def extract_metrics(
    results: Dict, 
    method: str, 
    classifier_name: Optional[str] = None,
    override_threshold: Optional[float] = None,
) -> Dict:
    """
    Extract key metrics from a results dictionary.
    
    Args:
        results: The full results dictionary for a pathology/method
        method: 'graph' or 'text'
        classifier_name: Specific classifier name for multi-classifier results,
                        or None/'default' for single-classifier format
        override_threshold: If provided, recompute flip metrics with this threshold
    """
    metrics = {
        'method': method,
        'classifier': classifier_name or 'default',
        'num_samples': results.get('num_samples', 0),
    }
    
    # Edit stats
    edit_stats = results.get('edit_stats', {})
    metrics['edits_added'] = edit_stats.get('added', 0)
    metrics['edits_removed'] = edit_stats.get('removed', 0)
    
    # Combined evaluation metrics
    eval_combined = results.get('evaluation_combined', {})
    
    # Detect classifier format
    is_multi, _ = detect_classifier_format(eval_combined)
    
    # Get classifier data
    classifier_data = eval_combined.get('classifier', {})
    
    # Get the specific classifier's data for potential recomputation
    if is_multi and classifier_name and classifier_name != 'default':
        clf_specific_data = classifier_data.get(classifier_name, {})
    else:
        clf_specific_data = classifier_data
    
    # Use override threshold if provided and per_image data exists
    if override_threshold is not None and clf_specific_data.get('per_image'):
        target_pathology = results.get('target_pathology', '')
        # Determine direction from edit_stats (which direction dominates)
        added = edit_stats.get('added', 0)
        removed = edit_stats.get('removed', 0)
        if added > removed * 2:
            direction = 'increase'
        elif removed > added * 2:
            direction = 'decrease'
        else:
            direction = None  # Mixed - any flip counts
        
        agg = recompute_flip_metrics(
            clf_specific_data, target_pathology, override_threshold, direction
        )
    else:
        agg = _get_classifier_aggregated(classifier_data, classifier_name, is_multi)
    
    target_metrics = agg.get('target_metrics', {})
    non_target = agg.get('non_target_preservation', {})
    
    metrics['flip_rate'] = target_metrics.get('flip_rate', 0)
    metrics['intended_flip_rate'] = target_metrics.get('intended_flip_rate', 0)
    metrics['mean_confidence_delta'] = target_metrics.get('mean_confidence_delta', 0)
    metrics['non_target_preservation'] = non_target.get('mean_rate', 0)
    metrics['perfect_preservation_rate'] = non_target.get('perfect_preservation_rate', 0)
    
    # Image quality metrics (total change: original vs counterfactual)
    # Note: image metrics are per-method, not per-classifier
    image_metrics = eval_combined.get('image', {}).get('aggregated', {})
    metrics['lpips'] = image_metrics.get('lpips', {}).get('mean', 0)
    metrics['lpips_std'] = image_metrics.get('lpips', {}).get('std', 0)
    metrics['ssim'] = image_metrics.get('ssim', {}).get('mean', 0)
    metrics['ssim_std'] = image_metrics.get('ssim', {}).get('std', 0)
    metrics['l1'] = image_metrics.get('l1', {}).get('mean', 0)
    
    # FRD metrics (Frechet Radiomic Distance - distribution level, per method)
    frd = results.get('frd', {})
    metrics['frd_increase'] = frd.get('increase')
    metrics['frd_decrease'] = frd.get('decrease')
    metrics['frd_reconstructed'] = frd.get('reconstructed')  # graph only
    
    # Direction-specific metrics (format-aware)
    for direction in ['increase', 'decrease']:
        eval_dir = results.get(f'evaluation_{direction}', {})
        dir_classifier_data = eval_dir.get('classifier', {})
        
        # Use same format detection for direction-specific data
        dir_is_multi, _ = detect_classifier_format(eval_dir)
        dir_agg = _get_classifier_aggregated(dir_classifier_data, classifier_name, dir_is_multi)
        dir_target = dir_agg.get('target_metrics', {})
        
        metrics[f'{direction}_flip_rate'] = dir_target.get('flip_rate', 0)
        metrics[f'{direction}_intended_flip_rate'] = dir_target.get('intended_flip_rate', 0)
        metrics[f'{direction}_confidence_delta'] = dir_target.get('mean_confidence_delta', 0)
        metrics[f'{direction}_count'] = eval_dir.get('batch_size', 0)
    
    # Graph-specific: inversion fidelity and edit effect (decomposed metrics)
    if 'inversion_fidelity' in results:
        inv_fid = results['inversion_fidelity'].get('aggregated', {})
        metrics['inversion_lpips'] = inv_fid.get('lpips', {}).get('mean', 0)
        metrics['inversion_ssim'] = inv_fid.get('ssim', {}).get('mean', 0)
        
        edit_effect = results.get('edit_effect', {}).get('aggregated', {})
        metrics['edit_effect_lpips'] = edit_effect.get('lpips', {}).get('mean', 0)
        metrics['edit_effect_ssim'] = edit_effect.get('ssim', {}).get('mean', 0)
        
        # Edit precision: what fraction of change is pure edit vs inversion noise
        if metrics['lpips'] > 0:
            metrics['edit_precision'] = metrics['edit_effect_lpips'] / metrics['lpips']
        else:
            metrics['edit_precision'] = 0
    
    # Embedding similarity (graph only)
    if 'embedding_similarity' in results:
        emb_sim = results['embedding_similarity']
        metrics['embedding_similarity'] = emb_sim.get('mean', 0)
    
    return metrics


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics from the base metrics."""
    df = df.copy()
    
    # Collateral damage rate (unintended flips)
    df['collateral_damage'] = 1 - df['non_target_preservation']
    
    # Selectivity ratio: how well does it flip target vs accidentally flipping others
    # Higher is better - means more targeted edits
    df['selectivity'] = df.apply(
        lambda row: row['intended_flip_rate'] / row['collateral_damage'] 
        if row['collateral_damage'] > 0.01 else row['intended_flip_rate'] * 100,
        axis=1
    )
    
    # Edit efficiency: classifier change per unit of image change
    # Higher means more semantic change with less visual change
    df['edit_efficiency'] = df.apply(
        lambda row: abs(row['mean_confidence_delta']) / row['lpips'] 
        if row['lpips'] > 0.01 else 0,
        axis=1
    )
    
    return df


def build_comparison_df(
    all_results: Dict[str, Dict[str, Dict]],
    override_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame with metrics from all methods.
    
    For multi-classifier results, creates a row for each (pathology, method, classifier) combo.
    
    Args:
        all_results: Dict of method_name -> {pathology -> results_dict}
        override_threshold: If provided, recompute flip metrics with this threshold
    """
    rows = []
    
    # Collect all pathologies across all methods
    all_pathologies = set()
    for method_results in all_results.values():
        all_pathologies.update(method_results.keys())
    
    for pathology in sorted(all_pathologies):
        for method, results_dict in all_results.items():
            if pathology not in results_dict:
                continue
            
            results = results_dict[pathology]
            eval_combined = results.get('evaluation_combined', {})
            
            # Detect classifier format and iterate over classifiers
            is_multi, classifier_names = detect_classifier_format(eval_combined)
            
            for clf_name in classifier_names:
                metrics = extract_metrics(results, method, clf_name, override_threshold)
                metrics['pathology'] = pathology
                rows.append(metrics)
    
    df = pd.DataFrame(rows)
    df = compute_derived_metrics(df)
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print("COUNTERFACTUAL EVALUATION - PER PATHOLOGY & CLASSIFIER")
    print("=" * 120)
    
    pathologies = df['pathology'].unique()
    methods = df['method'].unique()
    classifiers = df['classifier'].unique()
    has_multi_classifier = len(classifiers) > 1 or (len(classifiers) == 1 and classifiers[0] != 'default')
    has_multi_method = len(methods) > 2  # More than just graph/text
    
    print(f"  Methods: {', '.join(methods)}")
    
    for pathology in pathologies:
        print(f"\n{'─' * 120}")
        print(f"  {pathology}")
        print(f"{'─' * 120}")
        
        pathology_df = df[df['pathology'] == pathology]
        
        if has_multi_classifier or has_multi_method:
            # Table format for multi-classifier or 3+ methods
            print(f"\n  {'Method':<8} {'Classifier':<25} {'Flip Rate':>12} {'Preserv.':>10} "
                  f"{'LPIPS':>8} {'SSIM':>8} {'FRD↓':>8} {'Conf Δ':>10}")
            print(f"  {'-' * 100}")
            
            for _, row in pathology_df.sort_values(['method', 'classifier']).iterrows():
                clf_short = row['classifier'][:22] + '...' if len(row['classifier']) > 25 else row['classifier']
                flip_rate = f"{row['intended_flip_rate']:.1%}" if pd.notna(row['intended_flip_rate']) else 'N/A'
                preserv = f"{row['non_target_preservation']:.1%}" if pd.notna(row['non_target_preservation']) else 'N/A'
                lpips = f"{row['lpips']:.4f}" if pd.notna(row['lpips']) else 'N/A'
                ssim = f"{row['ssim']:.4f}" if pd.notna(row['ssim']) else 'N/A'
                # FRD: use increase or decrease, whichever is available
                frd_val = row.get('frd_increase') or row.get('frd_decrease')
                frd = f"{frd_val:.4f}" if pd.notna(frd_val) else 'N/A'
                conf_delta = f"{row['mean_confidence_delta']:+.4f}" if pd.notna(row['mean_confidence_delta']) else 'N/A'
                
                print(f"  {row['method']:<8} {clf_short:<25} {flip_rate:>12} {preserv:>10} "
                      f"{lpips:>8} {ssim:>8} {frd:>8} {conf_delta:>10}")
        else:
            # Simple 2-method side-by-side comparison
            path_methods = pathology_df['method'].unique()
            if len(path_methods) < 2:
                print("  [Only one method available]")
                continue
            
            # Use first two methods for comparison
            method_a, method_b = path_methods[0], path_methods[1]
            row_a = pathology_df[pathology_df['method'] == method_a].iloc[0]
            row_b = pathology_df[pathology_df['method'] == method_b].iloc[0]
            
            metrics_config = [
                ('intended_flip_rate', 'Intended Flip Rate', True, '{:.1%}'),
                ('non_target_preservation', 'Non-Target Preservation', True, '{:.1%}'),
                ('selectivity', 'Selectivity Ratio', True, '{:.2f}'),
                ('lpips', 'LPIPS (↓)', False, '{:.4f}'),
                ('ssim', 'SSIM (↑)', True, '{:.4f}'),
                ('frd_increase', 'FRD Increase (↓)', False, '{:.4f}'),
                ('frd_decrease', 'FRD Decrease (↓)', False, '{:.4f}'),
                ('mean_confidence_delta', 'Confidence Delta', None, '{:+.4f}'),
            ]
            
            # Column headers using method names
            label_a = method_a.upper()[:15]
            label_b = method_b.upper()[:15]
            print(f"  {'Metric':<30} {label_a:>15} {label_b:>15} {'Winner':>12}")
            print(f"  {'-' * 72}")
            
            for metric, label, higher_better, fmt in metrics_config:
                a_val = row_a.get(metric, 0)
                b_val = row_b.get(metric, 0)
                
                # Skip if both are None/NaN (e.g., FRD not computed)
                if pd.isna(a_val) and pd.isna(b_val):
                    continue
                
                a_str = fmt.format(a_val) if pd.notna(a_val) else 'N/A'
                b_str = fmt.format(b_val) if pd.notna(b_val) else 'N/A'
                
                if higher_better is None or pd.isna(a_val) or pd.isna(b_val):
                    winner = ''
                elif higher_better:
                    winner = f'← {method_a.upper()[:8]}' if a_val > b_val else f'→ {method_b.upper()[:8]}' if b_val > a_val else 'TIE'
                else:
                    winner = f'← {method_a.upper()[:8]}' if a_val < b_val else f'→ {method_b.upper()[:8]}' if b_val < a_val else 'TIE'
                
                print(f"  {label:<30} {a_str:>15} {b_str:>15} {winner:>12}")
            
            # Graph-specific metrics (show for any method that has them)
            for _, row in pathology_df.iterrows():
                if 'edit_precision' in row and pd.notna(row['edit_precision']):
                    print(f"  {'Edit Precision (' + row['method'] + ')':<30} {row['edit_precision']:>15.2%}")


def print_aggregate_summary(df: pd.DataFrame):
    """Print aggregate summary across all pathologies."""
    print("\n" + "=" * 120)
    print("AGGREGATE SUMMARY (Mean ± Std across pathologies)")
    print("=" * 120)
    
    methods = sorted(df['method'].unique())
    classifiers = df['classifier'].unique()
    has_multi_classifier = len(classifiers) > 1 or (len(classifiers) == 1 and classifiers[0] != 'default')
    
    summary_metrics = [
        ('intended_flip_rate', 'Intended Flip Rate', True),
        ('non_target_preservation', 'Non-Target Preservation', True),
        ('selectivity', 'Selectivity Ratio', True),
        ('collateral_damage', 'Collateral Damage Rate', False),
        ('lpips', 'LPIPS', False),
        ('ssim', 'SSIM', True),
        ('frd_increase', 'FRD Increase', False),
        ('frd_decrease', 'FRD Decrease', False),
        ('edit_efficiency', 'Edit Efficiency', True),
    ]
    
    def _print_method_comparison(sub_df: pd.DataFrame, methods: List[str], title: str = ""):
        """Print comparison table for given methods."""
        if title:
            print(f"\n  ─── {title} ───")
        
        # Build header
        col_width = 18
        header = f"  {'Metric':<25}"
        for m in methods:
            header += f" {m.upper():>{col_width}}"
        header += f" {'Best':>8}"
        print(f"\n{header}")
        print(f"  {'-' * (25 + (col_width + 1) * len(methods) + 10)}")
        
        method_wins = {m: 0 for m in methods}
        
        for metric, label, higher_better in summary_metrics:
            if metric not in sub_df.columns:
                continue
            
            method_stats = {}
            for m in methods:
                m_df = sub_df[sub_df['method'] == m]
                vals = m_df[metric].dropna()
                if not vals.empty:
                    method_stats[m] = (vals.mean(), vals.std())
                else:
                    method_stats[m] = (float('nan'), float('nan'))
            
            # Skip if all NaN
            if all(pd.isna(method_stats[m][0]) for m in methods):
                continue
            
            row = f"  {label:<25}"
            valid_means = {m: method_stats[m][0] for m in methods if pd.notna(method_stats[m][0])}
            
            for m in methods:
                mean, std = method_stats[m]
                if pd.notna(mean):
                    row += f" {mean:>{col_width - 8}.4f} ± {std:.4f}"
                else:
                    row += f" {'N/A':>{col_width}}"
            
            # Determine best
            if valid_means:
                if higher_better:
                    best = max(valid_means, key=valid_means.get)
                else:
                    best = min(valid_means, key=valid_means.get)
                method_wins[best] += 1
                row += f" {best.upper():>8}"
            else:
                row += f" {'':>8}"
            
            print(row)
        
        # Summary
        wins_str = ", ".join(f"{m.upper()}={method_wins[m]}" for m in methods)
        print(f"\n  WINS: {wins_str}")
    
    if has_multi_classifier:
        # Multi-classifier: show breakdown by classifier
        for clf in classifiers:
            clf_df = df[df['classifier'] == clf]
            clf_short = clf[:30] + '...' if len(clf) > 33 else clf
            _print_method_comparison(clf_df, methods, f"Classifier: {clf_short}")
    else:
        _print_method_comparison(df, methods)
    
    # Direction breakdown (aggregate across all classifiers and methods)
    print("\n" + "-" * 80)
    print("  DIRECTION BREAKDOWN")
    print("-" * 80)
    
    for direction in ['increase', 'decrease']:
        col = f'{direction}_intended_flip_rate'
        if col in df.columns:
            parts = []
            for m in methods:
                m_df = df[df['method'] == m]
                m_mean = m_df[col].mean()
                if pd.notna(m_mean):
                    parts.append(f"{m.upper()}={m_mean:.1%}")
            if parts:
                # Find best
                method_means = {m: df[df['method'] == m][col].mean() for m in methods}
                valid = {k: v for k, v in method_means.items() if pd.notna(v)}
                best = max(valid, key=valid.get) if valid else ''
                print(f"  {direction.capitalize()} Flip Rate: {', '.join(parts)} → {best.upper()}")


def print_graph_specific_analysis(df: pd.DataFrame):
    """Print graph-specific decomposed metrics for methods that have them."""
    if 'edit_precision' not in df.columns:
        return
    
    # Find methods that have edit_precision data (graph-based methods)
    methods_with_decomposed = df[df['edit_precision'].notna()]['method'].unique()
    if len(methods_with_decomposed) == 0:
        return
    
    print("\n" + "=" * 100)
    print("DECOMPOSED METRICS (Graph-based methods)")
    print("=" * 100)
    print("\nThese metrics decompose total change into inversion error vs pure edit effect.")
    print("Edit Precision = edit_effect / total_change (lower = more change from inversion, not edit)\n")
    
    for method in methods_with_decomposed:
        method_df = df[df['method'] == method]
        print(f"\n  ─── {method.upper()} ───")
        print(f"  {'Pathology':<20} {'Inversion LPIPS':>18} {'Edit Effect LPIPS':>18} {'Edit Precision':>15}")
        print(f"  {'-' * 75}")
        
        for _, row in method_df.iterrows():
            inv_lpips = row.get('inversion_lpips', 0)
            edit_lpips = row.get('edit_effect_lpips', 0)
            precision = row.get('edit_precision', 0)
            
            if pd.notna(precision):
                print(f"  {row['pathology']:<20} {inv_lpips:>18.4f} {edit_lpips:>18.4f} {precision:>15.2%}")
        
        mean_precision = method_df['edit_precision'].mean()
        if pd.notna(mean_precision):
            print(f"\n  Mean Edit Precision: {mean_precision:.2%}")
            print(f"  → {100 - mean_precision * 100:.1f}% of visual change is from inversion, "
                  f"{mean_precision * 100:.1f}% is from the actual edit")


def plot_comparisons(df: pd.DataFrame, output_dir: Path, xkcd_style: bool = False):
    """Generate comparison boxplots as separate files."""
    if not HAS_MATPLOTLIB:
        print("\n[matplotlib not available - skipping plots]")
        return
    
    methods = sorted(df['method'].unique())
    
    # Base colors for known method types
    base_colors = {
        'graph': '#2ecc71',   # Green
        'text': '#3498db',    # Blue
        'hybrid': '#e74c3c',  # Red
    }
    
    # Extended palette for unknown methods
    extra_colors = ['#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff5722']
    
    # Build method_colors dynamically
    method_colors = {}
    extra_idx = 0
    for m in methods:
        # Check if method name contains a known base type
        matched = False
        for base, color in base_colors.items():
            if base in m.lower():
                method_colors[m] = color
                matched = True
                break
        if not matched:
            method_colors[m] = extra_colors[extra_idx % len(extra_colors)]
            extra_idx += 1
    
    palette = [method_colors.get(m, '#999999') for m in methods]
    
    # Define metrics to plot
    metrics_config = [
        ('intended_flip_rate', 'Intended Flip Rate', '↑ better', (0, 1)),
        ('non_target_preservation', 'Non-Target Preservation', '↑ better', (0, 1)),
        ('selectivity', 'Selectivity Ratio', '↑ better', None),
        ('lpips', 'LPIPS (Perceptual Distance)', '↓ better', None),
        ('ssim', 'SSIM (Structural Similarity)', '↑ better', (0, 1)),
        ('frd_increase', 'FRD Increase', '↓ better', None),
        ('frd_decrease', 'FRD Decrease', '↓ better', None),
        ('increase_intended_flip_rate', 'Flip Rate (Increase Direction)', '↑ better', (0, 1)),
        ('decrease_intended_flip_rate', 'Flip Rate (Decrease Direction)', '↑ better', (0, 1)),
    ]
    
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    for metric, title, direction, ylim in metrics_config:
        if metric not in df.columns:
            continue
        
        # Check if we have valid data for this metric
        valid_data = df[metric].dropna()
        if valid_data.empty:
            continue
        
        # Prepare data for barplot
        means = []
        stds = []
        labels = []
        methods_used = []  # Track original method names for color lookup
        for m in methods:
            m_vals = df[df['method'] == m][metric].dropna().values
            if len(m_vals) > 0:
                means.append(m_vals.mean())
                stds.append(m_vals.std() if len(m_vals) > 1 else 0)
                labels.append(m.upper())
                methods_used.append(m)
        
        if not means:
            continue
        
        # Create figure
        ctx = plt.xkcd() if xkcd_style else nullcontext()
        with ctx:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            x = np.arange(len(labels))
            colors_used = [method_colors.get(m, '#999999') for m in methods_used]
            
            bars = ax.bar(x, means, 
                         color=colors_used,
                         alpha=0.8,
                         yerr=stds,
                         capsize=5,
                         error_kw={'linewidth': 1.5})
            
            ax.set_title(f'{title} ({direction})', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_xlabel('Method', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            
            if ylim:
                ax.set_ylim(ylim)
            
            # Add grid (skip for xkcd)
            if not xkcd_style:
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save
            plot_path = plot_dir / f'barplot_{metric}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_plots.append(plot_path)
    
    # Per-pathology flip rate plot (bar chart with error bars)
    pathologies = sorted(df['pathology'].unique())
    metric = 'intended_flip_rate'
    
    if metric in df.columns:
        ctx = plt.xkcd() if xkcd_style else nullcontext()
        with ctx:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(pathologies))
            n_methods = len(methods)
            width = 0.25
            
            for i, m in enumerate(methods):
                means = []
                stds = []
                for pathology in pathologies:
                    path_df = df[(df['pathology'] == pathology) & (df['method'] == m)]
                    vals = path_df[metric].dropna().values
                    if len(vals) > 0:
                        means.append(vals.mean())
                        stds.append(vals.std() if len(vals) > 1 else 0)
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - (n_methods - 1) / 2) * width
                bars = ax.bar(x + offset, means, width, 
                             label=m.upper(), 
                             color=method_colors.get(m, '#999999'),
                             alpha=0.8,
                             yerr=stds,
                             capsize=3,
                             error_kw={'linewidth': 1.5})
            
            ax.set_title('Intended Flip Rate by Pathology (↑ better)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Intended Flip Rate', fontsize=12)
            ax.set_xlabel('Pathology', fontsize=12)
            ax.set_ylim(0, 1)
            
            ax.set_xticks(x)
            ax.set_xticklabels(pathologies, rotation=45, ha='right')
            
            if not xkcd_style:
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)
            
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            plot_path = plot_dir / 'barplot_flip_rate_by_pathology.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_plots.append(plot_path)
    
    print(f"\nSaved {len(saved_plots)} plots to: {plot_dir}")
    for p in saved_plots:
        print(f"  - {p.name}")


def save_results(df: pd.DataFrame, output_dir: Path):
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full comparison
    comparison_path = output_dir / "sweep_comparison.csv"
    df.to_csv(comparison_path, index=False)
    print(f"\nDetailed comparison saved to: {comparison_path}")
    
    # Summary aggregates
    methods = sorted(df['method'].unique())
    summary_rows = []
    for method in methods:
        method_df = df[df['method'] == method]
        if method_df.empty:
            continue
        summary = {'method': method}
        for col in df.select_dtypes(include=[np.number]).columns:
            summary[f'{col}_mean'] = method_df[col].mean()
            summary[f'{col}_std'] = method_df[col].std()
        summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sweep results comparing graph vs text counterfactual generation"
    )
    parser.add_argument("sweep_dir", type=str, help="Path to sweep output directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for CSV and plots (default: sweep_dir)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Override confidence threshold for flip detection "
                             "(recompute from raw predictions). Default uses stored metrics.")
    parser.add_argument("--xkcd", action="store_true", 
                        help="Use xkcd hand-drawn style for plots")
    
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output) if args.output else sweep_dir
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return 1
    
    print(f"Loading results from: {sweep_dir}")
    all_results = load_sweep_results(sweep_dir)
    
    # Print summary of found methods
    method_summary = ", ".join(f"{len(v)} {k}" for k, v in sorted(all_results.items()))
    print(f"Found: {method_summary}")
    
    if not all_results:
        print("Error: No results found!")
        return 1
    
    # Print threshold info
    if args.confidence_threshold is not None:
        print(f"\n*** Using custom confidence threshold: {args.confidence_threshold} ***")
        print("    (Flip rates recomputed from raw per-image predictions)\n")
    
    # Build comparison DataFrame
    df = build_comparison_df(all_results, args.confidence_threshold)
    
    # Print analysis
    print_comparison_table(df)
    print_aggregate_summary(df)
    print_graph_specific_analysis(df)
    
    # Save results
    save_results(df, output_dir)
    
    # Generate plots
    if not args.no_plots:
        plot_comparisons(df, output_dir, xkcd_style=args.xkcd)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    exit(main())


