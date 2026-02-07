#!/usr/bin/env python3
"""
Compute true 0.5 decision boundary crossing metrics from sweep/ablation results.

This script calculates actual binary classification flip rates (crossing the 0.5 threshold)
as opposed to the confidence-threshold-based 'intended_flip_rate' in the original evaluation.

Metrics computed:
- target_boundary_flip_rate: % of samples where target pathology crossed 0.5
- intended_boundary_flip_rate: % where target crossed 0.5 in the intended direction
- non_target_boundary_flip_rate: avg % of non-target pathologies that crossed 0.5
- non_target_preservation_rate: 1 - non_target_boundary_flip_rate

Usage:
    python scripts/compute_boundary_metrics.py outputs/sweep_20260126_123642/
    python scripts/compute_boundary_metrics.py outputs/template_ablation_20260129_135815/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def load_results(results_path: Path) -> Dict:
    """Load a single results.json file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_experiment_results(experiment_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load all results from an experiment directory (sweep or template ablation).
    
    Dynamically detects method/template directories by looking for subdirectories
    containing pathology folders with results.json files.
    
    Returns:
        Dict of {method_name: {pathology: results_dict}}
    """
    all_results = {}
    
    for method_dir in experiment_dir.iterdir():
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


def compute_boundary_crossing(pred_before: float, pred_after: float, threshold: float = 0.5) -> bool:
    """Check if prediction crossed the threshold."""
    before_positive = pred_before >= threshold
    after_positive = pred_after >= threshold
    return before_positive != after_positive


def compute_intended_boundary_crossing(
    pred_before: float, 
    pred_after: float, 
    direction: str,
    threshold: float = 0.5
) -> bool:
    """
    Check if prediction crossed the threshold in the intended direction.
    
    Args:
        pred_before: Prediction before edit
        pred_after: Prediction after edit
        direction: 'increase' or 'decrease'
        threshold: Decision threshold (default 0.5)
        
    Returns:
        True if crossed in intended direction
    """
    if direction == 'increase':
        # Intended: go from negative (<0.5) to positive (>=0.5)
        return pred_before < threshold and pred_after >= threshold
    elif direction == 'decrease':
        # Intended: go from positive (>=0.5) to negative (<0.5)
        return pred_before >= threshold and pred_after < threshold
    else:
        # Any crossing counts
        return compute_boundary_crossing(pred_before, pred_after, threshold)


def process_per_image_data(
    per_image_data: List[Dict],
    target_pathology: str,
    direction: str,
    threshold: float = 0.5
) -> Dict:
    """
    Process per-image predictions and compute boundary crossing metrics.
    
    Args:
        per_image_data: List of per-image prediction dictionaries
        target_pathology: Target pathology name
        direction: 'increase' or 'decrease'
        threshold: Decision threshold
        
    Returns:
        Dictionary with boundary crossing metrics
    """
    if not per_image_data:
        return None
    
    num_samples = len(per_image_data)
    target_crossings = 0
    intended_crossings = 0
    non_target_crossing_rates = []
    
    for item in per_image_data:
        preds_before = item.get('predictions_before', {})
        preds_after = item.get('predictions_after', {})
        
        if not preds_before or not preds_after:
            continue
        
        # Target pathology boundary crossing
        if target_pathology in preds_before and target_pathology in preds_after:
            target_before = preds_before[target_pathology]
            target_after = preds_after[target_pathology]
            
            if compute_boundary_crossing(target_before, target_after, threshold):
                target_crossings += 1
            
            if compute_intended_boundary_crossing(target_before, target_after, direction, threshold):
                intended_crossings += 1
        
        # Non-target pathology boundary crossings
        non_target_flips = 0
        non_target_count = 0
        
        for pathology in preds_before:
            if pathology == target_pathology:
                continue
            if pathology not in preds_after:
                continue
            
            non_target_count += 1
            if compute_boundary_crossing(preds_before[pathology], preds_after[pathology], threshold):
                non_target_flips += 1
        
        if non_target_count > 0:
            non_target_crossing_rates.append(non_target_flips / non_target_count)
    
    return {
        'num_samples': num_samples,
        'target_boundary_flip_rate': target_crossings / num_samples if num_samples > 0 else 0,
        'intended_boundary_flip_rate': intended_crossings / num_samples if num_samples > 0 else 0,
        'non_target_boundary_flip_rate': np.mean(non_target_crossing_rates) if non_target_crossing_rates else 0,
        'non_target_preservation_rate': 1 - (np.mean(non_target_crossing_rates) if non_target_crossing_rates else 0),
    }


def extract_boundary_metrics(
    results: Dict,
    target_pathology: str,
) -> List[Dict]:
    """
    Extract boundary crossing metrics from a results dictionary.
    
    Args:
        results: Full results dictionary from results.json
        target_pathology: Target pathology name
        
    Returns:
        List of metric dictionaries (one per classifier/direction combination)
    """
    metrics_list = []
    
    # Process each direction (increase, decrease, combined)
    for direction_key, direction_name in [
        ('evaluation_increase', 'increase'),
        ('evaluation_decrease', 'decrease'),
        ('evaluation_combined', 'combined'),
    ]:
        eval_data = results.get(direction_key, {})
        if not eval_data:
            continue
        
        classifier_data = eval_data.get('classifier', {})
        if not classifier_data:
            continue
        
        # Detect classifier format
        # Multi-classifier: keys are model names with nested 'per_image'
        # Single classifier: has 'aggregated' and 'per_image' at top level
        
        if 'aggregated' in classifier_data:
            # Single classifier format (old)
            per_image = classifier_data.get('per_image', [])
            if per_image:
                boundary_metrics = process_per_image_data(
                    per_image, target_pathology, direction_name
                )
                if boundary_metrics:
                    boundary_metrics['classifier'] = 'default'
                    boundary_metrics['direction'] = direction_name
                    metrics_list.append(boundary_metrics)
        else:
            # Multi-classifier format
            for classifier_name, clf_data in classifier_data.items():
                if not isinstance(clf_data, dict):
                    continue
                
                per_image = clf_data.get('per_image', [])
                if per_image:
                    boundary_metrics = process_per_image_data(
                        per_image, target_pathology, direction_name
                    )
                    if boundary_metrics:
                        boundary_metrics['classifier'] = classifier_name
                        boundary_metrics['direction'] = direction_name
                        metrics_list.append(boundary_metrics)
    
    return metrics_list


def build_boundary_metrics_df(all_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """
    Build a DataFrame with boundary crossing metrics from all results.
    
    Args:
        all_results: Dict of {method: {pathology: results_dict}}
        
    Returns:
        DataFrame with boundary metrics
    """
    rows = []
    
    for method_name, method_results in all_results.items():
        for pathology, results in method_results.items():
            target_pathology = results.get('target_pathology', pathology)
            
            metrics_list = extract_boundary_metrics(results, target_pathology)
            
            for metrics in metrics_list:
                row = {
                    'method': method_name,
                    'pathology': pathology,
                    **metrics
                }
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = [
        'method', 'pathology', 'classifier', 'direction', 'num_samples',
        'target_boundary_flip_rate', 'intended_boundary_flip_rate',
        'non_target_boundary_flip_rate', 'non_target_preservation_rate'
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]
    
    return df


def build_summary_table(df: pd.DataFrame, classifier: str = 'densenet121-res224-all') -> pd.DataFrame:
    """
    Build a summary table with methods as rows and averaged metrics as columns.
    
    Args:
        df: Full boundary metrics DataFrame
        classifier: Classifier to use for summary (default: densenet121-res224-all)
        
    Returns:
        Summary DataFrame with one row per method
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to specified classifier and combined direction
    classifiers = df['classifier'].unique()
    if classifier not in classifiers:
        classifier = classifiers[0]
    
    subset = df[(df['classifier'] == classifier) & (df['direction'] == 'combined')]
    
    if subset.empty:
        # Fallback to any direction
        subset = df[df['classifier'] == classifier]
    
    if subset.empty:
        return pd.DataFrame()
    
    # Group by method and compute mean ± std
    methods = sorted(subset['method'].unique())
    rows = []
    
    for method in methods:
        method_df = subset[subset['method'] == method]
        
        row = {'method': method}
        
        for metric in ['target_boundary_flip_rate', 'intended_boundary_flip_rate', 
                       'non_target_boundary_flip_rate', 'non_target_preservation_rate']:
            if metric in method_df.columns:
                mean_val = method_df[metric].mean()
                std_val = method_df[metric].std()
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                row[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    if df.empty:
        print("No data to summarize.")
        return
    
    print("\n" + "=" * 100)
    print("BOUNDARY CROSSING METRICS SUMMARY")
    print("=" * 100)
    
    # Get unique values
    methods = sorted(df['method'].unique())
    classifiers = df['classifier'].unique()
    
    print(f"\nMethods: {', '.join(methods)}")
    print(f"Classifiers: {', '.join(classifiers)}")
    print(f"Total rows: {len(df)}")
    
    # Summary by method and direction (for primary classifier if multiple)
    primary_clf = 'densenet121-res224-all' if 'densenet121-res224-all' in classifiers else classifiers[0]
    
    print(f"\n--- Summary for {primary_clf} ---")
    
    for direction in ['increase', 'decrease', 'combined']:
        subset = df[(df['classifier'] == primary_clf) & (df['direction'] == direction)]
        if subset.empty:
            continue
        
        print(f"\n{direction.upper()} Direction:")
        for method in methods:
            method_subset = subset[subset['method'] == method]
            if method_subset.empty:
                continue
            
            target_flip = method_subset['target_boundary_flip_rate'].mean()
            intended_flip = method_subset['intended_boundary_flip_rate'].mean()
            preservation = method_subset['non_target_preservation_rate'].mean()
            
            print(f"  {method:20s}: target_flip={target_flip:.1%}, intended_flip={intended_flip:.1%}, preservation={preservation:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute true 0.5 boundary crossing metrics from experiment results"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment output directory (sweep or template ablation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <experiment_dir>/boundary_metrics.csv)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Directory not found: {experiment_dir}")
        return 1
    
    # Determine output path
    output_path = Path(args.output) if args.output else experiment_dir / "boundary_metrics.csv"
    
    print(f"Loading results from: {experiment_dir}")
    
    # Load all results
    all_results = load_experiment_results(experiment_dir)
    
    if not all_results:
        print("Error: No valid results found!")
        return 1
    
    print(f"Found {len(all_results)} methods/templates:")
    for method_name, method_results in all_results.items():
        print(f"  - {method_name}: {len(method_results)} pathologies")
    
    # Build metrics DataFrame
    print("\nComputing boundary crossing metrics...")
    df = build_boundary_metrics_df(all_results)
    
    if df.empty:
        print("Error: No per-image prediction data found in results!")
        print("Note: Older results may not have per-image predictions stored.")
        return 1
    
    # Save detailed results to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Build and save summary tables - one per classifier
    classifiers = sorted(df['classifier'].unique())
    
    for clf in classifiers:
        summary_df = build_summary_table(df, classifier=clf)
        if not summary_df.empty:
            # Create filename with classifier name (sanitize for filesystem)
            clf_safe = clf.replace('/', '_').replace(' ', '_')
            summary_path = output_path.parent / f"boundary_metrics_{clf_safe}.csv"
            
            # Save clean version with formatted strings
            summary_cols = ['method', 'intended_boundary_flip_rate', 'non_target_preservation_rate']
            summary_clean = summary_df[['method'] + [c for c in summary_cols[1:] if c in summary_df.columns]]
            summary_clean.to_csv(summary_path, index=False)
            print(f"Summary table ({clf}) saved to: {summary_path}")
            
            # Print summary table
            print("\n" + "-" * 80)
            print(f"SUMMARY TABLE (averaged across pathologies, classifier: {clf})")
            print("-" * 80)
            print(summary_clean.to_string(index=False))
    
    # Print detailed summary
    print_summary(df)
    
    print("\n" + "=" * 100)
    print("Done!")
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    exit(main())
