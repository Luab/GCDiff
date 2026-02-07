#!/usr/bin/env python3
"""
Analyze template ablation results comparing different prompt template sets.

Extracts key metrics, computes derived statistics, and generates comparison tables
for default, freeform, and detailed template styles.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Template set descriptions for reporting
TEMPLATE_DESCRIPTIONS = {
    'default': 'Structured ("Chest x-ray showing X")',
    'freeform': 'Simple ("X is present")',
    'detailed': 'Clinical ("...with enlarged cardiac silhouette")',
}


def load_results(results_path: Path) -> Dict:
    """Load a single results.json file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def detect_classifier_format(eval_results: Dict) -> Tuple[bool, List[str]]:
    """
    Detect if results use multi-classifier format and return classifier names.
    
    Returns:
        Tuple of (is_multi_classifier, classifier_names)
    """
    classifier_data = eval_results.get('classifier', {})
    
    # Check if 'aggregated' exists at top level (old single-classifier format)
    if 'aggregated' in classifier_data:
        return False, ['default']
    
    # Multi-classifier: keys are model names (filter out non-dict entries)
    classifier_names = [k for k, v in classifier_data.items() if isinstance(v, dict)]
    if classifier_names:
        return True, classifier_names
    
    return False, ['default']


def load_ablation_results(ablation_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load all results from a template ablation directory.
    
    Returns:
        Dict of {template_set: {pathology: results_dict}}
    """
    results = {}
    
    for template_dir in ablation_dir.iterdir():
        if template_dir.is_dir() and template_dir.name not in ['plots', 'logs']:
            template_set = template_dir.name
            results[template_set] = {}
            
            for pathology_dir in template_dir.iterdir():
                if pathology_dir.is_dir():
                    results_file = pathology_dir / "results.json"
                    if results_file.exists():
                        results[template_set][pathology_dir.name] = load_results(results_file)
    
    return results


def _get_classifier_aggregated(classifier_data: Dict, classifier_name: Optional[str], is_multi: bool) -> Dict:
    """Helper to get aggregated metrics from classifier data based on format."""
    if is_multi and classifier_name and classifier_name != 'default':
        return classifier_data.get(classifier_name, {}).get('aggregated', {})
    else:
        return classifier_data.get('aggregated', {})


def extract_metrics(
    results: Dict,
    template_set: str,
    classifier_name: Optional[str] = None,
) -> Dict:
    """
    Extract key metrics from a results dictionary.
    
    Args:
        results: The full results dictionary for a pathology/template_set
        template_set: Name of the template set ('default', 'freeform', 'detailed')
        classifier_name: Specific classifier name for multi-classifier results
    """
    metrics = {
        'template_set': template_set,
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
    agg = _get_classifier_aggregated(classifier_data, classifier_name, is_multi)
    
    target_metrics = agg.get('target_metrics', {})
    non_target = agg.get('non_target_preservation', {})
    
    metrics['flip_rate'] = target_metrics.get('flip_rate', 0)
    metrics['intended_flip_rate'] = target_metrics.get('intended_flip_rate', 0)
    metrics['mean_confidence_delta'] = target_metrics.get('mean_confidence_delta', 0)
    metrics['non_target_preservation'] = non_target.get('mean_rate', 0)
    metrics['perfect_preservation_rate'] = non_target.get('perfect_preservation_rate', 0)
    
    # Image quality metrics
    image_metrics = eval_combined.get('image', {}).get('aggregated', {})
    metrics['lpips'] = image_metrics.get('lpips', {}).get('mean', 0)
    metrics['lpips_std'] = image_metrics.get('lpips', {}).get('std', 0)
    metrics['ssim'] = image_metrics.get('ssim', {}).get('mean', 0)
    metrics['ssim_std'] = image_metrics.get('ssim', {}).get('std', 0)
    metrics['l1'] = image_metrics.get('l1', {}).get('mean', 0)
    
    # FRD metrics
    frd = results.get('frd', {})
    metrics['frd_increase'] = frd.get('increase')
    metrics['frd_decrease'] = frd.get('decrease')
    metrics['frd_reconstructed'] = frd.get('reconstructed')
    
    # Direction-specific metrics
    for direction in ['increase', 'decrease']:
        eval_dir = results.get(f'evaluation_{direction}', {})
        dir_classifier_data = eval_dir.get('classifier', {})
        
        dir_is_multi, _ = detect_classifier_format(eval_dir)
        dir_agg = _get_classifier_aggregated(dir_classifier_data, classifier_name, dir_is_multi)
        dir_target = dir_agg.get('target_metrics', {})
        
        metrics[f'{direction}_flip_rate'] = dir_target.get('flip_rate', 0)
        metrics[f'{direction}_intended_flip_rate'] = dir_target.get('intended_flip_rate', 0)
        metrics[f'{direction}_confidence_delta'] = dir_target.get('mean_confidence_delta', 0)
        metrics[f'{direction}_count'] = eval_dir.get('batch_size', 0)
    
    # Inversion fidelity metrics (if available)
    if 'inversion_fidelity' in results:
        inv_fid = results['inversion_fidelity'].get('aggregated', {})
        metrics['inversion_lpips'] = inv_fid.get('lpips', {}).get('mean', 0)
        metrics['inversion_ssim'] = inv_fid.get('ssim', {}).get('mean', 0)
        
        edit_effect = results.get('edit_effect', {}).get('aggregated', {})
        metrics['edit_effect_lpips'] = edit_effect.get('lpips', {}).get('mean', 0)
        metrics['edit_effect_ssim'] = edit_effect.get('ssim', {}).get('mean', 0)
    
    return metrics


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics from the base metrics."""
    df = df.copy()
    
    # Collateral damage rate
    df['collateral_damage'] = 1 - df['non_target_preservation']
    
    # Selectivity ratio
    df['selectivity'] = df.apply(
        lambda row: row['intended_flip_rate'] / row['collateral_damage'] 
        if row['collateral_damage'] > 0.01 else row['intended_flip_rate'] * 100,
        axis=1
    )
    
    # Edit efficiency
    df['edit_efficiency'] = df.apply(
        lambda row: abs(row['mean_confidence_delta']) / row['lpips'] 
        if row['lpips'] > 0.01 else 0,
        axis=1
    )
    
    return df


# Default classifier to use for analysis (for consistency with sweep results)
DEFAULT_CLASSIFIER = 'densenet121-res224-all'


def build_comparison_df(ablation_results: Dict[str, Dict[str, Dict]], classifier_filter: str = DEFAULT_CLASSIFIER) -> pd.DataFrame:
    """
    Build a comparison DataFrame with metrics from all template sets.
    
    Args:
        ablation_results: Dict of {template_set: {pathology: results_dict}}
        classifier_filter: Only use this classifier (default: densenet121-res224-all)
    """
    rows = []
    
    all_pathologies = set()
    for template_results in ablation_results.values():
        all_pathologies.update(template_results.keys())
    
    for pathology in sorted(all_pathologies):
        for template_set, template_results in ablation_results.items():
            if pathology not in template_results:
                continue
            
            results = template_results[pathology]
            eval_combined = results.get('evaluation_combined', {})
            
            # Detect classifier format
            is_multi, classifier_names = detect_classifier_format(eval_combined)
            
            # Filter to only use the specified classifier
            if is_multi and classifier_filter in classifier_names:
                clf_name = classifier_filter
            elif is_multi and classifier_names:
                # Fallback to first classifier if filter not found
                clf_name = classifier_names[0]
                print(f"Warning: Classifier '{classifier_filter}' not found, using '{clf_name}'")
            else:
                clf_name = 'default'
            
            metrics = extract_metrics(results, template_set, clf_name)
            metrics['pathology'] = pathology
            rows.append(metrics)
    
    df = pd.DataFrame(rows)
    df = compute_derived_metrics(df)
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print("\n" + "=" * 130)
    print("TEMPLATE ABLATION - PER PATHOLOGY & CLASSIFIER")
    print("=" * 130)
    
    pathologies = df['pathology'].unique()
    template_sets = sorted(df['template_set'].unique())
    classifiers = df['classifier'].unique()
    has_multi_classifier = len(classifiers) > 1 or (len(classifiers) == 1 and classifiers[0] != 'default')
    
    print(f"\n  Template Sets: {', '.join(template_sets)}")
    for ts in template_sets:
        desc = TEMPLATE_DESCRIPTIONS.get(ts, ts)
        print(f"    - {ts}: {desc}")
    
    for pathology in pathologies:
        print(f"\n{'─' * 130}")
        print(f"  {pathology}")
        print(f"{'─' * 130}")
        
        pathology_df = df[df['pathology'] == pathology]
        
        # Table format
        print(f"\n  {'Template':<12} {'Classifier':<25} {'Flip Rate':>12} {'Intended':>12} {'Preserv.':>10} "
              f"{'LPIPS':>8} {'SSIM':>8} {'Conf Δ':>10}")
        print(f"  {'-' * 110}")
        
        for _, row in pathology_df.sort_values(['template_set', 'classifier']).iterrows():
            clf_short = row['classifier'][:22] + '...' if len(row['classifier']) > 25 else row['classifier']
            flip_rate = f"{row['flip_rate']:.1%}" if pd.notna(row['flip_rate']) else 'N/A'
            intended = f"{row['intended_flip_rate']:.1%}" if pd.notna(row['intended_flip_rate']) else 'N/A'
            preserv = f"{row['non_target_preservation']:.1%}" if pd.notna(row['non_target_preservation']) else 'N/A'
            lpips = f"{row['lpips']:.4f}" if pd.notna(row['lpips']) else 'N/A'
            ssim = f"{row['ssim']:.4f}" if pd.notna(row['ssim']) else 'N/A'
            conf_delta = f"{row['mean_confidence_delta']:+.4f}" if pd.notna(row['mean_confidence_delta']) else 'N/A'
            
            print(f"  {row['template_set']:<12} {clf_short:<25} {flip_rate:>12} {intended:>12} {preserv:>10} "
                  f"{lpips:>8} {ssim:>8} {conf_delta:>10}")


def print_aggregate_summary(df: pd.DataFrame):
    """Print aggregate summary across all pathologies."""
    print("\n" + "=" * 130)
    print("AGGREGATE SUMMARY (Mean ± Std across pathologies)")
    print("=" * 130)
    
    template_sets = sorted(df['template_set'].unique())
    classifiers = df['classifier'].unique()
    has_multi_classifier = len(classifiers) > 1 or (len(classifiers) == 1 and classifiers[0] != 'default')
    
    summary_metrics = [
        ('intended_flip_rate', 'Intended Flip Rate', True),
        ('flip_rate', 'Total Flip Rate', True),
        ('non_target_preservation', 'Non-Target Preservation', True),
        ('selectivity', 'Selectivity Ratio', True),
        ('collateral_damage', 'Collateral Damage Rate', False),
        ('lpips', 'LPIPS', False),
        ('ssim', 'SSIM', True),
        ('frd_increase', 'FRD Increase', False),
        ('frd_decrease', 'FRD Decrease', False),
        ('edit_efficiency', 'Edit Efficiency', True),
    ]
    
    def _print_template_comparison(sub_df: pd.DataFrame, template_sets: List[str], title: str = ""):
        """Print comparison table for given template sets."""
        if title:
            print(f"\n  ─── {title} ───")
        
        col_width = 20
        header = f"  {'Metric':<28}"
        for ts in template_sets:
            header += f" {ts.upper():>{col_width}}"
        header += f" {'Best':>10}"
        print(f"\n{header}")
        print(f"  {'-' * (28 + (col_width + 1) * len(template_sets) + 12)}")
        
        template_wins = {ts: 0 for ts in template_sets}
        
        for metric, label, higher_better in summary_metrics:
            if metric not in sub_df.columns:
                continue
            
            template_stats = {}
            for ts in template_sets:
                ts_df = sub_df[sub_df['template_set'] == ts]
                vals = ts_df[metric].dropna()
                if not vals.empty:
                    template_stats[ts] = (vals.mean(), vals.std())
                else:
                    template_stats[ts] = (float('nan'), float('nan'))
            
            # Skip if all NaN
            if all(pd.isna(template_stats[ts][0]) for ts in template_sets):
                continue
            
            row = f"  {label:<28}"
            valid_means = {ts: template_stats[ts][0] for ts in template_sets if pd.notna(template_stats[ts][0])}
            
            for ts in template_sets:
                mean, std = template_stats[ts]
                if pd.notna(mean):
                    row += f" {mean:>{col_width - 9}.4f} ± {std:.4f}"
                else:
                    row += f" {'N/A':>{col_width}}"
            
            # Determine best
            if valid_means:
                if higher_better:
                    best = max(valid_means, key=valid_means.get)
                else:
                    best = min(valid_means, key=valid_means.get)
                template_wins[best] += 1
                row += f" {best.upper():>10}"
            else:
                row += f" {'':>10}"
            
            print(row)
        
        # Summary
        wins_str = ", ".join(f"{ts.upper()}={template_wins[ts]}" for ts in template_sets)
        print(f"\n  WINS: {wins_str}")
        
        # Determine overall winner
        max_wins = max(template_wins.values())
        winners = [ts for ts, wins in template_wins.items() if wins == max_wins]
        if len(winners) == 1:
            print(f"  OVERALL BEST TEMPLATE: {winners[0].upper()}")
        else:
            print(f"  TIE: {', '.join(w.upper() for w in winners)}")
    
    if has_multi_classifier:
        # Multi-classifier: show breakdown by classifier
        for clf in classifiers:
            clf_df = df[df['classifier'] == clf]
            clf_short = clf[:35] + '...' if len(clf) > 38 else clf
            _print_template_comparison(clf_df, template_sets, f"Classifier: {clf_short}")
    else:
        _print_template_comparison(df, template_sets)
    
    # Direction breakdown
    print("\n" + "-" * 100)
    print("  DIRECTION BREAKDOWN (Mean intended flip rate)")
    print("-" * 100)
    
    for direction in ['increase', 'decrease']:
        col = f'{direction}_intended_flip_rate'
        if col in df.columns:
            parts = []
            template_means = {}
            for ts in template_sets:
                ts_df = df[df['template_set'] == ts]
                ts_mean = ts_df[col].mean()
                template_means[ts] = ts_mean
                if pd.notna(ts_mean):
                    parts.append(f"{ts.upper()}={ts_mean:.1%}")
            if parts:
                valid = {k: v for k, v in template_means.items() if pd.notna(v)}
                best = max(valid, key=valid.get) if valid else ''
                print(f"  {direction.capitalize():<10}: {', '.join(parts):<50} → Best: {best.upper()}")


def print_template_analysis(df: pd.DataFrame):
    """Print analysis specific to template comparison."""
    print("\n" + "=" * 130)
    print("TEMPLATE STYLE ANALYSIS")
    print("=" * 130)
    
    template_sets = sorted(df['template_set'].unique())
    
    # Per-template summary
    for ts in template_sets:
        ts_df = df[df['template_set'] == ts]
        desc = TEMPLATE_DESCRIPTIONS.get(ts, ts)
        
        print(f"\n  {ts.upper()}: {desc}")
        print(f"  {'-' * 60}")
        
        avg_flip = ts_df['intended_flip_rate'].mean()
        avg_preserv = ts_df['non_target_preservation'].mean()
        avg_lpips = ts_df['lpips'].mean()
        avg_ssim = ts_df['ssim'].mean()
        avg_delta = ts_df['mean_confidence_delta'].mean()
        
        print(f"    Avg Intended Flip Rate:   {avg_flip:.1%}")
        print(f"    Avg Preservation:         {avg_preserv:.1%}")
        print(f"    Avg LPIPS:                {avg_lpips:.4f}")
        print(f"    Avg SSIM:                 {avg_ssim:.4f}")
        print(f"    Avg Confidence Delta:     {avg_delta:+.4f}")
    
    # Relative comparison
    print("\n" + "-" * 100)
    print("  RELATIVE PERFORMANCE (vs Default)")
    print("-" * 100)
    
    if 'default' in template_sets:
        default_df = df[df['template_set'] == 'default']
        default_flip = default_df['intended_flip_rate'].mean()
        default_lpips = default_df['lpips'].mean()
        
        for ts in template_sets:
            if ts == 'default':
                continue
            ts_df = df[df['template_set'] == ts]
            ts_flip = ts_df['intended_flip_rate'].mean()
            ts_lpips = ts_df['lpips'].mean()
            
            flip_diff = ts_flip - default_flip
            lpips_diff = ts_lpips - default_lpips
            
            flip_pct = (flip_diff / default_flip * 100) if default_flip > 0 else 0
            lpips_pct = (lpips_diff / default_lpips * 100) if default_lpips > 0 else 0
            
            print(f"\n  {ts.upper()} vs DEFAULT:")
            print(f"    Flip Rate:  {flip_diff:+.1%} ({flip_pct:+.1f}% relative)")
            print(f"    LPIPS:      {lpips_diff:+.4f} ({lpips_pct:+.1f}% relative)")


def plot_comparisons(df: pd.DataFrame, output_path: Path):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("\n[matplotlib not available - skipping plots]")
        return
    
    pathologies = df['pathology'].unique()
    n_pathologies = len(pathologies)
    template_sets = sorted(df['template_set'].unique())
    
    # Colors for template sets
    template_colors = {
        'default': '#3498db',   # Blue
        'freeform': '#2ecc71',  # Green
        'detailed': '#e74c3c',  # Red
    }
    
    # Use df directly (single classifier)
    agg_df = df
    
    # Create template dataframes
    template_dfs = {ts: agg_df[agg_df['template_set'] == ts].set_index('pathology') for ts in template_sets}
    
    # Check if we have FRD data
    has_frd = ('frd_increase' in agg_df.columns and agg_df['frd_increase'].notna().any())
    
    if has_frd:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle('Template Ablation: Prompt Style Comparison', fontsize=14, fontweight='bold')
    
    x = np.arange(n_pathologies)
    n_templates = len(template_sets)
    width = 0.8 / n_templates
    
    def _get_vals(template_df, pathology, metric):
        """Get metric value for a pathology, returning 0 if missing."""
        if pathology not in template_df.index:
            return 0
        val = template_df.loc[pathology, metric]
        return val if pd.notna(val) else 0
    
    def _plot_grouped_bar(ax, metric, title, ylim=None):
        """Plot grouped bar chart for a metric."""
        for i, ts in enumerate(template_sets):
            ts_df = template_dfs[ts]
            vals = [_get_vals(ts_df, p, metric) for p in pathologies]
            offset = (i - (n_templates - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=ts.upper(), 
                   color=template_colors.get(ts, '#999999'))
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(pathologies, rotation=45, ha='right')
        ax.legend()
        if ylim:
            ax.set_ylim(ylim)
    
    # Intended Flip Rate
    _plot_grouped_bar(axes[0, 0], 'intended_flip_rate', 'Intended Flip Rate (↑ better)', (0, 1))
    axes[0, 0].set_ylabel('Rate')
    
    # Non-target Preservation
    _plot_grouped_bar(axes[0, 1], 'non_target_preservation', 'Non-Target Preservation (↑ better)', (0, 1))
    axes[0, 1].set_ylabel('Rate')
    
    # LPIPS
    _plot_grouped_bar(axes[1, 0], 'lpips', 'LPIPS - Perceptual Distance (↓ better)')
    axes[1, 0].set_ylabel('LPIPS')
    
    # Selectivity or Confidence Delta
    _plot_grouped_bar(axes[1, 1], 'mean_confidence_delta', 'Mean Confidence Delta')
    axes[1, 1].set_ylabel('Delta')
    
    # FRD plots (if available)
    if has_frd:
        _plot_grouped_bar(axes[0, 2], 'frd_increase', 'FRD Increase (↓ better)')
        axes[0, 2].set_ylabel('FRD')
        
        _plot_grouped_bar(axes[1, 2], 'frd_decrease', 'FRD Decrease (↓ better)')
        axes[1, 2].set_ylabel('FRD')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def save_results(df: pd.DataFrame, output_dir: Path):
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full comparison
    comparison_path = output_dir / "template_ablation_comparison.csv"
    df.to_csv(comparison_path, index=False)
    print(f"\nDetailed comparison saved to: {comparison_path}")
    
    # Summary aggregates by template set
    template_sets = sorted(df['template_set'].unique())
    summary_rows = []
    for ts in template_sets:
        ts_df = df[df['template_set'] == ts]
        if ts_df.empty:
            continue
        summary = {'template_set': ts}
        for col in df.select_dtypes(include=[np.number]).columns:
            summary[f'{col}_mean'] = ts_df[col].mean()
            summary[f'{col}_std'] = ts_df[col].std()
        summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "template_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # Pivot table for easy comparison
    pivot_metrics = ['intended_flip_rate', 'non_target_preservation', 'lpips', 'ssim', 'mean_confidence_delta']
    # Higher is better for these metrics
    higher_is_better = {'intended_flip_rate', 'non_target_preservation', 'ssim'}
    
    for metric in pivot_metrics:
        if metric in df.columns:
            # Create pivot with mean values
            pivot_mean = df.pivot_table(
                index='pathology',
                columns='template_set',
                values=metric,
                aggfunc='mean'
            )
            
            if pivot_mean.empty:
                continue
            
            # Add average row to pivot_mean
            avg_row = pivot_mean.mean()
            std_row = pivot_mean.std()
            pivot_mean.loc['Average'] = avg_row
            
            # For intended_flip_rate and non_target_preservation, include std
            if metric in ['intended_flip_rate', 'non_target_preservation']:
                # Create combined table with mean ± std format
                # For individual pathologies: just show value (no std with single classifier)
                # For Average row: show mean ± std across pathologies
                combined = pivot_mean.copy().astype(object)
                for idx in combined.index:
                    row_means = {col: pivot_mean.loc[idx, col] for col in combined.columns}
                    valid_means = {k: v for k, v in row_means.items() if pd.notna(v)}
                    
                    if valid_means:
                        if metric in higher_is_better:
                            best_col = max(valid_means, key=valid_means.get)
                        else:
                            best_col = min(valid_means, key=valid_means.get)
                    else:
                        best_col = None
                    
                    for col in combined.columns:
                        mean_val = pivot_mean.loc[idx, col]
                        if idx == 'Average':
                            # For Average row, show std across pathologies
                            std_val = std_row[col] if col in std_row else np.nan
                            if pd.notna(mean_val) and pd.notna(std_val):
                                formatted = f"{mean_val:.3f} ± {std_val:.3f}"
                            elif pd.notna(mean_val):
                                formatted = f"{mean_val:.3f}"
                            else:
                                formatted = "N/A"
                        else:
                            # For individual pathologies, just show the value
                            if pd.notna(mean_val):
                                formatted = f"{mean_val:.3f}"
                            else:
                                formatted = "N/A"
                        
                        # Wrap best value in \textbf{}
                        if col == best_col:
                            formatted = f"\\textbf{{{formatted}}}"
                        
                        combined.loc[idx, col] = formatted
                
                pivot_path = output_dir / f"table_{metric}.csv"
                combined.to_csv(pivot_path)
            else:
                pivot_path = output_dir / f"table_{metric}.csv"
                pivot_mean.to_csv(pivot_path)
    
    print(f"Pivot tables saved to: {output_dir}/table_*.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze template ablation results comparing different prompt styles"
    )
    parser.add_argument("ablation_dir", type=str, help="Path to template ablation output directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for CSV and plots (default: ablation_dir)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output) if args.output else ablation_dir
    
    if not ablation_dir.exists():
        print(f"Error: Ablation directory not found: {ablation_dir}")
        return 1
    
    print(f"Loading results from: {ablation_dir}")
    ablation_results = load_ablation_results(ablation_dir)
    
    if not ablation_results:
        print("Error: No results found!")
        return 1
    
    # Print what we found
    print(f"\nFound {len(ablation_results)} template sets:")
    for ts, pathology_results in ablation_results.items():
        desc = TEMPLATE_DESCRIPTIONS.get(ts, "")
        print(f"  - {ts}: {len(pathology_results)} pathologies {desc}")
    
    # Build comparison DataFrame
    df = build_comparison_df(ablation_results)
    
    if df.empty:
        print("Error: No metrics could be extracted!")
        return 1
    
    # Print analysis
    print_comparison_table(df)
    print_aggregate_summary(df)
    print_template_analysis(df)
    
    # Save results
    save_results(df, output_dir)
    
    # Generate plots
    if not args.no_plots:
        plot_path = output_dir / "template_ablation_plots.png"
        plot_comparisons(df, plot_path)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    exit(main())



