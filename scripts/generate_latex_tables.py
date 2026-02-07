#!/usr/bin/env python3
"""
Script to generate LaTeX-ready CSV tables from sweep results.

Usage:
    python generate_latex_tables.py --frd <frd_csv> --sweep <sweep_csv> [--output-dir <dir>]

Example:
    python generate_latex_tables.py \
        --frd outputs/sweep_20260107_094119/frd_results.csv \
        --sweep outputs/sweep_20260107_094119/sweep_comparison.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def format_mean_std(mean_val, std_val, decimals=2):
    """Format mean ± std string."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return '-'
    return f'{mean_val:.{decimals}f} ± {std_val:.{decimals}f}'


def get_methods(df: pd.DataFrame) -> list:
    """Get sorted list of methods from DataFrame."""
    return sorted(df['method'].unique())


# =============================================================================
# FRD Tables
# =============================================================================

def generate_frd_reconstruction_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate FRD table with methods as rows and reconstruction comparisons as columns.
    Values: mean ± std across pathologies
    """
    comparisons = [
        'original_vs_reconstructed',
        'original_vs_increase',
        'original_vs_decrease',
        'reconstructed_vs_increase', 
        'reconstructed_vs_decrease'
    ]
    
    methods = get_methods(df)
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        for comp in comparisons:
            comp_df = method_df[method_df['comparison'] == comp]
            if len(comp_df) > 0:
                mean_val = comp_df['frd_score'].mean()
                std_val = comp_df['frd_score'].std()
                row[f'{comp}_mean'] = mean_val
                row[f'{comp}_std'] = std_val
                row[comp] = format_mean_std(mean_val, std_val)
            else:
                row[f'{comp}_mean'] = np.nan
                row[f'{comp}_std'] = np.nan
                row[comp] = '-'
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# LPIPS/SSIM Tables
# =============================================================================

def generate_lpips_ssim_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate LPIPS and SSIM table with methods as rows.
    Averaged across pathologies (and classifiers).
    """
    methods = get_methods(df)
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        # LPIPS (lower is better - more similar)
        lpips_mean = method_df['lpips'].mean()
        lpips_std = method_df['lpips'].std()
        row['lpips_mean'] = lpips_mean
        row['lpips_std'] = lpips_std
        row['lpips'] = format_mean_std(lpips_mean, lpips_std, decimals=3)
        
        # SSIM (higher is better - more similar)
        ssim_mean = method_df['ssim'].mean()
        ssim_std = method_df['ssim'].std()
        row['ssim_mean'] = ssim_mean
        row['ssim_std'] = ssim_std
        row['ssim'] = format_mean_std(ssim_mean, ssim_std, decimals=3)
        
        results.append(row)
    
    return pd.DataFrame(results)


def generate_lpips_ssim_by_classifier_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate LPIPS and SSIM table with methods as rows, split by classifier.
    Averaged across pathologies.
    """
    methods = get_methods(df)
    classifiers = df['classifier'].unique()
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        for clf in classifiers:
            clf_short = clf.split('-')[0] if '-' in clf else clf  # Shorten name
            clf_df = method_df[method_df['classifier'] == clf]
            
            if len(clf_df) > 0:
                # LPIPS
                lpips_mean = clf_df['lpips'].mean()
                lpips_std = clf_df['lpips'].std()
                row[f'lpips_{clf_short}'] = format_mean_std(lpips_mean, lpips_std, decimals=3)
                
                # SSIM
                ssim_mean = clf_df['ssim'].mean()
                ssim_std = clf_df['ssim'].std()
                row[f'ssim_{clf_short}'] = format_mean_std(ssim_mean, ssim_std, decimals=3)
            else:
                row[f'lpips_{clf_short}'] = '-'
                row[f'ssim_{clf_short}'] = '-'
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# Classifier Metric Tables
# =============================================================================

def generate_classifier_metrics_table(df: pd.DataFrame, classifier: str = None) -> pd.DataFrame:
    """
    Generate classifier metrics table with methods as rows.
    Columns: intended_flip_rate, non_target_preservation, selectivity
    Averaged across pathologies.
    
    If classifier is specified, filter by that classifier. Otherwise average across all.
    """
    if classifier:
        df = df[df['classifier'] == classifier]
    
    methods = get_methods(df)
    metrics = ['intended_flip_rate', 'non_target_preservation', 'selectivity']
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        for metric in metrics:
            if metric in method_df.columns and len(method_df) > 0:
                mean_val = method_df[metric].mean()
                std_val = method_df[metric].std()
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                row[metric] = format_mean_std(mean_val, std_val)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[metric] = '-'
        
        results.append(row)
    
    return pd.DataFrame(results)


def generate_classifier_metrics_by_classifier_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate classifier metrics table for each classifier separately.
    """
    methods = get_methods(df)
    metrics = ['intended_flip_rate', 'non_target_preservation', 'selectivity']
    classifiers = df['classifier'].unique()
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        for clf in classifiers:
            clf_short = clf.split('-')[0] if '-' in clf else clf
            clf_df = method_df[method_df['classifier'] == clf]
            
            for metric in metrics:
                if metric in clf_df.columns and len(clf_df) > 0:
                    mean_val = clf_df[metric].mean()
                    std_val = clf_df[metric].std()
                    row[f'{metric}_{clf_short}'] = format_mean_std(mean_val, std_val)
                else:
                    row[f'{metric}_{clf_short}'] = '-'
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# Confidence Delta Tables (per pathology)
# =============================================================================

def generate_confidence_delta_table(
    df: pd.DataFrame, 
    direction: str, 
    classifier: str = None
) -> pd.DataFrame:
    """
    Generate confidence delta table with methods as rows and pathologies as columns.
    
    Args:
        df: DataFrame with sweep results
        direction: 'increase' or 'decrease'
        classifier: Optional classifier filter
    
    Returns:
        DataFrame with confidence deltas per pathology
    """
    if classifier:
        df = df[df['classifier'] == classifier]
    
    methods = get_methods(df)
    pathologies = df['pathology'].unique()
    
    delta_col = f'{direction}_confidence_delta'
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        for pathology in pathologies:
            path_df = method_df[method_df['pathology'] == pathology]
            
            if len(path_df) > 0 and delta_col in path_df.columns:
                # If multiple classifiers, average across them
                mean_val = path_df[delta_col].mean()
                row[pathology] = f'{mean_val:.3f}'
                row[f'{pathology}_raw'] = mean_val
            else:
                row[pathology] = '-'
                row[f'{pathology}_raw'] = np.nan
        
        # Add mean across pathologies
        raw_cols = [row.get(f'{p}_raw', np.nan) for p in pathologies]
        valid_vals = [v for v in raw_cols if not pd.isna(v)]
        if valid_vals:
            row['mean'] = f'{np.mean(valid_vals):.3f}'
            row['mean_raw'] = np.mean(valid_vals)
        else:
            row['mean'] = '-'
            row['mean_raw'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def generate_confidence_delta_combined_table(
    df: pd.DataFrame, 
    classifier: str = None
) -> pd.DataFrame:
    """
    Generate combined confidence delta table showing both increase and decrease.
    Rows: methods
    Columns: pathologies with increase/decrease sub-columns
    """
    if classifier:
        df = df[df['classifier'] == classifier]
    
    methods = get_methods(df)
    pathologies = sorted(df['pathology'].unique())
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        for pathology in pathologies:
            path_df = method_df[method_df['pathology'] == pathology]
            
            if len(path_df) > 0:
                # Increase delta
                inc_val = path_df['increase_confidence_delta'].mean()
                row[f'{pathology}_inc'] = f'{inc_val:.3f}'
                
                # Decrease delta  
                dec_val = path_df['decrease_confidence_delta'].mean()
                row[f'{pathology}_dec'] = f'{dec_val:.3f}'
            else:
                row[f'{pathology}_inc'] = '-'
                row[f'{pathology}_dec'] = '-'
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# Flip Rate by Direction Tables
# =============================================================================

def generate_flip_rate_by_direction_table(
    df: pd.DataFrame,
    classifier: str = None
) -> pd.DataFrame:
    """
    Generate flip rate table split by increase/decrease direction.
    Rows: methods
    Columns: increase_intended_flip_rate, decrease_intended_flip_rate (averaged across pathologies)
    """
    if classifier:
        df = df[df['classifier'] == classifier]
    
    methods = get_methods(df)
    
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        row = {'method': method}
        
        # Increase direction
        inc_flip = method_df['increase_intended_flip_rate'].mean()
        inc_std = method_df['increase_intended_flip_rate'].std()
        row['increase_flip'] = format_mean_std(inc_flip, inc_std)
        row['increase_flip_mean'] = inc_flip
        
        # Decrease direction
        dec_flip = method_df['decrease_intended_flip_rate'].mean()
        dec_std = method_df['decrease_intended_flip_rate'].std()
        row['decrease_flip'] = format_mean_std(dec_flip, dec_std)
        row['decrease_flip_mean'] = dec_flip
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# Save and Print Utilities
# =============================================================================

def save_latex_csv(df: pd.DataFrame, output_path: Path, columns: list = None):
    """Save a subset of columns as a clean CSV for LaTeX."""
    if columns:
        df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def print_table(df: pd.DataFrame, title: str, columns: list = None):
    """Print table with title for preview."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    if columns:
        print(df[columns].to_string(index=False))
    else:
        print(df.to_string(index=False))
    print()


# =============================================================================
# Main
# =============================================================================

def resolve_csv_path(path_str: str, default_filename: str) -> Path:
    """Resolve a path to a CSV file, auto-detecting if directory is passed."""
    path = Path(path_str)
    if path.is_dir():
        csv_path = path / default_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected {default_filename} in directory {path}")
        return csv_path
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX-ready CSV tables from sweep results'
    )
    parser.add_argument('--frd', type=str, help='Path to FRD results CSV (or sweep directory)')
    parser.add_argument('--sweep', type=str, help='Path to sweep comparison CSV (or sweep directory)')
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: same as first input file)'
    )
    args = parser.parse_args()
    
    if not args.frd and not args.sweep:
        parser.error("At least one of --frd or --sweep must be provided")
    
    # Resolve paths (handle directories)
    frd_path = resolve_csv_path(args.frd, 'frd_results.csv') if args.frd else None
    sweep_path = resolve_csv_path(args.sweep, 'sweep_comparison.csv') if args.sweep else None
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif frd_path:
        output_dir = frd_path.parent
    else:
        output_dir = sweep_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Process FRD results
    # ==========================================================================
    if frd_path:
        frd_df = pd.read_csv(frd_path)
        print(f"Loaded {len(frd_df)} rows from FRD: {frd_path}")
        
        # Table 1: FRD Reconstruction Comparison
        frd_table = generate_frd_reconstruction_table(frd_df)
        
        formatted_cols = [
            'method',
            'original_vs_reconstructed',
            'original_vs_increase',
            'original_vs_decrease',
            'reconstructed_vs_increase',
            'reconstructed_vs_decrease'
        ]
        save_latex_csv(frd_table, output_dir / 'table_frd.csv', formatted_cols)
        print_table(frd_table, "FRD Table (mean ± std across pathologies)", formatted_cols)
    
    # ==========================================================================
    # Process Sweep comparison results
    # ==========================================================================
    if sweep_path:
        sweep_df = pd.read_csv(sweep_path)
        print(f"Loaded {len(sweep_df)} rows from sweep: {sweep_path}")
        
        # Table 2: LPIPS and SSIM (averaged across classifiers)
        lpips_ssim_table = generate_lpips_ssim_table(sweep_df)
        save_latex_csv(
            lpips_ssim_table, 
            output_dir / 'table_lpips_ssim.csv',
            ['method', 'lpips', 'ssim']
        )
        print_table(
            lpips_ssim_table, 
            "LPIPS & SSIM Table (mean ± std across pathologies & classifiers)",
            ['method', 'lpips', 'ssim']
        )
        
        # Table 3: Classifier Metrics (per classifier)
        classifiers = sweep_df['classifier'].unique()
        for clf in classifiers:
            clf_short = clf.split('-')[0] if '-' in clf else clf
            clf_table = generate_classifier_metrics_table(sweep_df, classifier=clf)
            
            formatted_cols = ['method', 'intended_flip_rate', 'non_target_preservation', 'selectivity']
            save_latex_csv(
                clf_table,
                output_dir / f'table_classifier_metrics_{clf_short}.csv',
                formatted_cols
            )
            print_table(
                clf_table,
                f"Classifier Metrics - {clf} (mean ± std across pathologies)",
                formatted_cols
            )
        
        # Table 4: Confidence Deltas per Pathology (per classifier, per direction)
        pathologies = sorted(sweep_df['pathology'].unique())
        
        for clf in classifiers:
            clf_short = clf.split('-')[0] if '-' in clf else clf
            
            # Increase direction
            inc_table = generate_confidence_delta_table(sweep_df, 'increase', classifier=clf)
            inc_cols = ['method'] + pathologies + ['mean']
            save_latex_csv(
                inc_table,
                output_dir / f'table_confidence_delta_increase_{clf_short}.csv',
                inc_cols
            )
            print_table(
                inc_table,
                f"Confidence Delta (Increase) - {clf}",
                inc_cols
            )
            
            # Decrease direction
            dec_table = generate_confidence_delta_table(sweep_df, 'decrease', classifier=clf)
            dec_cols = ['method'] + pathologies + ['mean']
            save_latex_csv(
                dec_table,
                output_dir / f'table_confidence_delta_decrease_{clf_short}.csv',
                dec_cols
            )
            print_table(
                dec_table,
                f"Confidence Delta (Decrease) - {clf}",
                dec_cols
            )
        
        # Table 5: Flip Rate by Direction (per classifier)
        for clf in classifiers:
            clf_short = clf.split('-')[0] if '-' in clf else clf
            
            flip_dir_table = generate_flip_rate_by_direction_table(sweep_df, classifier=clf)
            flip_cols = ['method', 'increase_flip', 'decrease_flip']
            save_latex_csv(
                flip_dir_table,
                output_dir / f'table_flip_rate_by_direction_{clf_short}.csv',
                flip_cols
            )
            print_table(
                flip_dir_table,
                f"Flip Rate by Direction - {clf}",
                flip_cols
            )


if __name__ == '__main__':
    main()
