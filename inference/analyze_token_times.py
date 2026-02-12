"""
Analyze token generation times from nanochat benchmark runs.

Parses JSON files from nanochat_base_dir/charts/runs/, filters runs with >10k tokens,
and plots average time per token for bands of 100 tokens for each individual run.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_nanochat_base_dir():
    """Get nanochat base directory from environment variable or use default."""
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', '/home/basharin/nanochat_base_dir')
    return base_dir


def parse_json_file(filepath):
    """Parse a single JSON results file and return config and measurements."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('config', {}), data.get('measurements', {})
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None


def get_run_name(filepath):
    """Extract run name from filepath (directory name containing results.json)."""
    # Filepath is like: /path/to/charts/runs/{run_name}/results.json
    # or /path/to/charts/runs/{run_name}/results_{timestamp}.json
    dir_path = os.path.dirname(filepath)
    run_name = os.path.basename(dir_path)
    return run_name


def calculate_token_bands(token_times, band_size=100):
    """
    Calculate average time per token for bands of tokens.
    
    Args:
        token_times: List of time per token
        band_size: Size of each band (default 100)
    
    Returns:
        List of (band_center, avg_time) tuples
    """
    if not token_times:
        return []
    
    bands = []
    num_tokens = len(token_times)
    
    for start_idx in range(0, num_tokens, band_size):
        end_idx = min(start_idx + band_size, num_tokens)
        band_times = token_times[start_idx:end_idx]
        
        if band_times:
            avg_time = np.mean(band_times)
            # Use the center token index of the band
            band_center = (start_idx + end_idx - 1) / 2
            bands.append((band_center, avg_time))
    
    return bands


def main():
    """Main function to parse and plot token times for each individual run."""
    base_dir = get_nanochat_base_dir()
    runs_dir = os.path.join(base_dir, 'charts', 'runs')
    charts_dir = os.path.join(base_dir, 'charts')
    
    if not os.path.exists(runs_dir):
        print(f"Runs directory not found: {runs_dir}")
        return
    
    # Ensure charts directory exists
    os.makedirs(charts_dir, exist_ok=True)
    
    # Find all results.json files (including timestamped ones)
    pattern = os.path.join(runs_dir, '**', 'results*.json')
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    
    # Parse all files and filter by num_tokens > 10000
    runs_data = []
    for json_file in json_files:
        config, measurements = parse_json_file(json_file)
        
        if config is None or measurements is None:
            continue
        
        num_tokens = config.get('num_tokens', 0)
        if num_tokens <= 10000:
            continue
        
        token_times = measurements.get('token_times', [])
        if not token_times:
            continue
        
        run_name = get_run_name(json_file)
        runs_data.append({
            'run_name': run_name,
            'token_times': token_times,
            'num_tokens': num_tokens,
            'config': config
        })
    
    print(f"\nFound {len(runs_data)} runs with >10k tokens")
    
    if not runs_data:
        print("No data to plot!")
        return
    
    # Calculate bands for each run
    run_bands = {}
    for run in runs_data:
        run_name = run['run_name']
        token_times = run['token_times']
        bands = calculate_token_bands(token_times, band_size=100)
        run_bands[run_name] = bands
    
    # Plot the results - make figure bigger to accommodate long run names
    plt.figure(figsize=(16, 10))
    
    # Stylish color palette - modern, vibrant colors
    stylish_colors = [
        '#2E86AB',  # Blue
        '#A23B72',  # Purple
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#6A994E',  # Green
        '#BC4749',  # Dark red
        '#219EBC',  # Cyan
        '#FFB703',  # Yellow
        '#8B5CF6',  # Violet
        '#06FFA5',  # Mint
    ]
    
    # Plot each run individually
    for idx, (run_name, bands) in enumerate(run_bands.items()):
        if not bands:
            continue
        
        band_centers = [b[0] for b in bands]
        avg_times = [b[1] * 1000 for b in bands]  # Convert to milliseconds
        
        color = stylish_colors[idx % len(stylish_colors)]
        plt.plot(band_centers, avg_times, label=run_name, 
                color=color, linewidth=2.5)
    
    plt.xlabel('Token Position (band center)', fontsize=12)
    plt.ylabel('Average Time per Token (ms)', fontsize=12)
    plt.title('Token Generation Time vs Position\n(Averaged over 100-token bands, runs with >10k tokens)', fontsize=14)
    
    # Make legend bigger and place it outside the plot area to accommodate long names
    plt.legend(fontsize=9, framealpha=0.9, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Enhanced grid - more visible
    plt.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    plt.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(charts_dir, 'token_times_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = os.path.join(charts_dir, 'token_times_analysis.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_path_pdf}")


if __name__ == '__main__':
    main()
