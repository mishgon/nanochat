"""
Analyze token generation times from nanochat benchmark runs.

Parses JSON files from nanochat_base_dir/charts/runs/, filters runs with >10k tokens,
groups them by model type (and stage for UNet), and plots average time per token
for bands of 100 tokens.
"""

import json
import os
import glob
from collections import defaultdict
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


def get_group_key(config):
    """Get a grouping key from config (model_type and stage for UNet)."""
    model_type = config.get('model_type', 'unknown')
    if model_type == 'unet':
        stage = config.get('stage', 'unknown')
        return f"unet_stage_{stage}"
    else:
        return model_type


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
    """Main function to parse, group, and plot token times."""
    base_dir = get_nanochat_base_dir()
    runs_dir = os.path.join(base_dir, 'charts', 'runs')
    charts_dir = os.path.join(base_dir, 'charts')
    
    if not os.path.exists(runs_dir):
        print(f"Runs directory not found: {runs_dir}")
        return
    
    # Ensure charts directory exists
    os.makedirs(charts_dir, exist_ok=True)
    
    # Find all results.json files
    pattern = os.path.join(runs_dir, '**', 'results.json')
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    
    # Group data by model type and stage
    grouped_data = defaultdict(list)
    
    # Parse all files and filter by num_tokens > 10000
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
        
        group_key = get_group_key(config)
        grouped_data[group_key].append({
            'token_times': token_times,
            'num_tokens': num_tokens,
            'config': config
        })
    
    print(f"\nFound {sum(len(runs) for runs in grouped_data.values())} runs with >10k tokens")
    print(f"Grouped into {len(grouped_data)} groups:")
    for key, runs in grouped_data.items():
        print(f"  {key}: {len(runs)} runs")
    
    if not grouped_data:
        print("No data to plot!")
        return
    
    # Calculate average bands for each group
    group_bands = {}
    for group_key, runs in grouped_data.items():
        all_bands = []
        
        for run in runs:
            token_times = run['token_times']
            bands = calculate_token_bands(token_times, band_size=100)
            all_bands.append(bands)
        
        # Average across all runs for each band position
        if all_bands:
            # Find the maximum number of bands
            max_bands = max(len(bands) for bands in all_bands)
            
            # For each band position, average across runs
            averaged_bands = []
            for band_idx in range(max_bands):
                band_times = []
                for bands in all_bands:
                    if band_idx < len(bands):
                        band_times.append(bands[band_idx][1])  # Get the avg_time
                
                if band_times:
                    # Use the band center from the first run that has this band
                    band_center = None
                    for bands in all_bands:
                        if band_idx < len(bands):
                            band_center = bands[band_idx][0]
                            break
                    
                    if band_center is not None:
                        avg_time = np.mean(band_times)
                        averaged_bands.append((band_center, avg_time))
            
            group_bands[group_key] = averaged_bands
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
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
    
    # Use stylish colors, cycling if needed
    for idx, (group_key, bands) in enumerate(group_bands.items()):
        if not bands:
            continue
        
        band_centers = [b[0] for b in bands]
        avg_times = [b[1] * 1000 for b in bands]  # Convert to milliseconds
        
        color = stylish_colors[idx % len(stylish_colors)]
        plt.plot(band_centers, avg_times, label=group_key, 
                color=color, linewidth=2.5)
    
    plt.xlabel('Token Position (band center)', fontsize=12)
    plt.ylabel('Average Time per Token (ms)', fontsize=12)
    plt.title('Token Generation Time vs Position\n(Averaged over 100-token bands, runs with >10k tokens)', fontsize=14)
    plt.legend(fontsize=10, framealpha=0.9)
    
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
