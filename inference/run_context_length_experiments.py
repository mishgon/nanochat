"""
Run speed experiments and generate charts comparing GPT vs UNet at different stages
with varying number of generated tokens and fixed prefill size.

Experiments:
1. GPT (with cache): tokens/sec vs number of generated tokens, fixed prefill size, 16 layers
2. UNet with all 16 layers on stage 1 (0 on stage 0): tokens/sec vs number of generated tokens
3. UNet with all 16 layers on stage 2: tokens/sec vs number of generated tokens
4. etc.

Uses --use-timing flag and parses results from the runs folder after all benchmarks complete.
"""

import subprocess
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nanochat.common import get_base_dir


def get_config_filename_suffix(n_embd, n_layer, prefill_size, max_stage, num_runs):
    """Generate a filename suffix with key configuration parameters."""
    return f"nembd_{n_embd}_nlayer_{n_layer}_prefill_{prefill_size}_maxstage_{max_stage}_runs_{num_runs}"


def run_benchmark(model_type, n_embd, n_layer=12, stage=None, prompt_len=1024, num_tokens=100, num_runs=1):
    """Run a benchmark with --use-timing flag. Results are saved to runs folder."""
    script_path = os.path.join(os.path.dirname(__file__), "benchmark_speed.py")
    # Get project root (parent of inference directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cmd = [
        sys.executable, script_path,
        "--model-type", model_type,
        "--n-embd", str(n_embd),
        "--n-layer", str(n_layer),
        "--prompt-len", str(prompt_len),
        "--num-tokens", str(num_tokens),
        "--use-timing",
    ]
    if stage is not None:
        cmd.extend(["--stage", str(stage)])
    
    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{pythonpath}"
    else:
        env["PYTHONPATH"] = project_root
    
    # Run the benchmark - results will be saved to runs folder
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        return False
    
    return True


def parse_results_from_runs_folder(base_dir, model_type, n_embd, n_layer, prompt_len, num_tokens, stage=None):
    """Parse timing results from the runs folder for a specific configuration.
    
    Returns:
        dict with 'tokens_per_sec' (float) and optionally 'measurements' (full dict)
        or None if not found
    """
    runs_dir = os.path.join(base_dir, "charts", "runs")
    if not os.path.exists(runs_dir):
        return None
    
    # Build expected config name pattern
    # Format: model={model_type}_embd={n_embd}_layer={n_layer}_prompt={prompt_len}_tokens={num_tokens}
    # For UNet: model=unet_stage={stage}_embd={n_embd}_layer={n_layer}_prompt={prompt_len}_tokens={num_tokens}
    if model_type == "gpt":
        config_name_pattern = f"model=gpt_embd={n_embd}_layer={n_layer}_prompt={prompt_len}_tokens={num_tokens}"
    else:
        config_name_pattern = f"model=unet_stage={stage}_embd={n_embd}_layer={n_layer}_prompt={prompt_len}_tokens={num_tokens}"
    
    # Look for exact match or pattern match
    config_dir = os.path.join(runs_dir, config_name_pattern)
    results_file = os.path.join(config_dir, "results.json")
    
    if not os.path.exists(results_file):
        # Try to find by pattern (in case there are extra fields like head/kv_head)
        pattern = os.path.join(runs_dir, f"{config_name_pattern}*", "results.json")
        matches = glob.glob(pattern)
        if matches:
            results_file = matches[0]  # Use first match
        else:
            return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract measurements
        measurements = data.get('measurements', {})
        if not measurements:
            return None
        
        # Calculate tokens/sec from token_times
        token_times = measurements.get('token_times', [])
        if not token_times:
            return None
        
        avg_token_time = sum(token_times) / len(token_times)
        tokens_per_sec = 1.0 / avg_token_time
        
        return {
            'tokens_per_sec': tokens_per_sec,
            'measurements': measurements,
            'config': data.get('config', {})
        }
    except Exception as e:
        print(f"Error parsing results file {results_file}: {e}")
        return None


def run_gpt_experiments(n_embd, n_layer=12, prefill_size=1024, num_tokens_list=None, num_runs=1):
    """Run GPT benchmarks for different numbers of generated tokens.
    
    Runs benchmarks and parses results immediately after each run to handle multiple runs.
    """
    if num_tokens_list is None:
        # Default number of generated tokens: powers of 2 from 16 to 1024
        num_tokens_list = [16, 32, 64, 128, 256, 512, 1024]
    
    print(f"Running GPT experiments with n_embd={n_embd}, prefill_size={prefill_size}...")
    
    base_dir = get_base_dir()
    results_by_tokens = {}  # num_tokens -> list of tokens_per_sec values
    
    # Run benchmarks and collect results
    for num_tokens in tqdm(num_tokens_list, desc="GPT experiments"):
        tokens_per_sec_list = []
        for run_idx in range(num_runs):
            # Run benchmark
            success = run_benchmark(
                'gpt',
                n_embd,
                n_layer,
                None,
                prefill_size,
                num_tokens,
                num_runs=1
            )
            if success:
                # Parse result immediately (before next run overwrites it)
                result = parse_results_from_runs_folder(
                    base_dir,
                    'gpt',
                    n_embd,
                    n_layer,
                    prefill_size,
                    num_tokens,
                    None
                )
                if result:
                    tokens_per_sec_list.append(result['tokens_per_sec'])
        
        # Aggregate results
        if tokens_per_sec_list:
            if len(tokens_per_sec_list) == 1:
                results_by_tokens[num_tokens] = tokens_per_sec_list[0]
                tqdm.write(f"  num_tokens={num_tokens}: {tokens_per_sec_list[0]:.2f} tokens/sec")
            else:
                mean = np.mean(tokens_per_sec_list)
                std = np.std(tokens_per_sec_list)
                results_by_tokens[num_tokens] = (mean, std)
                tqdm.write(f"  num_tokens={num_tokens}: {mean:.2f} ± {std:.2f} tokens/sec")
        else:
            tqdm.write(f"  num_tokens={num_tokens}: No results found")
    
    return results_by_tokens


def run_unet_experiments(stage, n_embd, n_layer=12, prefill_size=1024, num_tokens_list=None, num_runs=1):
    """Run UNet benchmarks for different numbers of generated tokens at a specific stage.
    
    Runs benchmarks and parses results immediately after each run to handle multiple runs.
    """
    if num_tokens_list is None:
        # Default number of generated tokens: powers of 2 from 16 to 1024
        num_tokens_list = [16, 32, 64, 128, 256, 512, 1024]
    
    base_dir = get_base_dir()
    results_by_tokens = {}  # num_tokens -> list of tokens_per_sec values
    
    # Run benchmarks and collect results
    for num_tokens in tqdm(num_tokens_list, desc=f"UNet stage {stage} experiments"):
        tokens_per_sec_list = []
        for run_idx in range(num_runs):
            # Run benchmark
            success = run_benchmark(
                'unet',
                n_embd,
                n_layer,
                stage,
                prefill_size,
                num_tokens,
                num_runs=1
            )
            if success:
                # Parse result immediately (before next run overwrites it)
                result = parse_results_from_runs_folder(
                    base_dir,
                    'unet',
                    n_embd,
                    n_layer,
                    prefill_size,
                    num_tokens,
                    stage
                )
                if result:
                    tokens_per_sec_list.append(result['tokens_per_sec'])
        
        # Aggregate results
        if tokens_per_sec_list:
            if len(tokens_per_sec_list) == 1:
                results_by_tokens[num_tokens] = tokens_per_sec_list[0]
                tqdm.write(f"  stage {stage}, num_tokens={num_tokens}: {tokens_per_sec_list[0]:.2f} tokens/sec")
            else:
                mean = np.mean(tokens_per_sec_list)
                std = np.std(tokens_per_sec_list)
                results_by_tokens[num_tokens] = (mean, std)
                tqdm.write(f"  stage {stage}, num_tokens={num_tokens}: {mean:.2f} ± {std:.2f} tokens/sec")
        else:
            tqdm.write(f"  stage {stage}, num_tokens={num_tokens}: No results found")
    
    return results_by_tokens


def generate_charts(gpt_results, unet_results_by_stage, output_dir, n_embd, n_layer=12, prefill_size=1024, max_stage=3, num_runs=1):
    """Generate charts comparing GPT vs UNet at different stages with varying number of generated tokens."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Helper to extract mean and std from result (handles both single values and tuples)
    def get_mean_std(value):
        if isinstance(value, tuple):
            return value[0], value[1]
        else:
            return value, 0.0
    
    # Plot GPT results
    if gpt_results:
        gpt_num_tokens = sorted(gpt_results.keys())
        gpt_speed = [get_mean_std(gpt_results[n])[0] for n in gpt_num_tokens]
        gpt_std = [get_mean_std(gpt_results[n])[1] for n in gpt_num_tokens]
        if num_runs > 1 and any(std > 0 for std in gpt_std):
            ax.errorbar(gpt_num_tokens, gpt_speed, yerr=gpt_std, fmt='o-', 
                      label=f'GPT ({n_layer} layers)', linewidth=2, markersize=8, capsize=5, capthick=2)
        else:
            ax.plot(gpt_num_tokens, gpt_speed, 'o-', label=f'GPT ({n_layer} layers)', linewidth=2, markersize=8)
    
    # Plot UNet results for each stage
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for stage, results in sorted(unet_results_by_stage.items()):
        if results:
            num_tokens_list = sorted(results.keys())
            speed = [get_mean_std(results[n])[0] for n in num_tokens_list]
            std = [get_mean_std(results[n])[1] for n in num_tokens_list]
            if num_runs > 1 and any(s > 0 for s in std):
                ax.errorbar(num_tokens_list, speed, yerr=std, fmt='s-', 
                          label=f'UNet stage {stage} ({n_layer} layers)', 
                          linewidth=2, markersize=8, color=colors[stage % len(colors)],
                          capsize=5, capthick=2)
            else:
                ax.plot(num_tokens_list, speed, 's-', label=f'UNet stage {stage} ({n_layer} layers)', 
                       linewidth=2, markersize=8, color=colors[stage % len(colors)])
    
    ax.set_xlabel('Number of Generated Tokens', fontsize=12)
    ax.set_ylabel('Tokens/sec', fontsize=12)
    title = f'Inference Speed vs Number of Generated Tokens: GPT vs UNet at Different Stages\n(n_embd={n_embd}, {n_layer} layers, prefill_size={prefill_size}, with KV cache)'
    if num_runs > 1:
        title += f', {num_runs} runs per config'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    # Generate config suffix for filename
    config_suffix = get_config_filename_suffix(n_embd, n_layer, prefill_size, max_stage, num_runs)
    
    # Save chart
    chart_path = os.path.join(output_dir, f"inference_speed_vs_generated_tokens_{config_suffix}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart to: {chart_path}")
    
    # Also save as PDF
    chart_path_pdf = os.path.join(output_dir, f"inference_speed_vs_generated_tokens_{config_suffix}.pdf")
    plt.savefig(chart_path_pdf, bbox_inches='tight')
    print(f"Saved chart to: {chart_path_pdf}")
    
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run experiments with varying number of generated tokens and generate charts")
    parser.add_argument("--n-embd", type=int, default=1024, help="Fixed hidden size (default: 1024)")
    parser.add_argument("--n-layer", type=int, default=16, help="Number of layers (default: 16)")
    parser.add_argument("--prefill-size", type=int, default=1024, help="Fixed prefill size (default: 1024)")
    parser.add_argument("--num-tokens-list", type=int, nargs='+', default=None, 
                       help="Number of generated tokens to test (default: 16, 32, 64, 128, 256, 512, 1024)")
    parser.add_argument("--max-stage", type=int, default=3, help="Maximum UNet stage to test")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for charts")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT experiments")
    parser.add_argument("--skip-unet", action="store_true", help="Skip UNet experiments")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs per configuration (for error bars)")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "charts")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Fixed hidden size (n_embd): {args.n_embd}")
    print(f"Fixed prefill size: {args.prefill_size}")
    print(f"Running {args.num_runs} runs per configuration for statistics")
    
    # Default number of generated tokens if not specified
    num_tokens_list = args.num_tokens_list if args.num_tokens_list else [16, 32, 64, 128, 256, 512, 1024]
    print(f"Number of generated tokens to test: {num_tokens_list}")
    
    # Run experiments
    gpt_results = {}
    unet_results_by_stage = {}
    
    if not args.skip_gpt:
        try:
            gpt_results = run_gpt_experiments(args.n_embd, args.n_layer, args.prefill_size, num_tokens_list, args.num_runs)
        except Exception as e:
            print(f"Error running GPT experiments: {e}")
            print("Continuing with other experiments...")
    
    if not args.skip_unet:
        for stage in tqdm(range(args.max_stage + 1), desc="UNet stages"):
            try:
                unet_results_by_stage[stage] = run_unet_experiments(
                    stage, args.n_embd, args.n_layer, args.prefill_size, num_tokens_list, args.num_runs
                )
            except Exception as e:
                print(f"Error running UNet experiments for stage {stage}: {e}")
                print(f"Skipping stage {stage}, continuing with other stages...")
                unet_results_by_stage[stage] = {}
    
    # Generate config suffix for filename
    config_suffix = get_config_filename_suffix(args.n_embd, args.n_layer, args.prefill_size, args.max_stage, args.num_runs)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f"generated_tokens_results_{config_suffix}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'gpt': gpt_results,
            'unet_by_stage': unet_results_by_stage,
            'config': {
                'n_embd': args.n_embd,
                'n_layer': args.n_layer,
                'prefill_size': args.prefill_size,
                'max_stage': args.max_stage,
                'num_tokens_list': num_tokens_list,
                'num_runs': args.num_runs,
            }
        }, f, indent=2)
    print(f"Saved results to: {results_file}")
    
    # Generate charts
    generate_charts(gpt_results, unet_results_by_stage, output_dir, args.n_embd, args.n_layer, args.prefill_size, args.max_stage, args.num_runs)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
