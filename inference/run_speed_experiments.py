"""
Run speed experiments and generate charts comparing GPT vs UNet at different stages.

Experiments:
1. GPT (with cache): tokens/sec vs hidden size, 16 layers, context 1024
2. UNet with all 16 layers on stage 1 (0 on stage 0): tokens/sec vs hidden size on stage 1
3. UNet with all 16 layers on stage 2: tokens/sec vs hidden size on stage 2
4. etc.
"""

import subprocess
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nanochat.common import get_base_dir


def run_benchmark(model_type, n_embd, n_layer=16, stage=None, prompt_len=1024, num_tokens=100, num_runs=1):
    """Run a benchmark and return tokens/sec (mean, std) if num_runs > 1, else single value."""
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
        "--num-runs", str(num_runs),
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
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        return None
    
    # Parse output for "RESULT: X.XXXX" or "RESULT: X.XXXX Y.YYYY"
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT:"):
            parts = line.split(":")[1].strip().split()
            if len(parts) == 1:
                return float(parts[0])
            elif len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
    
    return None


def run_gpt_experiments(n_layer=16, prompt_len=1024, num_tokens=100, num_runs=1):
    """Run GPT benchmarks for different hidden sizes."""
    print("Running GPT experiments...")
    hidden_sizes = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    results = {}
    
    for n_embd in tqdm(hidden_sizes, desc="GPT experiments"):
        result = run_benchmark("gpt", n_embd, n_layer, None, prompt_len, num_tokens, num_runs)
        if result is not None:
            if num_runs == 1:
                results[n_embd] = result
                tqdm.write(f"  n_embd={n_embd}: {result:.2f} tokens/sec")
            else:
                mean, std = result
                results[n_embd] = (mean, std)
                tqdm.write(f"  n_embd={n_embd}: {mean:.2f} ± {std:.2f} tokens/sec")
        else:
            tqdm.write(f"  n_embd={n_embd}: Failed")
    
    return results


def run_unet_experiments(stage, n_layer=16, prompt_len=1024, num_tokens=100, num_runs=1):
    """Run UNet benchmarks for different hidden sizes at a specific stage."""
    # Hidden sizes for the target stage (same range as GPT)
    hidden_sizes = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    results = {}
    
    for n_embd in tqdm(hidden_sizes, desc=f"UNet stage {stage} experiments"):
        result = run_benchmark("unet", n_embd, n_layer, stage, prompt_len, num_tokens, num_runs)
        if result is not None:
            if num_runs == 1:
                results[n_embd] = result
                tqdm.write(f"  stage {stage}, n_embd={n_embd}: {result:.2f} tokens/sec")
            else:
                mean, std = result
                results[n_embd] = (mean, std)
                tqdm.write(f"  stage {stage}, n_embd={n_embd}: {mean:.2f} ± {std:.2f} tokens/sec")
        else:
            tqdm.write(f"  stage {stage}, n_embd={n_embd}: Failed")
    
    return results


def generate_charts(gpt_results, unet_results_by_stage, output_dir, num_runs=1):
    """Generate charts comparing GPT vs UNet at different stages with error bars."""
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
        gpt_hidden = sorted(gpt_results.keys())
        gpt_speed = [get_mean_std(gpt_results[h])[0] for h in gpt_hidden]
        gpt_std = [get_mean_std(gpt_results[h])[1] for h in gpt_hidden]
        if num_runs > 1 and any(std > 0 for std in gpt_std):
            ax.errorbar(gpt_hidden, gpt_speed, yerr=gpt_std, fmt='o-', 
                      label='GPT (16 layers)', linewidth=2, markersize=8, capsize=5, capthick=2)
        else:
            ax.plot(gpt_hidden, gpt_speed, 'o-', label='GPT (16 layers)', linewidth=2, markersize=8)
    
    # Plot UNet results for each stage
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for stage, results in sorted(unet_results_by_stage.items()):
        if results:
            hidden = sorted(results.keys())
            speed = [get_mean_std(results[h])[0] for h in hidden]
            std = [get_mean_std(results[h])[1] for h in hidden]
            if num_runs > 1 and any(s > 0 for s in std):
                ax.errorbar(hidden, speed, yerr=std, fmt='s-', 
                          label=f'UNet stage {stage} (16 layers)', 
                          linewidth=2, markersize=8, color=colors[stage % len(colors)],
                          capsize=5, capthick=2)
            else:
                ax.plot(hidden, speed, 's-', label=f'UNet stage {stage} (16 layers)', 
                       linewidth=2, markersize=8, color=colors[stage % len(colors)])
    
    ax.set_xlabel('Hidden Size', fontsize=12)
    ax.set_ylabel('Tokens/sec', fontsize=12)
    title = 'Inference Speed: GPT vs UNet at Different Stages\n(16 layers, prompt_len=1024, with KV cache)'
    if num_runs > 1:
        title += f', {num_runs} runs per config'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, "inference_speed_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart to: {chart_path}")
    
    # Also save as PDF
    chart_path_pdf = os.path.join(output_dir, "inference_speed_comparison.pdf")
    plt.savefig(chart_path_pdf, bbox_inches='tight')
    print(f"Saved chart to: {chart_path_pdf}")
    
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run speed experiments and generate charts")
    parser.add_argument("--n-layer", type=int, default=16, help="Number of layers")
    parser.add_argument("--prompt-len", type=int, default=1024, help="Prompt length")
    parser.add_argument("--num-tokens", type=int, default=100, help="Number of decode tokens to benchmark")
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
    print(f"Running {args.num_runs} runs per configuration for statistics")
    
    # Run experiments
    gpt_results = {}
    unet_results_by_stage = {}
    
    if not args.skip_gpt:
        gpt_results = run_gpt_experiments(args.n_layer, args.prompt_len, args.num_tokens, args.num_runs)
    
    if not args.skip_unet:
        for stage in tqdm(range(args.max_stage + 1), desc="UNet stages"):
            unet_results_by_stage[stage] = run_unet_experiments(stage, args.n_layer, args.prompt_len, args.num_tokens, args.num_runs)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "speed_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'gpt': gpt_results,
            'unet_by_stage': unet_results_by_stage,
            'config': {
                'n_layer': args.n_layer,
                'prompt_len': args.prompt_len,
                'num_tokens': args.num_tokens,
                'num_runs': args.num_runs,
            }
        }, f, indent=2)
    print(f"Saved results to: {results_file}")
    
    # Generate charts
    generate_charts(gpt_results, unet_results_by_stage, output_dir, args.num_runs)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
