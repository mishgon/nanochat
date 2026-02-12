"""
Benchmark inference speed for GPT and UNet models with KV cache.

Measures tokens/sec during decode phase (after prefill).
"""

import sys
import os

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sympy.sets.sets import true
import torch
import time
import argparse
import json
import os
from datetime import datetime
from collections import deque
from contextlib import nullcontext
from tqdm import tqdm
from nanochat.common import compute_init, autodetect_device_type, print0, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.unet import UNet, UNetConfig
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from inference.unet_engine import UNetEngine


def benchmark_gpt(model, device, prompt_len=1024, num_decode_tokens=100, warmup=10, num_runs=1):
    """Benchmark GPT model inference speed with KV cache using Engine.generate().
    
    Returns:
        If num_runs == 1: single tokens_per_sec value
        If num_runs > 1: (mean, std) tuple
    """
    tokenizer = get_tokenizer()
    engine = Engine(model, tokenizer)
    
    # Create a prompt and tokenize it to get approximately prompt_len tokens
    # Use a simple repeating prompt to get the desired length
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 10
    prompt_tokens = tokenizer.encode(base_prompt, prepend=tokenizer.get_bos_token_id())
    
    # Adjust to desired length by repeating or truncating
    if len(prompt_tokens) < prompt_len:
        # Repeat tokens to reach desired length
        multiplier = (prompt_len // len(prompt_tokens)) + 1
        prompt_tokens = (prompt_tokens * multiplier)[:prompt_len]
    else:
        # Truncate to desired length
        prompt_tokens = prompt_tokens[:prompt_len]
    
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else nullcontext()
    
    results = []
    
    with autocast_ctx:
        for run_idx in range(num_runs):
            # Use generate method - it handles prefill and decode internally
            # We'll time only the decode phase (after warmup)
            total_tokens = warmup + num_decode_tokens
            
            synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
            synchronize()
            
            # Generate tokens - consume warmup tokens first
            token_count = 0
            t0 = None
            
            generate_iter = engine.generate(
                prompt_tokens, 
                num_samples=1, 
                max_tokens=total_tokens,
                temperature=0.0,  # Deterministic for consistent benchmarking
                seed=42
            )
            
            # Add progress bar for generation
            if num_runs == 1:
                generate_iter = tqdm(generate_iter, total=total_tokens, desc="Generating", leave=False)
            
            for token_column, _ in generate_iter:
                token_count += 1
                # Start timing after warmup
                if token_count == warmup + 1:
                    synchronize()
                    t0 = time.time()
            
            synchronize()
            t1 = time.time()
            
            if t0 is None:
                # Not enough tokens generated
                raise RuntimeError(f"Only generated {token_count} tokens, needed at least {warmup + 1}")
            
            elapsed = t1 - t0
            tokens_per_sec = num_decode_tokens / elapsed
            results.append(tokens_per_sec)
    
    if num_runs == 1:
        return results[0]
    else:
        import numpy as np
        mean = np.mean(results)
        std = np.std(results)
        return mean, std


def benchmark_unet(model, device, prompt_len=1024, num_decode_tokens=100, warmup=10, num_runs=1):
    """Benchmark UNet model inference speed with KV cache using UNetEngine.generate().
    
    Returns:
        If num_runs == 1: single tokens_per_sec value
        If num_runs > 1: (mean, std) tuple
    """
    tokenizer = get_tokenizer()
    engine = UNetEngine(model, tokenizer)
    
    # Create a prompt and tokenize it to get approximately prompt_len tokens
    # Use a simple repeating prompt to get the desired length
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 10
    prompt_tokens = tokenizer.encode(base_prompt, prepend=tokenizer.get_bos_token_id())
    
    # Adjust to desired length by repeating or truncating
    if len(prompt_tokens) < prompt_len:
        # Repeat tokens to reach desired length
        multiplier = (prompt_len // len(prompt_tokens)) + 1
        prompt_tokens = (prompt_tokens * multiplier)[:prompt_len]
    else:
        # Truncate to desired length
        prompt_tokens = prompt_tokens[:prompt_len]
    
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else nullcontext()
    
    results = []
    
    with autocast_ctx:
        run_iter = tqdm(range(num_runs), desc="UNet benchmark runs", leave=False) if num_runs > 1 else range(num_runs)
        for run_idx in run_iter:
            # Use generate method - it handles prefill and decode internally
            # We'll time only the decode phase (after warmup)
            total_tokens = warmup + num_decode_tokens
            
            synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
            synchronize()
            
            # Generate tokens - consume warmup tokens first
            token_count = 0
            t0 = None
            
            generate_iter = engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=total_tokens,
                temperature=0.0,  # Deterministic for consistent benchmarking
                seed=42,
                ignore_assistant_end=True
            )
            
            # Add progress bar for generation
            if num_runs == 1:
                generate_iter = tqdm(generate_iter, total=total_tokens, desc="Generating", leave=False)
            
            for token_column, _ in generate_iter:
                token_count += 1
                # Start timing after warmup
                if token_count == warmup + 1:
                    synchronize()
                    t0 = time.time()
            
            synchronize()
            t1 = time.time()
            
            if t0 is None:
                # Not enough tokens generated
                raise RuntimeError(f"Only generated {token_count} tokens, needed at least {warmup + 1}")
            
            elapsed = t1 - t0
            tokens_per_sec = num_decode_tokens / elapsed
            results.append(tokens_per_sec)
    
    if num_runs == 1:
        return results[0]
    else:
        import numpy as np
        mean = np.mean(results)
        std = np.std(results)
        return mean, std


def benchmark_gpt_w_timing(model, device, prompt_len=1024, num_decode_tokens=100, warmup=10):
    """Benchmark GPT model with detailed timing using generate_w_timing.
    
    Returns:
        dict with timing measurements and config
    """
    tokenizer = get_tokenizer()
    engine = Engine(model, tokenizer)
    
    # Create a prompt and tokenize it to get approximately prompt_len tokens
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 10
    prompt_tokens = tokenizer.encode(base_prompt, prepend=tokenizer.get_bos_token_id())
    
    # Adjust to desired length by repeating or truncating
    if len(prompt_tokens) < prompt_len:
        multiplier = (prompt_len // len(prompt_tokens)) + 1
        prompt_tokens = (prompt_tokens * multiplier)[:prompt_len]
    else:
        prompt_tokens = prompt_tokens[:prompt_len]
    
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else nullcontext()
    
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    synchronize()
    
    total_tokens = warmup + num_decode_tokens
    
    # Track timing information
    prefill_time = None
    token_times = []
    last_100_times = deque(maxlen=100)  # For rolling average
    
    # Progress bar with throttled updates
    # Update every 10 tokens or every 0.1 seconds, whichever comes first
    update_interval_tokens = 1000
    update_interval_seconds = 0.5
    last_update_time = time.time()
    last_update_token = 0
    
    with autocast_ctx:
        generate_iter = engine.generate_w_timing(
            prompt_tokens,
            num_samples=1,
            max_tokens=total_tokens,
            temperature=0.0,
            seed=42,
            ignore_assistant_end=true
        )
        
        # Initialize progress bar
        pbar = tqdm(total=total_tokens, desc="Generating", unit="tok", leave=False)
        
        token_count = 0
        current_time = time.time()
        
        # Engine yields: (token_column, token_masks, current_token_time)
        for token_column, token_masks, current_token_time in generate_iter:
            token_count += 1
            current_time = time.time()
            
            # Track token times
            if current_token_time > 0:
                token_times.append(current_token_time)
                last_100_times.append(current_token_time)
            
            # Calculate average of last 100 tokens
            avg_last_100 = sum(last_100_times) / len(last_100_times) if last_100_times else 0.0
            
            # Update progress bar with throttling
            should_update = (
                token_count - last_update_token >= update_interval_tokens or
                current_time - last_update_time >= update_interval_seconds or
                token_count == total_tokens
            )
            
            if should_update:
                # Format timing info for display
                current_time_str = f"{current_token_time*1000:.2f}ms" if current_token_time > 0 else "N/A"
                avg_time_str = f"{avg_last_100*1000:.2f}ms" if avg_last_100 > 0 else "N/A"
                
                pbar.set_postfix({
                    'current': current_time_str,
                    'avg100': avg_time_str
                })
                pbar.update(token_count - last_update_token)
                last_update_token = token_count
                last_update_time = current_time
        
        pbar.close()
    
    synchronize()
    
    if len(token_times) == 0:
        raise RuntimeError("No timing information collected")
    
    # Extract only decode tokens (skip warmup)
    decode_token_times = token_times[warmup:] if len(token_times) > warmup else []
    
    # Recalculate decode_times from decode tokens only
    decode_times = []
    cumulative = 0.0
    for t in decode_token_times:
        cumulative += t
        decode_times.append(cumulative)
    
    return {
        'prefill_time': prefill_time,
        'decode_times': decode_times,
        'token_times': decode_token_times,
        'num_decode_tokens': len(decode_token_times),
        'prompt_len': prompt_len,
        'warmup': warmup
    }


def benchmark_unet_w_timing(model, device, prompt_len=1024, num_decode_tokens=100, warmup=10):
    """Benchmark UNet model with detailed timing using generate_w_timing.
    
    Returns:
        dict with timing measurements and config
    """
    tokenizer = get_tokenizer()
    engine = UNetEngine(model, tokenizer)
    
    # Create a prompt and tokenize it to get approximately prompt_len tokens
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 10
    prompt_tokens = tokenizer.encode(base_prompt, prepend=tokenizer.get_bos_token_id())
    
    # Adjust to desired length by repeating or truncating
    if len(prompt_tokens) < prompt_len:
        multiplier = (prompt_len // len(prompt_tokens)) + 1
        prompt_tokens = (prompt_tokens * multiplier)[:prompt_len]
    else:
        prompt_tokens = prompt_tokens[:prompt_len]
    
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else nullcontext()
    
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    synchronize()
    
    total_tokens = warmup + num_decode_tokens
    
    # Track timing information
    prefill_time = None
    token_times = []
    last_100_times = deque(maxlen=100)  # For rolling average
    
    # Progress bar with throttled updates
    # Update every 10 tokens or every 0.1 seconds, whichever comes first
    update_interval_tokens = 10
    update_interval_seconds = 0.1
    last_update_time = time.time()
    last_update_token = 0
    
    with autocast_ctx:
        generate_iter = engine.generate_w_timing(
            prompt_tokens,
            num_samples=1,
            max_tokens=total_tokens,
            temperature=0.0,
            seed=42,
            ignore_assistant_end=True
        )
        
        # Initialize progress bar
        pbar = tqdm(total=total_tokens, desc="Generating", unit="tok", leave=False)
        
        token_count = 0
        current_time = time.time()
        
        # Engine yields: (token_column, token_masks, current_token_time)
        for token_column, token_masks, current_token_time in generate_iter:
            token_count += 1
            current_time = time.time()
            
            # Track token times
            if current_token_time > 0:
                token_times.append(current_token_time)
                last_100_times.append(current_token_time)
            
            # Calculate average of last 100 tokens
            avg_last_100 = sum(last_100_times) / len(last_100_times) if last_100_times else 0.0
            
            # Update progress bar with throttling
            should_update = (
                token_count - last_update_token >= update_interval_tokens or
                current_time - last_update_time >= update_interval_seconds or
                token_count == total_tokens
            )
            
            if should_update:
                # Format timing info for display
                current_time_str = f"{current_token_time*1000:.2f}ms" if current_token_time > 0 else "N/A"
                avg_time_str = f"{avg_last_100*1000:.2f}ms" if avg_last_100 > 0 else "N/A"
                
                pbar.set_postfix({
                    'current': current_time_str,
                    'avg100': avg_time_str
                })
                pbar.update(token_count - last_update_token)
                last_update_token = token_count
                last_update_time = current_time
        
        pbar.close()
    
    synchronize()
    
    if len(token_times) == 0:
        raise RuntimeError("No timing information collected")
    
    # Extract only decode tokens (skip warmup)
    decode_token_times = token_times[warmup:] if len(token_times) > warmup else []
    
    # Recalculate decode_times from decode tokens only
    decode_times = []
    cumulative = 0.0
    for t in decode_token_times:
        cumulative += t
        decode_times.append(cumulative)
    
    return {
        'prefill_time': prefill_time,
        'decode_times': decode_times,
        'token_times': decode_token_times,
        'num_decode_tokens': len(decode_token_times),
        'prompt_len': prompt_len,
        'warmup': warmup
    }


def create_gpt_model(n_embd, n_layer, n_head, n_kv_head, vocab_size=32768, sequence_len=2048, device=None):
    """Create a randomly initialized GPT model."""
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_empty(device=device)
    model.init_weights()
    model.eval()
    return model


def create_unet_model(n_layer_per_stage, n_embd_per_stage, n_head_per_stage, n_kv_head_per_stage, vocab_size=32768, sequence_len=2048, device=None):
    """Create a randomly initialized UNet model."""
    # n_layer_per_stage should be a sequence of tuples (encoder_n_layer, decoder_n_layer)
    # Ensure all n_layer values are even (UNet requirement) and convert to tuples if needed
    n_layer_tuples = []
    for n_layer in n_layer_per_stage:
        if isinstance(n_layer, tuple):
            # Already a tuple, ensure both values are even
            encoder_n = max(0, (n_layer[0] // 2) * 2)
            decoder_n = max(0, (n_layer[1] // 2) * 2)
            n_layer_tuples.append((encoder_n, decoder_n))
        else:
            # Single integer, split evenly between encoder and decoder
            n = max(0, (n_layer // 2) * 2)
            n_layer_tuples.append((n // 2, n // 2))
    n_layer_per_stage = tuple(n_layer_tuples)
    
    config = UNetConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=n_layer_per_stage,
        n_head=n_head_per_stage,
        n_kv_head=n_kv_head_per_stage,
        n_embd=n_embd_per_stage,
    )
    with torch.device("meta"):
        model = UNet(config)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_empty(device=device)
    model.init_weights()
    model.eval()
    return model


def load_config_from_file(config_path):
    """Load configuration from JSON file.
    
    Supports both absolute and relative paths. Relative paths are resolved
    relative to the project root or current working directory.
    """
    # If relative path, try resolving relative to project root first, then cwd
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path_in_project = os.path.join(project_root, config_path)
        if os.path.exists(config_path_in_project):
            config_path = config_path_in_project
        elif not os.path.exists(config_path):
            # Try relative to project root configs folder
            config_path_in_configs = os.path.join(project_root, "configs", config_path)
            if os.path.exists(config_path_in_configs):
                config_path = config_path_in_configs
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def generate_config_name(args, config_dict=None, config_file_path=None):
    """Generate a config name from arguments or config dict.
    
    Args:
        args: Parsed arguments
        config_dict: Dictionary with config values
        config_file_path: Optional path to config file (used to include filename in name)
    """
    if config_dict:
        # Use config dict if provided - include config filename if available
        parts = []
        
        # Include config filename (without extension) if provided
        if config_file_path:
            config_filename = os.path.splitext(os.path.basename(config_file_path))[0]
            parts.append(f"config={config_filename}")
        
        parts.append(f"model={config_dict.get('model_type', 'unet')}")
        
        if config_dict.get('model_type') == 'gpt':
            # GPT: include all key parameters
            parts.append(f"embd={config_dict.get('n_embd', 'unknown')}")
            parts.append(f"layer={config_dict.get('n_layer', 'unknown')}")
            if config_dict.get('n_head'):
                parts.append(f"head={config_dict.get('n_head')}")
            if config_dict.get('n_kv_head'):
                parts.append(f"kv_head={config_dict.get('n_kv_head')}")
            if config_dict.get('vocab_size'):
                parts.append(f"vocab={config_dict.get('vocab_size')}")
        else:
            # UNet: include comprehensive stage info
            n_layer_per_stage = config_dict.get('n_layer_per_stage', [])
            n_embd_per_stage = config_dict.get('n_embd_per_stage', [])
            
            parts.append(f"stages={len(n_layer_per_stage)}")
            
            # Encode layer configuration: "L[enc,dec]_L[enc,dec]_..."
            if n_layer_per_stage:
                layer_strs = []
                for stage_layers in n_layer_per_stage:
                    if isinstance(stage_layers, (list, tuple)) and len(stage_layers) == 2:
                        layer_strs.append(f"{stage_layers[0]},{stage_layers[1]}")
                    else:
                        layer_strs.append(str(stage_layers))
                parts.append(f"layers={'_'.join(layer_strs)}")
            
            # Encode embedding sizes: "E[embd]_E[embd]_..."
            if n_embd_per_stage:
                embd_strs = [str(e) for e in n_embd_per_stage]
                parts.append(f"embd={'_'.join(embd_strs)}")
            
            # Include head info if available
            n_head_per_stage = config_dict.get('n_head_per_stage', [])
            if n_head_per_stage:
                head_strs = [str(h) for h in n_head_per_stage]
                parts.append(f"head={'_'.join(head_strs)}")
            
            n_kv_head_per_stage = config_dict.get('n_kv_head_per_stage', [])
            if n_kv_head_per_stage:
                kv_head_strs = [str(k) for k in n_kv_head_per_stage]
                parts.append(f"kv_head={'_'.join(kv_head_strs)}")
            
            if config_dict.get('vocab_size'):
                parts.append(f"vocab={config_dict.get('vocab_size')}")
        
        # Include benchmark parameters
        parts.append(f"prompt={config_dict.get('prompt_len', 1024)}")
        parts.append(f"tokens={config_dict.get('num_tokens', 100)}")
        if config_dict.get('warmup', 10) != 10:  # Only include if non-default
            parts.append(f"warmup={config_dict.get('warmup')}")
        
        return "_".join(parts)
    
    # Original logic for args (when not using config file)
    parts = [f"model={args.model_type}"]
    if args.model_type == "gpt":
        parts.append(f"embd={args.n_embd}")
        parts.append(f"layer={args.n_layer}")
    else:
        parts.append(f"stage={args.stage}")
        parts.append(f"embd={args.n_embd}")
        parts.append(f"layer={args.n_layer}")
    parts.append(f"prompt={args.prompt_len}")
    parts.append(f"tokens={args.num_tokens}")
    if args.n_head:
        parts.append(f"head={args.n_head}")
    if args.n_kv_head:
        parts.append(f"kv_head={args.n_kv_head}")
    return "_".join(parts)


def save_timing_results(config_name, config_dict, measurements, base_dir):
    """Save timing results to JSON file."""
    # Create directory structure: $NANOCHAT_BASE_DIR/charts/runs/{config_name}/
    charts_dir = os.path.join(base_dir, "charts", "runs")
    config_dir = os.path.join(charts_dir, config_name)
    
    # Check if directory exists and if results.json already exists
    dir_exists = os.path.exists(config_dir)
    default_json_path = os.path.join(config_dir, "results.json")
    results_json_exists = os.path.exists(default_json_path)
    
    # Create directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # If directory existed and results.json exists, use timestamped filename
    if dir_exists and results_json_exists:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_filename = f"results_{timestamp}.json"
    else:
        json_filename = "results.json"
    
    json_path = os.path.join(config_dir, json_filename)
    results = {
        "config": config_dict,
        "measurements": measurements
    }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print0(f"Saved timing results to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides other args)")
    parser.add_argument("--model-type", type=str, choices=["gpt", "unet"], required=False)
    parser.add_argument("--n-embd", type=int, help="Hidden size (GPT) or hidden size for target stage (UNet)")
    parser.add_argument("--n-layer", type=int, default=16, help="Number of layers (GPT) or total layers (UNet)")
    parser.add_argument("--stage", type=int, default=None, help="UNet stage to put all layers on (0-indexed)")
    parser.add_argument("--prompt-len", type=int, default=1024, help="Prompt length for prefill")
    parser.add_argument("--num-tokens", type=int, default=100, help="Number of decode tokens to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--sequence-len", type=int, default=None, help="Model sequence_len (for rotary embeddings). If not set, auto-calculated from prompt_len + warmup + num_tokens")
    parser.add_argument("--n-head", type=int, default=None, help="Number of attention heads (auto if not specified)")
    parser.add_argument("--n-kv-head", type=int, default=None, help="Number of KV heads (auto if not specified)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of benchmark runs (for statistics)")
    parser.add_argument("--use-timing", action="store_true", help="Use generate_w_timing for detailed per-token timing and save to JSON")
    
    args = parser.parse_args()
    
    # Load config from file if provided
    config_dict = None
    config_file_path = None
    if args.config:
        # Resolve config path (similar logic to load_config_from_file)
        config_file_path = args.config
        if not os.path.isabs(config_file_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path_in_project = os.path.abspath(os.path.join(project_root, config_file_path))
            if os.path.exists(config_path_in_project):
                config_file_path = config_path_in_project
            elif not os.path.exists(config_file_path):
                config_path_in_configs = os.path.abspath(os.path.join(project_root, "configs", config_file_path))
                if os.path.exists(config_path_in_configs):
                    config_file_path = config_path_in_configs
        # Ensure path is absolute
        config_file_path = os.path.abspath(config_file_path)
        config_dict = load_config_from_file(config_file_path)
        # Override args with config values
        if 'model_type' in config_dict:
            args.model_type = config_dict['model_type']
        if 'n_embd' in config_dict:
            args.n_embd = config_dict['n_embd']
        if 'n_layer' in config_dict:
            args.n_layer = config_dict['n_layer']
        if 'n_head' in config_dict:
            args.n_head = config_dict['n_head']
        if 'n_kv_head' in config_dict:
            args.n_kv_head = config_dict['n_kv_head']
        if 'prompt_len' in config_dict:
            args.prompt_len = config_dict['prompt_len']
        if 'num_tokens' in config_dict:
            args.num_tokens = config_dict['num_tokens']
        if 'warmup' in config_dict:
            args.warmup = config_dict['warmup']
        if 'num_runs' in config_dict:
            args.num_runs = config_dict['num_runs']
        if 'use_timing' in config_dict:
            args.use_timing = config_dict['use_timing']
        if 'sequence_len' in config_dict:
            args.sequence_len = config_dict['sequence_len']
    
    # Validate required args
    if not args.model_type:
        raise ValueError("--model-type is required (or specify --config)")
    
    # Init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    if args.model_type == "gpt":
        # GPT model
        if args.n_embd is None:
            raise ValueError("n_embd is required (specify --n-embd or include it in --config)")
        n_head = args.n_head if args.n_head else max(1, args.n_embd // 64)
        n_kv_head = args.n_kv_head if args.n_kv_head else n_head
        # Set sequence_len to accommodate prompt + warmup + decode tokens
        # Rotary embeddings are precomputed as sequence_len * 10, so we need enough headroom
        if args.sequence_len is None:
            max_seq_len = args.prompt_len + args.warmup + args.num_tokens
            # Round up to next power of 2 for efficiency, with minimum of 2048
            sequence_len = max(2048, 2 ** (max_seq_len - 1).bit_length())
        else:
            sequence_len = args.sequence_len
        print0(f"Creating GPT model: n_embd={args.n_embd}, n_layer={args.n_layer}, n_head={n_head}, n_kv_head={n_kv_head}, sequence_len={sequence_len}")
        model = create_gpt_model(args.n_embd, args.n_layer, n_head, n_kv_head, sequence_len=sequence_len, device=device)
        
        if args.use_timing:
            # Use detailed timing method
            measurements = benchmark_gpt_w_timing(model, device, args.prompt_len, args.num_tokens, args.warmup)
            saved_config_dict = {
                "model_type": "gpt",
                "n_embd": args.n_embd,
                "n_layer": args.n_layer,
                "n_head": n_head,
                "n_kv_head": n_kv_head,
                "sequence_len": sequence_len,
                "prompt_len": args.prompt_len,
                "num_tokens": args.num_tokens,
                "warmup": args.warmup,
            }
            config_name = generate_config_name(args, saved_config_dict, config_file_path)
            base_dir = get_base_dir()
            save_timing_results(config_name, saved_config_dict, measurements, base_dir)
            
            # Print summary
            avg_token_time = sum(measurements['token_times']) / len(measurements['token_times'])
            tokens_per_sec = 1.0 / avg_token_time
            print0(f"GPT prefill time: {measurements['prefill_time']:.4f}s")
            print0(f"GPT avg token time: {avg_token_time:.6f}s")
            print0(f"GPT tokens/sec: {tokens_per_sec:.2f}")
        else:
            # Use standard benchmark method
            result = benchmark_gpt(model, device, args.prompt_len, args.num_tokens, args.warmup, args.num_runs)
            if args.num_runs == 1:
                tokens_per_sec = result
                mean = result
                std = 0.0
                print0(f"GPT tokens/sec: {tokens_per_sec:.2f}")
            else:
                mean, std = result
                tokens_per_sec = mean
                print0(f"GPT tokens/sec: {mean:.2f} ± {std:.2f} (mean ± std over {args.num_runs} runs)")
        
    elif args.model_type == "unet":
        # UNet model - support arbitrary configs from file or args
        if config_dict and 'n_layer_per_stage' in config_dict:
            # Load from config file - supports arbitrary layer configurations
            n_layer_per_stage = [tuple(layer) if isinstance(layer, list) else layer for layer in config_dict['n_layer_per_stage']]
            n_embd_per_stage = config_dict['n_embd_per_stage']
            n_head_per_stage = config_dict.get('n_head_per_stage', [])
            n_kv_head_per_stage = config_dict.get('n_kv_head_per_stage', [])
            vocab_size = config_dict.get('vocab_size', 32768)
            
            n_stages = len(n_layer_per_stage)
            
            # Auto-calculate heads if not provided
            if not n_head_per_stage:
                n_head_per_stage = [max(1, embd // 64) for embd in n_embd_per_stage]
            if not n_kv_head_per_stage:
                n_kv_head_per_stage = n_head_per_stage.copy()
            
            # Ensure head_dim is consistent across stages (UNet requirement)
            head_dim_base = n_embd_per_stage[0] // n_head_per_stage[0]
            for s in range(n_stages):
                # Adjust n_head to maintain consistent head_dim
                n_head_per_stage[s] = n_embd_per_stage[s] // head_dim_base
                n_kv_head_per_stage[s] = min(n_kv_head_per_stage[s], n_head_per_stage[s])
            
            # Set sequence_len
            if args.sequence_len is None:
                if 'sequence_len' in config_dict:
                    sequence_len = config_dict['sequence_len']
                else:
                    max_seq_len = args.prompt_len + args.warmup + args.num_tokens
                    sequence_len = max(2048, 2 ** (max_seq_len - 1).bit_length())
            else:
                sequence_len = args.sequence_len
            
        else:
            # Original logic: all layers on specified stage
            if args.stage is None:
                raise ValueError("--stage is required for UNet models when not using --config")
            
            n_stages = args.stage + 1
            # n_layer_per_stage should be tuples of (encoder_n_layer, decoder_n_layer)
            # Split args.n_layer evenly between encoder and decoder
            encoder_n = (args.n_layer // 2)
            decoder_n = args.n_layer - encoder_n  # Handle odd numbers
            n_layer_per_stage = [(0, 0)] * n_stages
            n_layer_per_stage[args.stage] = (encoder_n, decoder_n)
            
            # For hidden sizes, we need to set all stages
            # Stage 0 typically has smaller hidden size, but we want to test the target stage
            # Let's use a simple scaling: each stage doubles the hidden size
            # But for this experiment, we want to test the target stage's hidden size
            n_embd_per_stage = []
            n_head_per_stage = []
            n_kv_head_per_stage = []
            
            # Calculate base hidden size for stage 0
            # If target stage > 0, stage 0 should be smaller
            base_embd = args.n_embd // (2 ** args.stage) if args.stage > 0 else args.n_embd
            # Ensure base_embd is at least 64 for reasonable model
            base_embd = max(64, base_embd)
            
            for s in range(n_stages):
                if s == 0:
                    embd = base_embd
                else:
                    # Higher stages: double the hidden size
                    embd = n_embd_per_stage[s-1] * 2
                
                n_embd_per_stage.append(embd)
                n_head = args.n_head if args.n_head else max(1, embd // 64)
                n_kv_head = args.n_kv_head if args.n_kv_head else n_head
                n_head_per_stage.append(n_head)
                n_kv_head_per_stage.append(n_kv_head)
            
            # Override target stage with specified hidden size
            n_embd_per_stage[args.stage] = args.n_embd
            n_head_per_stage[args.stage] = args.n_head if args.n_head else max(1, args.n_embd // 64)
            n_kv_head_per_stage[args.stage] = args.n_kv_head if args.n_kv_head else n_head_per_stage[args.stage]
            
            # Ensure head_dim is consistent across stages (UNet requirement)
            head_dim_base = n_embd_per_stage[0] // n_head_per_stage[0]
            for s in range(n_stages):
                # Adjust n_head to maintain consistent head_dim
                n_head_per_stage[s] = n_embd_per_stage[s] // head_dim_base
                n_kv_head_per_stage[s] = min(n_kv_head_per_stage[s], n_head_per_stage[s])
            
            vocab_size = 32768
            
            # Set sequence_len to accommodate prompt + warmup + decode tokens
            # Rotary embeddings are precomputed as sequence_len * 10, so we need enough headroom
            if args.sequence_len is None:
                max_seq_len = args.prompt_len + args.warmup + args.num_tokens
                # Round up to next power of 2 for efficiency, with minimum of 2048
                sequence_len = max(2048, 2 ** (max_seq_len - 1).bit_length())
            else:
                sequence_len = args.sequence_len
        
        print0(f"Creating UNet model: n_layer_per_stage={n_layer_per_stage}")
        print0(f"  n_embd_per_stage={n_embd_per_stage}")
        print0(f"  n_head_per_stage={n_head_per_stage}")
        print0(f"  n_kv_head_per_stage={n_kv_head_per_stage}")
        print0(f"  sequence_len={sequence_len}")
        print0(f"  vocab_size={vocab_size}")
        
        model = create_unet_model(
            tuple(n_layer_per_stage),
            tuple(n_embd_per_stage),
            tuple(n_head_per_stage),
            tuple(n_kv_head_per_stage),
            vocab_size=vocab_size,
            sequence_len=sequence_len,
            device=device
        )
        
        if args.use_timing:
            # Use detailed timing method
            measurements = benchmark_unet_w_timing(model, device, args.prompt_len, args.num_tokens, args.warmup)
            saved_config_dict = {
                "model_type": "unet",
                "n_layer_per_stage": n_layer_per_stage,
                "n_embd_per_stage": n_embd_per_stage,
                "n_head_per_stage": n_head_per_stage,
                "n_kv_head_per_stage": n_kv_head_per_stage,
                "sequence_len": sequence_len,
                "vocab_size": vocab_size,
                "prompt_len": args.prompt_len,
                "num_tokens": args.num_tokens,
                "warmup": args.warmup,
            }
            config_name = generate_config_name(args, saved_config_dict, config_file_path)
            base_dir = get_base_dir()
            save_timing_results(config_name, saved_config_dict, measurements, base_dir)
            
            # Print summary
            avg_token_time = sum(measurements['token_times']) / len(measurements['token_times'])
            tokens_per_sec = 1.0 / avg_token_time
            print0(f"UNet prefill time: {measurements['prefill_time']:.4f}s")
            print0(f"UNet avg token time: {avg_token_time:.6f}s")
            print0(f"UNet tokens/sec: {tokens_per_sec:.2f}")
        else:
            # Use standard benchmark method
            result = benchmark_unet(model, device, args.prompt_len, args.num_tokens, args.warmup, args.num_runs)
            if args.num_runs == 1:
                tokens_per_sec = result
                mean = result
                std = 0.0
                print0(f"UNet tokens/sec: {tokens_per_sec:.2f}")
            else:
                mean, std = result
                tokens_per_sec = mean
                print0(f"UNet tokens/sec: {mean:.2f} ± {std:.2f} (mean ± std over {args.num_runs} runs)")
        
    # Output result in a parseable format (only for non-timing mode)
    if not args.use_timing:
        if args.num_runs == 1:
            print0(f"RESULT: {tokens_per_sec:.4f}")
        else:
            print0(f"RESULT: {mean:.4f} {std:.4f}")


if __name__ == "__main__":
    main()
