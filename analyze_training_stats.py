"""
Analyze training statistics from experiment results or metrics files.

Usage:
    python analyze_training_stats.py --results-file lora_rank_results/experiment_results.json
    python analyze_training_stats.py --metrics-file /path/to/metrics.jsonl
"""

import argparse
import json
from pathlib import Path


def analyze_experiment_results(results_file: str):
    """Analyze experiment_results.json format (LoRA rank experiments)."""
    with open(results_file) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"Training Statistics from: {results_file}")
    print(f"{'='*70}\n")

    for exp_name, exp_data in data.items():
        metrics = exp_data.get('training', {}).get('metrics_history', [])

        if not metrics:
            print(f"{exp_name}: No training metrics found")
            continue

        # Calculate statistics
        num_steps = len(metrics)
        total_tokens = sum(m.get('num_tokens', 0) for m in metrics)
        total_sequences = sum(m.get('num_sequences', 0) for m in metrics)

        # Infer batch size from first step
        first_step_tokens = metrics[0].get('num_tokens', 0)

        print(f"{exp_name}:")
        print(f"  Training steps: {num_steps}")
        print(f"  Total tokens processed: {total_tokens:,}")

        if total_sequences > 0:
            print(f"  Total sequences processed: {total_sequences:,}")
            print(f"  Avg tokens per sequence: {total_tokens / total_sequences:.1f}")

        # Calculate from step 0 if available
        if len(metrics) > 0:
            # Estimate batch size by looking at multiple steps
            token_counts = [m.get('num_tokens', 0) for m in metrics[:10]]
            avg_tokens_per_step = sum(token_counts) / len(token_counts)
            print(f"  Avg tokens per step: {avg_tokens_per_step:.1f}")

        print()


def analyze_metrics_jsonl(metrics_file: str):
    """Analyze metrics.jsonl format (standard Tinker training)."""
    with open(metrics_file) as f:
        metrics = [json.loads(line) for line in f]

    print(f"\n{'='*70}")
    print(f"Training Statistics from: {metrics_file}")
    print(f"{'='*70}\n")

    num_steps = len(metrics)
    total_tokens = sum(m.get('num_tokens', 0) for m in metrics)
    total_sequences = sum(m.get('num_sequences', 0) for m in metrics)
    total_loss_tokens = sum(m.get('num_loss_tokens', 0) for m in metrics)

    print(f"Training steps: {num_steps}")
    print(f"Total tokens processed: {total_tokens:,}")

    if total_sequences > 0:
        print(f"Total sequences (examples): {total_sequences:,}")
        print(f"Avg tokens per example: {total_tokens / total_sequences:.1f}")

    if total_loss_tokens > 0:
        print(f"Total loss tokens: {total_loss_tokens:,}")
        print(f"Loss token ratio: {total_loss_tokens / total_tokens:.2%}")

    # Get unique batch sizes
    batch_sizes = set(m.get('num_sequences', 0) for m in metrics if 'num_sequences' in m)
    if batch_sizes:
        print(f"Batch sizes used: {sorted(batch_sizes)}")

    # Get epoch info if available
    epochs = set(m.get('epoch', -1) for m in metrics if 'epoch' in m)
    if epochs and -1 not in epochs:
        print(f"Epochs completed: {len(epochs)}")
        print(f"Steps per epoch: ~{num_steps // len(epochs)}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze training statistics from experiment results or metrics files"
    )
    parser.add_argument(
        "--results-file",
        help="Path to experiment_results.json file",
    )
    parser.add_argument(
        "--metrics-file",
        help="Path to metrics.jsonl file",
    )

    args = parser.parse_args()

    if args.results_file:
        analyze_experiment_results(args.results_file)
    elif args.metrics_file:
        analyze_metrics_jsonl(args.metrics_file)
    else:
        print("ERROR: Must provide either --results-file or --metrics-file")
        parser.print_help()
