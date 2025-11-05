"""
Visualization Tool for LoRA Rank Sweep Results

Generates publication-quality plots analyzing the relationship between:
- LoRA rank and model quality (loss, perplexity)
- LoRA rank and efficiency (training speed, inference speed, memory)
- Learning rate and performance for each rank

Usage:
    python -m tinker_cookbook.recipes.visualize_sweep_results --input lora_rank_results/

Generates:
    - loss_vs_rank.png: Train/val loss for each rank across all LRs
    - perplexity_heatmap.png: Heatmap of val perplexity (Rank x LR)
    - efficiency_analysis.png: Tokens/sec and memory usage
    - best_hyperparameters.txt: Recommended rank/LR combination
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import chz
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@chz.chz
class VisualizationConfig:
    """Configuration for generating visualizations"""

    # Input directory containing experiment_progress.json
    input_dir: str = "./lora_rank_results"

    # Output directory for plots
    output_dir: str = "./lora_rank_results/visualizations"

    # Figure size and DPI
    figure_width: float = 12.0
    figure_height: float = 8.0
    dpi: int = 300

    # Color scheme
    colormap: str = "viridis"


def load_results(input_dir: Path) -> Dict:
    """
    Load experiment results from experiment_progress.json.

    Args:
        input_dir: Directory containing results

    Returns:
        Dictionary of results keyed by experiment name
    """
    results_file = input_dir / "experiment_progress.json"

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            f"Run lora_rank_experiment.py or run_sweep_parallel.py first."
        )

    with open(results_file, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} experiment results from {results_file}")
    return results


def extract_data_by_rank(results: Dict) -> Dict[int, List[Dict]]:
    """
    Organize results by rank for easier plotting.

    Args:
        results: Raw results dictionary

    Returns:
        Dict mapping rank -> list of experiment results for that rank
    """
    by_rank = {}

    for exp_key, result in results.items():
        rank = result["rank"]
        if rank not in by_rank:
            by_rank[rank] = []
        by_rank[rank].append(result)

    # Sort each rank's results by learning rate
    for rank in by_rank:
        by_rank[rank].sort(key=lambda x: x["learning_rate"])

    return by_rank


def plot_loss_vs_rank(results: Dict, config: VisualizationConfig):
    """
    Plot training and validation loss vs rank for all learning rates.

    Args:
        results: Experiment results
        config: Visualization configuration
    """
    logger.info("Generating loss vs rank plot...")

    by_rank = extract_data_by_rank(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.figure_width, config.figure_height))

    ranks = sorted(by_rank.keys())

    # Get unique learning rates
    all_lrs = set()
    for rank_results in by_rank.values():
        for result in rank_results:
            all_lrs.add(result["learning_rate"])
    lrs = sorted(all_lrs)

    # Plot for each learning rate
    for lr in lrs:
        train_losses = []
        val_losses = []
        ranks_with_data = []

        for rank in ranks:
            # Find result for this (rank, lr) combination
            matching = [r for r in by_rank[rank] if r["learning_rate"] == lr]
            if matching:
                result = matching[0]
                train_losses.append(result["training"]["final_loss"])
                val_losses.append(result["evaluation"]["val_loss"])
                ranks_with_data.append(rank)

        if ranks_with_data:
            ax1.plot(ranks_with_data, train_losses, marker='o', label=f"LR={lr:.1e}")
            ax2.plot(ranks_with_data, val_losses, marker='o', label=f"LR={lr:.1e}")

    ax1.set_xlabel("LoRA Rank", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss vs LoRA Rank", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("LoRA Rank", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss vs LoRA Rank", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(config.output_dir) / "loss_vs_rank.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_perplexity_heatmap(results: Dict, config: VisualizationConfig):
    """
    Plot heatmap of validation perplexity across rank x learning rate.

    Args:
        results: Experiment results
        config: Visualization configuration
    """
    logger.info("Generating perplexity heatmap...")

    # Extract unique ranks and LRs
    ranks = sorted(set(r["rank"] for r in results.values()))
    lrs = sorted(set(r["learning_rate"] for r in results.values()))

    # Create matrix
    perplexity_matrix = np.full((len(lrs), len(ranks)), np.nan)

    for result in results.values():
        rank = result["rank"]
        lr = result["learning_rate"]
        perplexity = result["evaluation"]["perplexity"]

        rank_idx = ranks.index(rank)
        lr_idx = lrs.index(lr)
        perplexity_matrix[lr_idx, rank_idx] = perplexity

    # Plot
    fig, ax = plt.subplots(figsize=(config.figure_width, config.figure_height))

    im = ax.imshow(perplexity_matrix, cmap=config.colormap, aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks)
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"{lr:.1e}" for lr in lrs])

    ax.set_xlabel("LoRA Rank", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Validation Perplexity Heatmap", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Perplexity", fontsize=12)

    # Annotate cells with values
    for i in range(len(lrs)):
        for j in range(len(ranks)):
            if not np.isnan(perplexity_matrix[i, j]):
                text = ax.text(j, i, f"{perplexity_matrix[i, j]:.2f}",
                             ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()

    output_path = Path(config.output_dir) / "perplexity_heatmap.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_analysis(results: Dict, config: VisualizationConfig):
    """
    Plot efficiency metrics: tokens/sec and GPU memory vs rank.

    Args:
        results: Experiment results
        config: Visualization configuration
    """
    logger.info("Generating efficiency analysis plot...")

    by_rank = extract_data_by_rank(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.figure_width, config.figure_height))

    ranks = sorted(by_rank.keys())

    # For each rank, average across all LRs
    train_tokens_per_sec = []
    infer_tokens_per_sec = []
    gpu_memory = []

    for rank in ranks:
        rank_results = by_rank[rank]

        avg_train_tps = np.mean([r["training"]["avg_tokens_per_sec"] for r in rank_results])
        avg_infer_tps = np.mean([r["inference"]["tokens_per_sec"] for r in rank_results])
        avg_memory = np.mean([r["gpu_memory"]["estimated_total_gb"] for r in rank_results])

        train_tokens_per_sec.append(avg_train_tps)
        infer_tokens_per_sec.append(avg_infer_tps)
        gpu_memory.append(avg_memory)

    # Plot 1: Throughput
    ax1.plot(ranks, train_tokens_per_sec, marker='o', label="Training", linewidth=2)
    ax1.plot(ranks, infer_tokens_per_sec, marker='s', label="Inference", linewidth=2)
    ax1.set_xlabel("LoRA Rank", fontsize=12)
    ax1.set_ylabel("Tokens/Second", fontsize=12)
    ax1.set_title("Throughput vs LoRA Rank", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory
    ax2.bar(ranks, gpu_memory, color='steelblue', alpha=0.7)
    ax2.set_xlabel("LoRA Rank", fontsize=12)
    ax2.set_ylabel("GPU Memory (GB)", fontsize=12)
    ax2.set_title("Estimated GPU Memory vs LoRA Rank", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path(config.output_dir) / "efficiency_analysis.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def find_best_hyperparameters(results: Dict, config: VisualizationConfig):
    """
    Identify the best (rank, lr) combination based on validation loss.

    Args:
        results: Experiment results
        config: Visualization configuration
    """
    logger.info("Finding best hyperparameters...")

    best_exp = None
    best_val_loss = float('inf')

    for exp_key, result in results.items():
        val_loss = result["evaluation"]["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_exp = (exp_key, result)

    if best_exp is None:
        logger.warning("No valid results found!")
        return

    exp_key, result = best_exp

    # Write report
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "best_hyperparameters.txt"

    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Experiment: {exp_key}\n\n")
        f.write(f"LoRA Rank: {result['rank']}\n")
        f.write(f"Learning Rate: {result['learning_rate']:.2e}\n\n")
        f.write("Performance:\n")
        f.write(f"  Training Loss: {result['training']['final_loss']:.4f}\n")
        f.write(f"  Validation Loss: {result['evaluation']['val_loss']:.4f}\n")
        f.write(f"  Perplexity: {result['evaluation']['perplexity']:.2f}\n\n")
        f.write("Efficiency:\n")
        f.write(f"  Training Tokens/sec: {result['training']['avg_tokens_per_sec']:.1f}\n")
        f.write(f"  Inference Tokens/sec: {result['inference']['tokens_per_sec']:.1f}\n")
        f.write(f"  Inference Latency: {result['inference']['avg_latency_seconds']:.3f}s\n")
        f.write(f"  GPU Memory: {result['gpu_memory']['estimated_total_gb']:.2f} GB\n\n")
        f.write("="*70 + "\n")
        f.write("\nRecommendation:\n")
        f.write(f"Use rank={result['rank']} with lr={result['learning_rate']:.2e} for production.\n")

    logger.info(f"Saved: {report_path}")

    # Print to console
    with open(report_path, "r") as f:
        print("\n" + f.read())


def main(config: VisualizationConfig):
    """
    Main visualization generator.

    Args:
        config: Visualization configuration
    """
    logger.info("="*70)
    logger.info("LoRA Rank Sweep Visualization")
    logger.info("="*70)

    # Load results
    input_dir = Path(config.input_dir)
    results = load_results(input_dir)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate plots
    plot_loss_vs_rank(results, config)
    plot_perplexity_heatmap(results, config)
    plot_efficiency_analysis(results, config)

    # Find best hyperparameters
    find_best_hyperparameters(results, config)

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info(f"All plots saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
