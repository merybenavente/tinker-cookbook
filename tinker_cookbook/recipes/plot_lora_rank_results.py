"""
Visualization script for LoRA Rank Experiment Results.

This script reads the experiment results JSON and generates insightful plots
to analyze how LoRA rank affects model quality, efficiency, and cost.

Usage:
    python -m tinker_cookbook.recipes.plot_lora_rank_results --results path/to/experiment_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def plot_quality_vs_rank(results: dict, output_dir: Path):
    """
    Plot model quality metrics (validation loss and perplexity) vs LoRA rank.

    Shows: How much quality improvement we get from increasing rank.
    """
    ranks = []
    val_losses = []
    perplexities = []

    for rank_key in sorted(results.keys(), key=lambda x: int(x.split("_")[1])):
        rank = results[rank_key]["rank"]
        ranks.append(rank)
        val_losses.append(results[rank_key]["evaluation"]["val_loss"])
        perplexities.append(results[rank_key]["evaluation"]["perplexity"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Validation Loss
    ax1.plot(ranks, val_losses, marker="o", linewidth=2, markersize=8, color="#2E86AB")
    ax1.set_xlabel("LoRA Rank", fontsize=12)
    ax1.set_ylabel("Validation Loss", fontsize=12)
    ax1.set_title("Validation Loss vs LoRA Rank", fontsize=14, fontweight="bold")
    ax1.set_xticks(ranks)
    ax1.grid(True, alpha=0.3)

    # Perplexity
    ax2.plot(ranks, perplexities, marker="s", linewidth=2, markersize=8, color="#A23B72")
    ax2.set_xlabel("LoRA Rank", fontsize=12)
    ax2.set_ylabel("Perplexity", fontsize=12)
    ax2.set_title("Perplexity vs LoRA Rank", fontsize=14, fontweight="bold")
    ax2.set_xticks(ranks)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "quality_vs_rank.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_training_curves(results: dict, output_dir: Path):
    """
    Plot training loss curves over time for each rank.

    Shows: Training stability and convergence speed across ranks.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, (rank_key, color) in enumerate(zip(sorted(results.keys(), key=lambda x: int(x.split("_")[1])), colors)):
        rank = results[rank_key]["rank"]
        metrics_history = results[rank_key]["training"]["metrics_history"]

        steps = [m["step"] for m in metrics_history]
        losses = [m["train_loss"] for m in metrics_history]

        ax.plot(steps, losses, label=f"Rank {rank}", linewidth=2, alpha=0.8, color=color)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Curves by LoRA Rank", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_speed_comparison(results: dict, output_dir: Path):
    """
    Compare training and inference speed across ranks.

    Shows: Whether larger ranks are slower in training/inference.
    """
    ranks = []
    train_speeds = []
    infer_speeds = []

    for rank_key in sorted(results.keys(), key=lambda x: int(x.split("_")[1])):
        rank = results[rank_key]["rank"]
        ranks.append(rank)
        train_speeds.append(results[rank_key]["training"]["avg_tokens_per_sec"])
        infer_speeds.append(results[rank_key]["inference"]["tokens_per_sec"])

    x = np.arange(len(ranks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, train_speeds, width, label="Training", color="#06A77D", alpha=0.8)
    bars2 = ax.bar(x + width / 2, infer_speeds, width, label="Inference", color="#F77F00", alpha=0.8)

    ax.set_xlabel("LoRA Rank", fontsize=12)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title("Training vs Inference Speed by LoRA Rank\n(Higher is Better)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = output_dir / "speed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_gpu_memory_vs_rank(results: dict, output_dir: Path):
    """
    Plot estimated GPU memory usage vs LoRA rank.

    Shows: Hardware requirements for different ranks.
    """
    ranks = []
    gpu_memory = []

    for rank_key in sorted(results.keys(), key=lambda x: int(x.split("_")[1])):
        rank = results[rank_key]["rank"]
        ranks.append(rank)
        gpu_memory.append(results[rank_key]["gpu_memory"]["estimated_total_gb"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ranks, gpu_memory, marker="D", linewidth=2, markersize=10, color="#D62828")
    ax.fill_between(ranks, gpu_memory, alpha=0.3, color="#D62828")

    ax.set_xlabel("LoRA Rank", fontsize=12)
    ax.set_ylabel("Estimated GPU Memory (GB)", fontsize=12)
    ax.set_title("GPU Memory Requirements vs LoRA Rank", fontsize=14, fontweight="bold")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for r, m in zip(ranks, gpu_memory):
        ax.annotate(f"{m:.2f} GB", (r, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "gpu_memory_vs_rank.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_efficiency_tradeoff(results: dict, output_dir: Path):
    """
    Scatter plot: Perplexity vs GPU Memory.

    Shows: The sweet spot - good quality with low memory usage.
    """
    ranks = []
    perplexities = []
    gpu_memory = []

    for rank_key in sorted(results.keys(), key=lambda x: int(x.split("_")[1])):
        rank = results[rank_key]["rank"]
        ranks.append(rank)
        perplexities.append(results[rank_key]["evaluation"]["perplexity"])
        gpu_memory.append(results[rank_key]["gpu_memory"]["estimated_total_gb"])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot with rank as color
    scatter = ax.scatter(
        gpu_memory,
        perplexities,
        s=300,
        c=ranks,
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    # Add rank labels
    for r, p, m in zip(ranks, perplexities, gpu_memory):
        ax.annotate(f"r={r}", (m, p), ha="center", va="center", fontweight="bold", fontsize=10)

    ax.set_xlabel("GPU Memory (GB)", fontsize=12)
    ax.set_ylabel("Perplexity (Lower is Better)", fontsize=12)
    ax.set_title(
        "Efficiency Trade-off: Model Quality vs Hardware Cost\n(Bottom-left corner is optimal)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("LoRA Rank", fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "efficiency_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(results: dict, output_dir: Path):
    """
    Multi-panel summary of all key metrics.

    Shows: Everything at a glance.
    """
    ranks = []
    val_losses = []
    train_speeds = []
    infer_speeds = []
    gpu_memory = []

    for rank_key in sorted(results.keys(), key=lambda x: int(x.split("_")[1])):
        rank = results[rank_key]["rank"]
        ranks.append(rank)
        val_losses.append(results[rank_key]["evaluation"]["val_loss"])
        train_speeds.append(results[rank_key]["training"]["avg_tokens_per_sec"])
        infer_speeds.append(results[rank_key]["inference"]["tokens_per_sec"])
        gpu_memory.append(results[rank_key]["gpu_memory"]["estimated_total_gb"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LoRA Rank Experiment - Comprehensive Summary", fontsize=16, fontweight="bold")

    # Validation Loss
    axes[0, 0].plot(ranks, val_losses, marker="o", linewidth=2, markersize=8, color="#2E86AB")
    axes[0, 0].set_xlabel("LoRA Rank")
    axes[0, 0].set_ylabel("Validation Loss")
    axes[0, 0].set_title("Model Quality")
    axes[0, 0].set_xticks(ranks)
    axes[0, 0].grid(True, alpha=0.3)

    # Speed Comparison
    x = np.arange(len(ranks))
    width = 0.35
    axes[0, 1].bar(x - width / 2, train_speeds, width, label="Training", alpha=0.8, color="#06A77D")
    axes[0, 1].bar(x + width / 2, infer_speeds, width, label="Inference", alpha=0.8, color="#F77F00")
    axes[0, 1].set_xlabel("LoRA Rank")
    axes[0, 1].set_ylabel("Tokens/sec")
    axes[0, 1].set_title("Training & Inference Speed")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(ranks)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # GPU Memory
    axes[1, 0].plot(ranks, gpu_memory, marker="D", linewidth=2, markersize=8, color="#D62828")
    axes[1, 0].fill_between(ranks, gpu_memory, alpha=0.3, color="#D62828")
    axes[1, 0].set_xlabel("LoRA Rank")
    axes[1, 0].set_ylabel("GPU Memory (GB)")
    axes[1, 0].set_title("Hardware Requirements")
    axes[1, 0].set_xticks(ranks)
    axes[1, 0].grid(True, alpha=0.3)

    # Normalized comparison (all metrics 0-1 scale)
    # Normalize: lower is better for val_loss and gpu_memory, higher is better for speeds
    norm_val_loss = 1 - (np.array(val_losses) - min(val_losses)) / (max(val_losses) - min(val_losses) + 1e-10)
    norm_train_speed = (np.array(train_speeds) - min(train_speeds)) / (max(train_speeds) - min(train_speeds) + 1e-10)
    norm_infer_speed = (np.array(infer_speeds) - min(infer_speeds)) / (max(infer_speeds) - min(infer_speeds) + 1e-10)
    norm_memory = 1 - (np.array(gpu_memory) - min(gpu_memory)) / (max(gpu_memory) - min(gpu_memory) + 1e-10)

    overall_score = (norm_val_loss + norm_train_speed + norm_infer_speed + norm_memory) / 4

    axes[1, 1].plot(ranks, overall_score, marker="*", linewidth=3, markersize=15, color="#7209B7")
    axes[1, 1].set_xlabel("LoRA Rank")
    axes[1, 1].set_ylabel("Overall Score (0-1)")
    axes[1, 1].set_title("Overall Efficiency Score\n(Quality + Speed - Memory)")
    axes[1, 1].set_xticks(ranks)
    axes[1, 1].grid(True, alpha=0.3)

    # Highlight best rank
    best_idx = np.argmax(overall_score)
    axes[1, 1].axvline(ranks[best_idx], color="red", linestyle="--", alpha=0.5, linewidth=2)
    axes[1, 1].text(
        ranks[best_idx],
        max(overall_score) * 0.5,
        f"Best: Rank {ranks[best_idx]}",
        rotation=90,
        va="center",
        fontweight="bold",
        color="red",
    )

    plt.tight_layout()
    output_path = output_dir / "comprehensive_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize LoRA Rank Experiment Results")
    parser.add_argument(
        "--results",
        type=str,
        default="./lora_rank_results/experiment_results.json",
        help="Path to experiment results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./lora_rank_results/plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load results
    print(f"\nLoading results from: {args.results}")
    results = load_results(args.results)
    print(f"Found results for {len(results)} ranks: {sorted([r['rank'] for r in results.values()])}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")

    # Generate plots
    print("\nGenerating visualizations...")
    print("-" * 60)

    plot_quality_vs_rank(results, output_dir)
    plot_training_curves(results, output_dir)
    plot_speed_comparison(results, output_dir)
    plot_gpu_memory_vs_rank(results, output_dir)
    plot_efficiency_tradeoff(results, output_dir)
    plot_comprehensive_summary(results, output_dir)

    print("-" * 60)
    print(f"\n✅ All plots generated successfully!")
    print(f"📁 View plots in: {output_dir}\n")


if __name__ == "__main__":
    main()
