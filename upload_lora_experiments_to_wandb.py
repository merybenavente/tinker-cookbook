"""
Upload LoRA rank experiment results to WandB.

This script handles the experiment_results.json format from your LoRA rank experiments.

Usage:
    python upload_lora_experiments_to_wandb.py --results-file lora_rank_results/experiment_results.json --project "lora-rank-comparison"
"""

import argparse
import json
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    exit(1)


def upload_lora_experiments_to_wandb(
    results_file: str,
    project: str,
    tags: list[str] | None = None,
):
    """Upload LoRA rank experiment results to WandB."""
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Load all experiment results
    with open(results_path) as f:
        all_results = json.load(f)

    print(f"Found {len(all_results)} experiments to upload:")
    for exp_name in all_results.keys():
        print(f"  - {exp_name}")
    print()

    # Track entity for final URL
    entity = None

    # Upload each experiment as a separate run
    for exp_name, exp_data in all_results.items():
        rank = exp_data.get("rank")
        training_data = exp_data.get("training", {})
        metrics_history = training_data.get("metrics_history", [])

        # Prepare config (hyperparameters)
        config = {
            "rank": rank,
            "final_loss": training_data.get("final_loss"),
            "avg_tokens_per_sec": training_data.get("avg_tokens_per_sec"),
        }

        # Add evaluation data if present
        if "evaluation" in exp_data:
            eval_data = exp_data["evaluation"]
            config["eval_perplexity"] = eval_data.get("perplexity")
            config["eval_loss"] = eval_data.get("loss")

        # Add inference data if present
        if "inference" in exp_data:
            inf_data = exp_data["inference"]
            config["avg_inference_tokens_per_sec"] = inf_data.get("avg_tokens_per_sec")

        # Initialize wandb run for this experiment
        run_tags = [f"rank_{rank}"] + (tags or [])
        run = wandb.init(
            project=project,
            name=exp_name,
            config=config,
            tags=run_tags,
            reinit=True,  # Allow multiple runs in same script
        )

        # Capture entity from first run
        if entity is None:
            entity = run.entity

        print(f"Uploading {exp_name} (rank={rank})...")
        print(f"  URL: {run.url}")
        print(f"  Metrics: {len(metrics_history)} steps")

        # Upload training metrics
        for metric_entry in metrics_history:
            step = metric_entry.get("step")

            # Prepare metrics for this step
            log_metrics = {
                "train_loss": metric_entry.get("train_loss"),
                "num_tokens": metric_entry.get("num_tokens"),
                "tokens_per_sec": metric_entry.get("tokens_per_sec"),
                "step_time": metric_entry.get("step_time"),
            }

            # Remove None values
            log_metrics = {k: v for k, v in log_metrics.items() if v is not None}

            wandb.log(log_metrics, step=step)

        # Log final summary metrics
        if "evaluation" in exp_data:
            eval_data = exp_data["evaluation"]
            wandb.summary["eval_perplexity"] = eval_data.get("perplexity")
            wandb.summary["eval_loss"] = eval_data.get("loss")
            wandb.summary["eval_avg_tokens_per_sec"] = eval_data.get("avg_tokens_per_sec")

        if "inference" in exp_data:
            inf_data = exp_data["inference"]
            wandb.summary["inference_avg_tokens_per_sec"] = inf_data.get("avg_tokens_per_sec")
            wandb.summary["inference_total_samples"] = inf_data.get("total_samples")

        print(f"  ✓ Uploaded successfully!")
        wandb.finish()
        print()

    print(f"🎉 All experiments uploaded to project: {project}")
    if entity:
        print(f"View at: https://wandb.ai/{entity}/{project}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload LoRA rank experiment results to WandB"
    )
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to experiment_results.json file",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="WandB project name",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Tags to add to all runs",
    )

    args = parser.parse_args()
    upload_lora_experiments_to_wandb(
        results_file=args.results_file,
        project=args.project,
        tags=args.tags,
    )
