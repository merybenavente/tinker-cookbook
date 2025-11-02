"""
Upload existing metrics.jsonl files to WandB retroactively.

Usage:
    python upload_metrics_to_wandb.py --metrics-file /path/to/metrics.jsonl --project my-project --run-name my-run
"""

import argparse
import json
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    exit(1)


def upload_metrics_to_wandb(
    metrics_file: str,
    project: str,
    run_name: str,
    config_file: str | None = None,
    tags: list[str] | None = None,
):
    """Upload metrics from a local metrics.jsonl file to WandB."""
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # Load config if available
    config = None
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
    elif metrics_path.parent / "config.json":
        # Try to find config.json in same directory
        config_path = metrics_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

    # Initialize wandb run
    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags or [],
    )

    print(f"Uploading metrics from {metrics_file} to WandB project '{project}'...")
    print(f"Run URL: {run.url}")

    # Read and upload metrics line by line
    with open(metrics_path) as f:
        for i, line in enumerate(f):
            metrics = json.loads(line.strip())
            step = metrics.pop("step", None)

            # Log to wandb
            wandb.log(metrics, step=step)

            if (i + 1) % 10 == 0:
                print(f"  Uploaded {i + 1} metric entries...")

    print(f"✓ Successfully uploaded metrics to {run.url}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload existing metrics.jsonl files to WandB"
    )
    parser.add_argument(
        "--metrics-file",
        required=True,
        help="Path to metrics.jsonl file",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="WandB project name",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this run in WandB",
    )
    parser.add_argument(
        "--config-file",
        help="Path to config.json file (optional, auto-detected if in same directory)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Tags to add to the run",
    )

    args = parser.parse_args()
    upload_metrics_to_wandb(
        metrics_file=args.metrics_file,
        project=args.project,
        run_name=args.run_name,
        config_file=args.config_file,
        tags=args.tags,
    )
