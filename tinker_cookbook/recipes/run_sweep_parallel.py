"""
Parallel Sweep Runner for LoRA Rank Experiments

This script launches multiple rank/lr experiments in parallel using subprocesses.
It monitors progress and generates a consolidated report when all experiments complete.

Usage:
    python -m tinker_cookbook.recipes.run_sweep_parallel

The script will:
1. Read config from lora_rank_experiment.py
2. Calculate all rank x lr combinations
3. Launch experiments in parallel (respecting max_parallel_runs limit)
4. Monitor progress until all complete
5. Generate visualizations and final report
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import chz

from tinker_cookbook.recipes.lora_rank_experiment import (
    LoRARankConfig,
    get_default_lrs,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@chz.chz
class ParallelSweepConfig:
    """Configuration for parallel sweep execution"""

    # Use config from lora_rank_experiment
    base_config: LoRARankConfig = chz.field(default_factory=LoRARankConfig)

    # Parallel execution settings
    max_parallel: int = 5  # Max concurrent experiments
    poll_interval: float = 10.0  # Seconds between progress checks

    # Python executable to use (in case you're using a venv)
    python_executable: str = "python"


def get_all_combinations(config: LoRARankConfig) -> List[Tuple[int, float]]:
    """
    Get all (rank, lr) combinations to test.

    Args:
        config: Experiment configuration

    Returns:
        List of (rank, lr) tuples
    """
    ranks = config.ranks_to_test

    if config.lrs_to_test is None:
        lrs = get_default_lrs(config.model_name)
    else:
        lrs = config.lrs_to_test

    combinations = [(rank, lr) for rank in ranks for lr in lrs]
    return combinations


def get_completed_experiments(output_dir: Path) -> set:
    """
    Read experiment_progress.json to see which experiments are done.

    Args:
        output_dir: Output directory containing experiment_progress.json

    Returns:
        Set of experiment keys that are completed
    """
    progress_file = output_dir / "experiment_progress.json"

    if not progress_file.exists():
        return set()

    with open(progress_file, "r") as f:
        results = json.load(f)

    return set(results.keys())


def launch_experiment(
    rank: int,
    lr: float,
    config: LoRARankConfig,
    python_exec: str
) -> subprocess.Popen:
    """
    Launch a single experiment as a subprocess.

    Args:
        rank: LoRA rank to test
        lr: Learning rate to test
        config: Base experiment configuration
        python_exec: Python executable to use

    Returns:
        Subprocess handle
    """
    exp_key = f"rank_{rank}_lr_{lr:.0e}"
    logger.info(f"Launching {exp_key}...")

    cmd = [
        python_exec,
        "-m",
        "tinker_cookbook.recipes.lora_rank_experiment",
        f"single_rank={rank}",
        f"single_lr={lr}",
        f"model_name={config.model_name}",
        f"num_training_steps={config.num_training_steps}",
        f"batch_size={config.batch_size}",
        f"output_dir={config.output_dir}",
        f"train_size={config.train_size}",
        f"val_size={config.val_size}",
    ]

    # Redirect stdout/stderr to log files
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{exp_key}.log"

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    logger.info(f"  Process ID: {process.pid}")
    logger.info(f"  Log file: {log_file}")

    return process


def monitor_progress(
    active_processes: dict,
    completed_keys: set,
    output_dir: Path,
    poll_interval: float
) -> Tuple[dict, set]:
    """
    Check status of running processes and update completed set.

    Args:
        active_processes: Dict mapping exp_key -> subprocess.Popen
        completed_keys: Set of already completed experiment keys
        output_dir: Output directory
        poll_interval: How long to wait between checks

    Returns:
        Updated (active_processes, completed_keys)
    """
    time.sleep(poll_interval)

    # Check which processes have finished
    finished = []
    for exp_key, process in active_processes.items():
        if process.poll() is not None:  # Process finished
            finished.append(exp_key)

            # Check if it completed successfully
            new_completed = get_completed_experiments(output_dir)
            if exp_key in new_completed:
                logger.info(f"✓ {exp_key} completed successfully")
                completed_keys.add(exp_key)
            else:
                logger.warning(f"✗ {exp_key} finished but not in results (may have failed)")

    # Remove finished processes
    for exp_key in finished:
        del active_processes[exp_key]

    return active_processes, completed_keys


def main(config: ParallelSweepConfig):
    """
    Main parallel sweep runner.

    Args:
        config: Parallel sweep configuration
    """
    logger.info("="*70)
    logger.info("Parallel LoRA Rank Sweep Starting")
    logger.info("="*70)

    base_config = config.base_config

    # Get all combinations to test
    all_combinations = get_all_combinations(base_config)
    total_experiments = len(all_combinations)

    logger.info(f"Model: {base_config.model_name}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Max parallel: {config.max_parallel}")
    logger.info(f"Output directory: {base_config.output_dir}")
    logger.info("="*70)

    # Check which experiments are already completed
    output_dir = Path(base_config.output_dir)
    completed_keys = get_completed_experiments(output_dir)

    if completed_keys:
        logger.info(f"\nResuming sweep. {len(completed_keys)} experiments already completed:")
        for key in sorted(completed_keys):
            logger.info(f"  ✓ {key}")

    # Filter out completed experiments
    remaining_combinations = [
        (rank, lr) for rank, lr in all_combinations
        if f"rank_{rank}_lr_{lr:.0e}" not in completed_keys
    ]

    logger.info(f"\nRemaining experiments to run: {len(remaining_combinations)}")

    if not remaining_combinations:
        logger.info("All experiments already completed!")
        logger.info("Run visualize_sweep_results.py to generate plots.")
        return

    # Track active processes
    active_processes = {}

    # Main execution loop
    try:
        while remaining_combinations or active_processes:
            # Launch new experiments if we have capacity
            while remaining_combinations and len(active_processes) < config.max_parallel:
                rank, lr = remaining_combinations.pop(0)
                exp_key = f"rank_{rank}_lr_{lr:.0e}"

                process = launch_experiment(
                    rank, lr, base_config, config.python_executable
                )
                active_processes[exp_key] = process

            # Show status
            logger.info(f"\nStatus: {len(completed_keys)}/{total_experiments} completed, "
                       f"{len(active_processes)} running, {len(remaining_combinations)} queued")

            if active_processes:
                logger.info("Active experiments:")
                for exp_key in active_processes.keys():
                    logger.info(f"  → {exp_key}")

            # Monitor progress
            if active_processes:
                active_processes, completed_keys = monitor_progress(
                    active_processes, completed_keys, output_dir, config.poll_interval
                )

    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user. Stopping all processes...")
        for process in active_processes.values():
            process.terminate()
        logger.info("You can resume the sweep by running this script again.")
        return

    logger.info("\n" + "="*70)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info(f"Total completed: {len(completed_keys)}/{total_experiments}")
    logger.info(f"Results saved to: {output_dir / 'experiment_progress.json'}")
    logger.info("="*70)

    logger.info("\nNext steps:")
    logger.info("1. Run visualize_sweep_results.py to generate plots")
    logger.info("2. Check experiment_progress.json for detailed results")
    logger.info("3. Train your best model using the optimal hyperparameters!")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
