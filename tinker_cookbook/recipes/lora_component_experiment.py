"""
LoRA Component Selection Experiment: Study how component selection affects fine-tuning.

This experiment compares training different components of the model (MLP, Attention, Unembedding)
with various LoRA ranks to understand:
- Which components are most important for fine-tuning
- Parameter efficiency: better to use high rank on specific components or low rank on all?
- Trade-offs between component coverage and rank size

Component configurations tested:
1. Full: train_mlp=True, train_attn=True, train_unembed=True
2. Attention-only: train_mlp=False, train_attn=True, train_unembed=False
3. MLP-only: train_mlp=True, train_attn=False, train_unembed=False
"""

import json
import logging
import random
import time
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Import helper functions from the rank experiment
from tinker_cookbook.recipes.lora_rank_experiment import (
    prepare_alpaca_dataset,
    evaluate_model,
    benchmark_inference,
    estimate_gpu_memory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@chz.chz
class LoRAComponentConfig:
    """Configuration for LoRA component selection experiment"""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # LoRA ranks to test
    ranks_to_test: list[int] = chz.field(default_factory=lambda: [4, 8, 16])

    # Component configurations to test
    component_configs: list[dict] = chz.field(default_factory=lambda: [
        {"name": "full", "train_mlp": True, "train_attn": True, "train_unembed": True},
        {"name": "attn_only", "train_mlp": False, "train_attn": True, "train_unembed": False},
        {"name": "mlp_only", "train_mlp": True, "train_attn": False, "train_unembed": False},
    ])

    # Training hyperparameters (constant across all experiments)
    learning_rate: float = 1e-4
    num_training_steps: int = 1000
    batch_size: int = 8
    max_length: int = 2048

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Dataset configuration
    dataset_name: str = "tatsu-lab/alpaca"
    train_size: int = 4500
    val_size: int = 500

    # Output configuration
    output_dir: str = "./lora_component_results"

    # Inference benchmarking configuration
    num_inference_samples: int = 10
    inference_max_tokens: int = 50
    inference_temperature: float = 0.7

    # Tinker service (None uses the cloud service)
    tinker_url: str | None = None


def train_single_config(
    rank: int,
    component_config: dict,
    train_conversations: list[dict],
    config: LoRAComponentConfig,
    service_client: tinker.ServiceClient,
) -> tuple[dict, tinker.TrainingClient]:
    """
    Train a model with a specific LoRA rank and component configuration.

    Args:
        rank: LoRA rank to use
        component_config: Dict with component flags (train_mlp, train_attn, train_unembed)
        train_conversations: List of training conversations
        config: Experiment configuration
        service_client: Tinker service client

    Returns:
        Tuple of (metrics_dict, training_client)
    """
    comp_name = component_config["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: Rank {rank} | Components: {comp_name}")
    logger.info(f"  train_mlp={component_config['train_mlp']}")
    logger.info(f"  train_attn={component_config['train_attn']}")
    logger.info(f"  train_unembed={component_config['train_unembed']}")
    logger.info(f"{'='*60}")

    # Setup tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create training client with specified rank and component config
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=rank,
        train_mlp=component_config["train_mlp"],
        train_attn=component_config["train_attn"],
        train_unembed=component_config["train_unembed"],
    )

    # Prepare Adam optimizer parameters
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )

    # Storage for metrics
    metrics_history = []

    # Shuffle training data once at the beginning
    shuffled_conversations = train_conversations.copy()
    random.shuffle(shuffled_conversations)

    # Calculate number of batches
    n_batches = len(shuffled_conversations) // config.batch_size
    actual_steps = min(config.num_training_steps, n_batches)
    logger.info(f"Starting training for {actual_steps} steps")

    # Training loop
    for step in range(actual_steps):
        step_start_time = time.time()

        # Get batch sequentially from shuffled data
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_conversations = shuffled_conversations[batch_start:batch_end]

        # Convert conversations to Datum format for Tinker
        batch = [
            conversation_to_datum(
                conversation,
                renderer,
                config.max_length,
                renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            for conversation in batch_conversations
        ]

        # Forward-backward pass
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        fwd_bwd_result = fwd_bwd_future.result()

        # Optimizer step
        optim_step_future = training_client.optim_step(adam_params)
        optim_step_future.result()

        # Compute metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        step_time = time.time() - step_start_time
        num_tokens = sum(d.model_input.length for d in batch)
        tokens_per_sec = num_tokens / step_time

        # Store metrics
        step_metrics = {
            "step": step,
            "rank": rank,
            "component_config": comp_name,
            "train_loss": train_nll,
            "num_tokens": num_tokens,
            "tokens_per_sec": tokens_per_sec,
            "step_time": step_time,
        }
        metrics_history.append(step_metrics)

        # Log progress
        if step % 10 == 0 or step == actual_steps - 1:
            logger.info(
                f"Step {step}/{actual_steps} | "
                f"Loss: {train_nll:.4f} | "
                f"Tokens/sec: {tokens_per_sec:.1f}"
            )

    logger.info(f"Training completed for rank {rank}, components {comp_name}")

    metrics_dict = {
        "rank": rank,
        "component_config": comp_name,
        "metrics_history": metrics_history,
        "final_loss": metrics_history[-1]["train_loss"],
        "avg_tokens_per_sec": sum(m["tokens_per_sec"] for m in metrics_history) / len(metrics_history),
    }

    return metrics_dict, training_client


def main(config: LoRAComponentConfig):
    """
    Main experiment runner: trains and evaluates models with different component configurations.

    For each combination of (rank, component_config):
    1. Train the model
    2. Evaluate on validation set
    3. Benchmark inference
    4. Save results
    """
    logger.info("="*70)
    logger.info("LoRA Component Selection Experiment Starting")
    logger.info("="*70)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Ranks to test: {config.ranks_to_test}")
    logger.info(f"Component configs: {[c['name'] for c in config.component_configs]}")
    logger.info(f"Training steps: {config.num_training_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("="*70)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Tinker service client
    service_client = tinker.ServiceClient(base_url=config.tinker_url)
    logger.info(f"Connected to Tinker service")

    # Load and prepare dataset
    logger.info("\nLoading dataset...")
    train_conversations, val_conversations = prepare_alpaca_dataset(config)

    # Setup tokenizer and renderer (shared across all experiments)
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Storage for all results
    all_results = {}

    # Try to load existing progress
    progress_file = output_dir / "experiment_progress.json"
    if progress_file.exists():
        logger.info(f"\nFound existing progress file: {progress_file}")
        with open(progress_file, "r") as f:
            all_results = json.load(f)
        logger.info(f"Resuming experiment. Already completed: {len(all_results)} configs")
    else:
        logger.info("\nStarting fresh experiment")

    # Train and evaluate each combination of rank and component config
    total_experiments = len(config.ranks_to_test) * len(config.component_configs)
    experiment_count = 0

    for component_config in config.component_configs:
        for rank in config.ranks_to_test:
            experiment_count += 1
            config_key = f"rank_{rank}_comp_{component_config['name']}"

            # Skip if already completed
            if config_key in all_results:
                logger.info(f"\n{'='*70}")
                logger.info(f"SKIPPING {config_key} (already completed)")
                logger.info(f"Progress: {experiment_count}/{total_experiments}")
                logger.info(f"{'='*70}")
                continue

            logger.info(f"\n{'='*70}")
            logger.info(f"STARTING EXPERIMENT {experiment_count}/{total_experiments}")
            logger.info(f"Config: {config_key}")
            logger.info(f"{'='*70}")

            exp_start_time = time.time()

            try:
                # Train the model with this configuration
                train_results, training_client = train_single_config(
                    rank=rank,
                    component_config=component_config,
                    train_conversations=train_conversations,
                    config=config,
                    service_client=service_client,
                )

                # Evaluate on validation set
                eval_results = evaluate_model(
                    training_client=training_client,
                    val_conversations=val_conversations,
                    renderer=renderer,
                    config=config,
                )

                # Benchmark inference
                inference_results = benchmark_inference(
                    training_client=training_client,
                    tokenizer=tokenizer,
                    renderer=renderer,
                    config=config,
                )

                # Estimate GPU memory
                gpu_memory = estimate_gpu_memory(rank=rank, model_name=config.model_name)

                exp_total_time = time.time() - exp_start_time

                # Combine results
                experiment_results = {
                    "rank": rank,
                    "component_config": component_config,
                    "training": {
                        "final_loss": train_results["final_loss"],
                        "avg_tokens_per_sec": train_results["avg_tokens_per_sec"],
                        "metrics_history": train_results["metrics_history"],
                    },
                    "evaluation": eval_results,
                    "inference": inference_results,
                    "gpu_memory": gpu_memory,
                    "total_time_seconds": exp_total_time,
                }

                all_results[config_key] = experiment_results

                # Save progress after each experiment
                with open(progress_file, "w") as f:
                    json.dump(all_results, f, indent=2)

                logger.info(f"\n{'='*70}")
                logger.info(f"EXPERIMENT {config_key} COMPLETED")
                logger.info(f"  Training Loss: {train_results['final_loss']:.4f}")
                logger.info(f"  Validation Loss: {eval_results['val_loss']:.4f}")
                logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                logger.info(f"  Total Time: {exp_total_time:.1f}s")
                logger.info(f"{'='*70}")

            except Exception as e:
                logger.error(f"\n{'!'*70}")
                logger.error(f"ERROR: {config_key} failed")
                logger.error(f"Error: {e}")
                logger.error(f"{'!'*70}")
                logger.info("Progress has been saved. Skipping to next config...")
                continue

    # Save final results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT COMPLETED")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*70}")

    # Print comparison table
    print_results_table(all_results, config)


def print_results_table(all_results: dict, config: LoRAComponentConfig):
    """Print a comparison table of all results"""
    logger.info("\n" + "="*120)
    logger.info("RESULTS SUMMARY")
    logger.info("="*120)

    # Group by component config
    for comp_config in config.component_configs:
        comp_name = comp_config["name"]
        logger.info(f"\n{comp_name.upper()} Configuration:")
        logger.info(f"  train_mlp={comp_config['train_mlp']}, train_attn={comp_config['train_attn']}, train_unembed={comp_config['train_unembed']}")
        logger.info("-" * 120)
        logger.info(
            f"{'Rank':<8} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12} "
            f"{'Train tok/s':<12} {'Infer tok/s':<12} {'Infer Lat':<12} {'GPU (GB)':<10}"
        )
        logger.info("-" * 120)

        for rank in config.ranks_to_test:
            config_key = f"rank_{rank}_comp_{comp_name}"
            if config_key not in all_results:
                logger.info(f"{rank:<8} {'FAILED - No results available'}")
                continue

            result = all_results[config_key]
            logger.info(
                f"{rank:<8} "
                f"{result['training']['final_loss']:<12.4f} "
                f"{result['evaluation']['val_loss']:<12.4f} "
                f"{result['evaluation']['perplexity']:<12.2f} "
                f"{result['training']['avg_tokens_per_sec']:<12.1f} "
                f"{result['inference']['tokens_per_sec']:<12.1f} "
                f"{result['inference']['avg_latency_seconds']:<12.3f} "
                f"{result['gpu_memory']['estimated_total_gb']:<10.2f}"
            )
        logger.info("-" * 120)

    logger.info("="*120)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
