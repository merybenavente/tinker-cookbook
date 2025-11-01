"""
LoRA Rank Experiment: Study how LoRA rank affects fine-tuning efficiency, stability, and quality.

This experiment trains models with different LoRA ranks (2, 4, 8, 16, 32) on the Alpaca dataset
and measures:
- Training efficiency (tokens/sec, GPU memory)
- Model quality (validation loss, perplexity)
- Inference performance (latency)
- Training stability (loss variance)
"""

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import chz
import tinker
from datasets import load_dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@chz.chz
@dataclass
class LoRARankConfig:
    """Configuration for LoRA rank experiment"""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # LoRA ranks to test
    ranks_to_test: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])

    # Training hyperparameters (constant across all ranks)
    learning_rate: float = 1e-4
    num_training_steps: int = 1000
    batch_size: int = 8
    eval_every_n_steps: int = 100
    max_length: int = 2048

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Dataset configuration
    dataset_name: str = "tatsu-lab/alpaca"
    train_size: int = 4500  # Number of examples for training
    val_size: int = 500     # Number of examples for validation

    # Output configuration
    output_dir: str = "./lora_rank_results"
    checkpoint_dir: str = "./lora_rank_checkpoints"

    # Inference benchmarking configuration
    num_inference_samples: int = 10  # Number of prompts to test
    inference_max_tokens: int = 50   # Tokens to generate per prompt
    inference_temperature: float = 0.7

    # Tinker service
    tinker_url: str = "http://localhost:8000"


def prepare_alpaca_dataset(config: LoRARankConfig) -> tuple[list[dict], list[dict]]:
    """
    Load and prepare the Alpaca dataset for Tinker training.

    Returns:
        Tuple of (train_data, val_data) where each item is a list of conversation dicts
    """
    print(f"Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name)

    # Alpaca has: instruction, input, output
    raw_data = ds["train"]

    train_conversations = []
    val_conversations = []

    # Split into train and validation
    train_examples = raw_data[:config.train_size]
    val_examples = raw_data[config.train_size:config.train_size + config.val_size]

    # Convert to conversation format
    for examples, target_list in [(train_examples, train_conversations),
                                   (val_examples, val_conversations)]:
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            output = examples["output"][i]

            # Combine instruction and input if input exists
            if input_text.strip():
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction

            # Create conversation format for Tinker
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]

            target_list.append(conversation)

    print(f"Prepared {len(train_conversations)} training examples")
    print(f"Prepared {len(val_conversations)} validation examples")

    return train_conversations, val_conversations


def train_single_rank(
    rank: int,
    train_conversations: list[dict],
    config: LoRARankConfig,
    service_client: tinker.ServiceClient,
) -> tuple[dict, tinker.TrainingClient]:
    """
    Train a model with a specific LoRA rank.

    Args:
        rank: LoRA rank to use
        train_conversations: List of training conversations
        config: Experiment configuration
        service_client: Tinker service client

    Returns:
        Tuple of (metrics_dict, training_client) where metrics_dict contains training history
        and training_client has the trained model weights
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with LoRA rank: {rank}")
    logger.info(f"{'='*60}")

    # Setup tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Create training client with specified rank
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=rank  # This is the only thing that changes between experiments
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
    logger.info(f"Starting training for {actual_steps} steps ({n_batches} batches available)")

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
                f"Tokens/sec: {tokens_per_sec:.1f} | "
                f"Time: {step_time:.2f}s"
            )

    logger.info(f"Training completed for rank {rank}")

    metrics_dict = {
        "rank": rank,
        "metrics_history": metrics_history,
        "final_loss": metrics_history[-1]["train_loss"],
        "avg_tokens_per_sec": sum(m["tokens_per_sec"] for m in metrics_history) / len(metrics_history),
    }

    return metrics_dict, training_client


def benchmark_inference(
    training_client: tinker.TrainingClient,
    tokenizer,
    renderer: renderers.Renderer,
    config: LoRARankConfig,
) -> dict:
    """
    Benchmark inference latency for the trained model.

    Creates a sampling client from the training client and measures
    generation speed on test prompts.

    Args:
        training_client: Tinker training client with trained model weights
        tokenizer: Tokenizer for the model
        renderer: Renderer for formatting prompts
        config: Experiment configuration

    Returns:
        Dictionary with inference metrics (latency, tokens/sec)
    """
    logger.info("Benchmarking inference performance...")

    # Create sampling client from trained model
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="inference_benchmark"
    )

    # Test prompts (diverse set from Alpaca-style instructions)
    test_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How do you make a good first impression in a job interview?",
        "What is the difference between RAM and ROM?",
        "Explain the theory of relativity.",
        "Write a short poem about the ocean.",
        "What are the benefits of regular exercise?",
        "How does the internet work?",
    ]

    # Use only the requested number of samples
    test_prompts = test_prompts[:config.num_inference_samples]

    # Configure sampling parameters
    sampling_params = tinker.SamplingParams(
        max_tokens=config.inference_max_tokens,
        temperature=config.inference_temperature,
        stop=renderer.get_stop_sequences(),
    )

    latencies = []
    total_tokens_generated = 0

    for i, prompt in enumerate(test_prompts):
        # Tokenize prompt
        tokenized_prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt))

        # Measure generation time
        start_time = time.time()
        future = sampling_client.sample(
            prompt=tokenized_prompt,
            sampling_params=sampling_params,
            num_samples=1,
        )
        result = future.result()
        latency = time.time() - start_time

        # Count generated tokens
        num_tokens = len(result.sequences[0].tokens)
        total_tokens_generated += num_tokens
        latencies.append(latency)

        if i == 0:
            # Log first example
            response = tokenizer.decode(result.sequences[0].tokens)
            logger.info(f"Sample generation:\n  Prompt: {prompt[:50]}...\n  Response: {response[:100]}...")

    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    total_time = sum(latencies)
    tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0

    logger.info(f"Inference benchmark complete:")
    logger.info(f"  Avg latency: {avg_latency:.3f}s per prompt")
    logger.info(f"  Tokens/sec: {tokens_per_sec:.1f}")
    logger.info(f"  Total tokens generated: {total_tokens_generated}")

    return {
        "avg_latency_seconds": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens_generated": total_tokens_generated,
        "num_prompts": len(test_prompts),
    }


def evaluate_model(
    training_client: tinker.TrainingClient,
    val_conversations: list[dict],
    renderer: renderers.Renderer,
    config: LoRARankConfig,
) -> dict:
    """
    Evaluate model on validation data.

    This does forward passes (no gradient computation) on the validation set
    to compute validation loss and perplexity.

    Args:
        training_client: Tinker training client with current model weights
        val_conversations: List of validation conversations
        renderer: Renderer for converting conversations to model format
        config: Experiment configuration

    Returns:
        Dictionary with evaluation metrics (val_loss, perplexity)
    """
    logger.info("Running evaluation on validation set...")

    # Convert all validation conversations to Datum format
    val_data = [
        conversation_to_datum(
            conversation,
            renderer,
            config.max_length,
            renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        for conversation in val_conversations
    ]

    # Forward pass only (no backward, no optimizer step)
    # This computes loss without updating weights
    future = training_client.forward(val_data, loss_fn="cross_entropy")
    result = future.result()

    # Extract logprobs and compute NLL
    logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in val_data]
    val_nll = compute_mean_nll(logprobs, weights)

    # Perplexity is exp(NLL)
    perplexity = math.exp(val_nll)

    logger.info(f"Validation NLL: {val_nll:.4f} | Perplexity: {perplexity:.2f}")

    return {
        "val_loss": val_nll,
        "perplexity": perplexity,
    }


def main(config: LoRARankConfig):
    """
    Main experiment runner: trains and evaluates models with different LoRA ranks.

    For each rank in config.ranks_to_test:
    1. Train the model
    2. Evaluate on validation set
    3. Collect metrics
    4. Save results

    Args:
        config: Experiment configuration
    """
    logger.info("="*70)
    logger.info("LoRA Rank Experiment Starting")
    logger.info("="*70)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Ranks to test: {config.ranks_to_test}")
    logger.info(f"Training steps: {config.num_training_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("="*70)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Tinker service client
    service_client = tinker.ServiceClient(base_url=config.tinker_url)
    logger.info(f"Connected to Tinker service at {config.tinker_url}")

    # Load and prepare dataset
    logger.info("\nLoading dataset...")
    train_conversations, val_conversations = prepare_alpaca_dataset(config)

    # Setup tokenizer and renderer (shared across all ranks)
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Storage for all results
    all_results = {}

    # Train and evaluate each rank
    for rank in config.ranks_to_test:
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING RANK {rank}")
        logger.info(f"{'='*70}")

        rank_start_time = time.time()

        # Train the model with this rank
        train_results, training_client = train_single_rank(
            rank=rank,
            train_conversations=train_conversations,
            config=config,
            service_client=service_client,
        )

        # Evaluate on validation set with the trained model
        eval_results = evaluate_model(
            training_client=training_client,
            val_conversations=val_conversations,
            renderer=renderer,
            config=config,
        )

        # Benchmark inference latency
        inference_results = benchmark_inference(
            training_client=training_client,
            tokenizer=tokenizer,
            renderer=renderer,
            config=config,
        )

        rank_total_time = time.time() - rank_start_time

        # Combine results
        rank_results = {
            "rank": rank,
            "training": {
                "final_loss": train_results["final_loss"],
                "avg_tokens_per_sec": train_results["avg_tokens_per_sec"],
                "metrics_history": train_results["metrics_history"],
            },
            "evaluation": eval_results,
            "inference": inference_results,
            "total_time_seconds": rank_total_time,
        }

        all_results[f"rank_{rank}"] = rank_results

        logger.info(f"\n{'='*70}")
        logger.info(f"RANK {rank} COMPLETED")
        logger.info(f"  Training Loss: {train_results['final_loss']:.4f}")
        logger.info(f"  Validation Loss: {eval_results['val_loss']:.4f}")
        logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
        logger.info(f"  Training Tokens/sec: {train_results['avg_tokens_per_sec']:.1f}")
        logger.info(f"  Inference Tokens/sec: {inference_results['tokens_per_sec']:.1f}")
        logger.info(f"  Inference Latency: {inference_results['avg_latency_seconds']:.3f}s")
        logger.info(f"  Total Time: {rank_total_time:.1f}s")
        logger.info(f"{'='*70}")

    # Save results to JSON
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT COMPLETED")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*70}")

    # Print summary table
    logger.info("\nSUMMARY:")
    logger.info("-" * 95)
    logger.info(
        f"{'Rank':<8} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12} "
        f"{'Train tok/s':<12} {'Infer tok/s':<12} {'Infer Lat':<12}"
    )
    logger.info("-" * 95)
    for rank in config.ranks_to_test:
        result = all_results[f"rank_{rank}"]
        logger.info(
            f"{rank:<8} "
            f"{result['training']['final_loss']:<12.4f} "
            f"{result['evaluation']['val_loss']:<12.4f} "
            f"{result['evaluation']['perplexity']:<12.2f} "
            f"{result['training']['avg_tokens_per_sec']:<12.1f} "
            f"{result['inference']['tokens_per_sec']:<12.1f} "
            f"{result['inference']['avg_latency_seconds']:<12.3f}"
        )
    logger.info("-" * 95)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
