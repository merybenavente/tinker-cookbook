"""
LoRA Rank Experiment: Study how LoRA rank affects fine-tuning efficiency, stability, and quality.

This experiment trains models with different LoRA ranks (2, 4, 8, 16, 32) on the Alpaca dataset
and measures:
- Training efficiency (tokens/sec, GPU memory)
- Model quality (validation loss, perplexity)
- Inference performance (latency)
- Training stability (loss variance)
"""

import logging
import random
import time
from dataclasses import dataclass, field

import chz
import tinker
from datasets import load_dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


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
) -> dict:
    """
    Train a model with a specific LoRA rank.

    Args:
        rank: LoRA rank to use
        train_conversations: List of training conversations
        config: Experiment configuration
        service_client: Tinker service client

    Returns:
        Dictionary with training metrics history
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

    return {
        "rank": rank,
        "metrics_history": metrics_history,
        "final_loss": metrics_history[-1]["train_loss"],
        "avg_tokens_per_sec": sum(m["tokens_per_sec"] for m in metrics_history) / len(metrics_history),
    }
