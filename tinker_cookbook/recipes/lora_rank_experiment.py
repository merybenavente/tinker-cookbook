"""
LoRA Rank Experiment: Study how LoRA rank affects fine-tuning efficiency, stability, and quality.

This experiment trains models with different LoRA ranks (2, 4, 8, 16, 32) on the Alpaca dataset
and measures:
- Training efficiency (tokens/sec, GPU memory)
- Model quality (validation loss, perplexity)
- Inference performance (latency)
- Training stability (loss variance)
"""

from dataclasses import dataclass, field

import chz
from datasets import load_dataset


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
