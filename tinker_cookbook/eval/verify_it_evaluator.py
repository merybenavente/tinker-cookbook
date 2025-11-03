"""
Evaluator for the verify_it classification task.

Usage:
    python -m tinker_cookbook.eval.verify_it_evaluator \
        model_path="tinker://c678a30e-090e-47ea-a348-f08a6e2de6d0/sampler_weights/final"
"""
import asyncio
import json
import logging
from typing import Any, Callable

import chz
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import sklearn for confusion matrix
try:
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    _sklearn_available = True
except ImportError:
    _sklearn_available = False
    logger.warning("sklearn not available. Confusion matrix will not be generated. Install with: pip install scikit-learn")


class VerifyItEvaluator(SamplingClientEvaluator):
    """
    Evaluator for the verify_it factual statement classification task.
    """

    def __init__(
        self,
        test_file_path: str,
        model_name: str,
        renderer_name: str,
        temperature: float = 0.0,
        max_tokens: int = 50,
    ):
        """
        Initialize the VerifyItEvaluator.
        Args:
            test_file_path: Path to the JSONL test file
            model_name: Name of the base model
            renderer_name: Name of the renderer to use
            temperature: Sampling temperature (0.0 for greedy)
            max_tokens: Maximum tokens to generate
        """
        self.test_file_path = test_file_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        # Load test dataset
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> list[dict[str, str]]:
        """Load the test dataset from JSONL file."""
        dataset = []
        with open(self.test_file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                messages = data["messages"]
                # Extract user input and expected output
                user_msg = next(m for m in messages if m["role"] == "user")
                assistant_msg = next(m for m in messages if m["role"] == "assistant")
                dataset.append({
                    "input": user_msg["content"],
                    "expected": assistant_msg["content"],
                })
        logger.info(f"Loaded {len(dataset)} test examples from {self.test_file_path}")
        return dataset

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation on the test set.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics
        """
        num_examples = len(self.dataset)
        num_correct = 0
        predictions = []

        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        logger.info(f"Running evaluation on {num_examples} examples...")

        for i, datum in enumerate(self.dataset):
            # Build the prompt
            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["input"])]
            )
            # Generate response
            response: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )

            # Parse the response
            print(">>> input:", datum["input"])
            tokens: list[int] = response.sequences[0].tokens
            parsed_response: renderers.Message = self.renderer.parse_response(tokens)[0]
            full_response = parsed_response["content"].strip()

            # Extract label from CoT response (if present)
            if "LABEL:" in full_response:
                # Extract text after "LABEL:"
                predicted = full_response.split("LABEL:")[-1].strip()
                print(">>> full response:", full_response)
            else:
                # Fallback for non-CoT responses
                predicted = full_response.replace("Classify: ", "").strip()

            expected = datum["expected"].strip()
            print(">>> predicted:", predicted)
            print(">>> expected:", datum["expected"].strip())

            # Check if correct (exact match)
            is_correct = predicted.lower() == expected.lower()
            if is_correct:
                print("correct")
                num_correct += 1
            else:
                print("incorrect")
            print("\n\n")

            predictions.append({
                "input": datum["input"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

            # Log progress every 100 examples
            if (i + 1) % 100 == 0:
                current_acc = num_correct / (i + 1)
                logger.info(f"Progress: {i + 1}/{num_examples} - Current accuracy: {current_acc:.4f}")

        accuracy = num_correct / num_examples
        logger.info(f"\nFinal Results:")
        logger.info(f"  Total examples: {num_examples}")
        logger.info(f"  Correct: {num_correct}")
        logger.info(f"  Accuracy: {accuracy:.4f}")

        # Generate confusion matrix if sklearn available
        if _sklearn_available:
            y_true = [pred["expected"].lower() for pred in predictions]
            y_pred = [pred["predicted"].lower() for pred in predictions]

            # Get unique labels
            labels = sorted(list(set(y_true + y_pred)))

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            logger.info("\n" + "=" * 60)
            logger.info("CONFUSION MATRIX")
            logger.info("=" * 60)

            # Print header
            header = "True \\ Pred".ljust(30) + " | ".join([label[:20].ljust(20) for label in labels])
            logger.info(header)
            logger.info("-" * len(header))

            # Print matrix rows
            for i, label in enumerate(labels):
                row = label[:28].ljust(30) + " | ".join([str(cm[i][j]).ljust(20) for j in range(len(labels))])
                logger.info(row)

            # Print classification report
            logger.info("\n" + "=" * 60)
            logger.info("CLASSIFICATION REPORT")
            logger.info("=" * 60)
            report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
            logger.info("\n" + report)

        # Show a few examples
        logger.info("\nSample predictions:")
        for i, pred in enumerate(predictions[:5]):
            logger.info(f"\nExample {i + 1}:")
            logger.info(f"  Input: {pred['input'][:80]}...")
            logger.info(f"  Expected: {pred['expected']}")
            logger.info(f"  Predicted: {pred['predicted']}")
            logger.info(f"  Correct: {pred['correct']}")

        return {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_examples,
        }


@chz.chz
class Config:
    model_path: str
    base_model: str = "meta-llama/Llama-3.1-70B"
    test_file: str = "example-data/verify_it_just_labels/groundtruth.jsonl"
    renderer_name: str = "llama3"
    temperature: float = 0.0
    max_tokens: int = 50


async def main(config: Config):
    """Main evaluation function."""
    logger.info(f"Evaluating model: {config.model_path}")
    logger.info(f"Test file: {config.test_file}")

    # Create service client and sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=config.base_model,
        model_path=config.model_path,
    )

    # Create evaluator
    evaluator = VerifyItEvaluator(
        test_file_path=config.test_file,
        model_name=config.base_model,
        renderer_name=config.renderer_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Run evaluation
    metrics = await evaluator(sampling_client)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value}")


def run(config: Config):
    asyncio.run(main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(run, allow_hyphens=True)
