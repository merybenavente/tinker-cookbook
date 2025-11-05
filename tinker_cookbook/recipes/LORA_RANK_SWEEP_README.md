# LoRA Rank Hyperparameter Sweep

Systematic evaluation of LoRA rank and learning rate combinations for fine-tuning, with parallel execution and automated analysis.

## Motivation

Choosing the right LoRA rank involves a trade-off between model capacity, training efficiency, and inference cost. This sweep framework automates that search across rank and learning rate dimensions, making it easy to identify the Pareto-optimal configuration for a given task.

## Components

| File | Purpose |
|------|---------|
| `lora_rank_experiment.py` | Core experiment runner (full sweep or single rank/LR run) |
| `run_sweep_parallel.py` | Parallel execution wrapper with progress tracking and auto-resume |
| `visualize_sweep_results.py` | Generates loss curves, perplexity heatmaps, and efficiency plots |

## Usage

### Parallel sweep (recommended)

```bash
python -m tinker_cookbook.recipes.run_sweep_parallel
python -m tinker_cookbook.recipes.visualize_sweep_results
```

### Single experiment

```bash
python -m tinker_cookbook.recipes.lora_rank_experiment \
    single_rank=16 \
    single_lr=3e-4
```

## Configuration

```python
@chz.chz
class LoRARankConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    ranks_to_test: list[int] = [2, 4, 8, 16, 32]
    lrs_to_test: list[float] | None = None  # None = auto-detect via get_lr()
    num_training_steps: int = 1000
    batch_size: int = 8
    train_size: int = 4500
    val_size: int = 500
    max_parallel_runs: int = 5
```

Learning rates default to one order of magnitude above/below the model's default LR (via `get_lr()`), following Tinker's recommended sweep strategy.

## Output

```
lora_rank_results/
├── experiment_progress.json     # Results + resume state
├── rank_{r}_lr_{lr}.log         # Per-experiment logs
└── visualizations/
    ├── loss_vs_rank.png
    ├── perplexity_heatmap.png
    ├── efficiency_analysis.png
    └── best_hyperparameters.txt
```

## Interpreting results

- **Validation loss** is the primary quality metric — training loss alone can be misleading at higher ranks due to overfitting.
- The **perplexity heatmap** shows rank x LR interactions at a glance. Look for the region with lowest values, not just the single best cell.
- **Efficiency plots** help evaluate whether a marginal quality gain justifies the inference cost increase (tokens/sec, memory).
- If results across ranks are nearly identical, the task may be too simple to discriminate — consider a more challenging dataset or evaluation.

## Resuming interrupted runs

The runner persists progress to `experiment_progress.json`. Re-running the same command skips completed experiments automatically.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Tinker Docs: Sweep Case Study](https://tinker-docs.thinkingmachines.ai/supervised-learning/sweep-case-study)
