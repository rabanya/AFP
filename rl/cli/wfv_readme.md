# Walk-Forward Validation with Grid Search

This module implements comprehensive walk-forward validation with hyperparameter grid search for the portfolio management reinforcement learning system.

## Overview

Walk-forward validation is a robust backtesting methodology that simulates real-world trading by training models on historical data and validating on future unseen data. This implementation combines walk-forward validation with extensive hyperparameter optimization to find optimal model configurations.

## Core Methodology

### Walk-Forward Validation Principle

**Traditional Backtesting Problems:**
- **Look-ahead bias**: Using future information to predict past
- **Overfitting**: Optimizing parameters on the entire dataset
- **Static training**: Not accounting for changing market conditions

**Walk-Forward Solution:**
- **Time-ordered validation**: Train on past, validate on future
- **Rolling windows**: Gradually expanding training sets
- **Out-of-sample testing**: Each validation period is truly unseen

### Grid Search Integration

The system performs exhaustive hyperparameter optimization across multiple model architectures and training configurations simultaneously with walk-forward validation.

## Configuration Parameters

### Data Configuration
```python
DATA_DIR = "rl-project/data"      # Path to processed parquet files
START_DATE = None                 # Start date filter (YYYY-MM-DD)
END_DATE = None                   # End date filter (YYYY-MM-DD)
```

### Walk-Forward Parameters
```python
INITIAL_TRAIN_YEARS = 4           # Initial training window (years)
VALIDATION_YEARS = 1              # Validation window (years)
STEP_YEARS = 2                    # Expansion step size (years)
```

### Hyperparameter Search Space

#### Model Architecture Parameters
| Parameter | Options | Description |
|-----------|---------|-------------|
| `d_model` | [32, 64, 128] | Transformer embedding dimension |
| `nhead` | [2, 4, 8] | Attention heads |
| `num_layers` | [1, 2, 3] | Transformer layers |
| `hidden` | [64, 128, 256] | Scoring network hidden size |
| `dropout` | [0.05, 0.1, 0.2] | Regularization rate |
| `activation` | ["gelu", "relu"] | Activation function |
| `norm_first` | [True, False] | Pre/Post-LN architecture |
| `ff_hidden_mult` | [2, 4] | Feed-forward multiplier |

#### Training Parameters
| Parameter | Options | Description |
|-----------|---------|-------------|
| `epochs_per_step` | [30, 50, 100] | Training epochs per walk-forward step |
| `train_length` | [8, 12, 16] | Training episode length |
| `learning_rate` | [1e-6, 1e-5, 1e-4] | Learning rate |

### Environment Parameters
```python
RET_COL = "ret_next"              # Target return column
ID_COL = "ticker"                 # Asset identifier column
DATE_COL = "date"                 # Date column
WINDOW = 48                       # Historical window size (weeks)
MIN_OBS_IN_WINDOW = 1             # Minimum observations required
MIN_PORTFOLIO_SIZE = 3            # Minimum assets per portfolio
MAX_PORTFOLIO_SIZE = 15           # Maximum assets per portfolio
```

## Implementation Details

### 1. Parameter Grid Generation

```python
def generate_parameter_combinations():
    """Generate all combinations of hyperparameters for grid search."""
    param_grid = {
        'd_model': D_MODEL_OPTIONS,
        'nhead': NHEAD_OPTIONS,
        'num_layers': NUM_LAYERS_OPTIONS,
        # ... other parameters
    }

    # Generate Cartesian product of all parameter combinations
    combinations = list(itertools.product(*param_grid.values()))
    return [dict(zip(param_grid.keys(), combo)) for combo in combinations]
```

**Total Combinations**: 3×3×3×3×3×2×2×2×3×3×3 = **58,320 parameter combinations**

### 2. Date Range Generation

```python
def get_date_ranges(data, date_col, initial_train_years, validation_years, step_years):
    """Generate walk-forward validation date ranges."""

    ranges = []
    current_train_years = initial_train_years

    while True:
        train_start = min_date
        train_end = min_date + pd.DateOffset(years=current_train_years)
        val_start = train_end
        val_end = val_start + pd.DateOffset(years=validation_years)

        if val_end > max_date:
            break

        ranges.append((train_start, train_end, val_start, val_end))
        current_train_years += step_years

    return ranges
```

**Example Walk-Forward Schedule:**
- **Step 1**: Train 2010-2014, Validate 2014-2015
- **Step 2**: Train 2010-2016, Validate 2016-2017
- **Step 3**: Train 2010-2018, Validate 2018-2019

### 3. Data Loading and Environment Building

```python
def load_and_prepare_data(data_dir, date_col, train_start, train_end, val_start, val_end):
    """Load and prepare data for training and validation periods."""

    # Build training environment
    X_train, R_train, ids_train, dates_train, M_feat_train = build_environment(
        data_dir=data_dir,
        split="processed_weekly_panel.parquet",
        start_date=train_start,
        end_date=train_end,
        # ... other parameters
    )

    # Build validation environment
    X_val, R_val, ids_val, dates_val, M_feat_val = build_environment(
        start_date=val_start,
        end_date=val_end,
        # ... same parameters
    )

    return (X_train, R_train, M_feat_train, ids_train,
            X_val, R_val, M_feat_val, ids_val, dates_train, dates_val)
```

### 4. Main Experiment Loop

#### Grid Search Structure
```python
for param_idx, params in enumerate(param_combinations):
    print(f"Parameter combination {param_idx + 1}/{len(param_combinations)}")

    # Initialize results for this parameter set
    param_results = {
        "param_combination": param_idx + 1,
        "parameters": params,
        "walk_forward_steps": []
    }

    # Run walk-forward validation for this parameter combination
    for step, (train_start, train_end, val_start, val_end) in enumerate(date_ranges):
        # Load data for this time period
        data = load_and_prepare_data(...)

        # Initialize fresh model for each training period
        policy = TransformerPolicy(K=K, **model_params)

        # Train model
        logs = train_policy(policy, X_train, R_train, **training_params)

        # Evaluate on validation set
        results = evaluate_episode(policy, X_val, R_val, detailed=True)

        # Store results
        step_results = {
            "step": step + 1,
            "train_period": f"{train_start} to {train_end}",
            "val_period": f"{val_start} to {val_end}",
            "val_sharpe": results[4],  # Sharpe ratio
            "training_logs": logs,
            "detailed_portfolio_data": results[5]
        }

        param_results["walk_forward_steps"].append(step_results)
```

#### Model Reinitialization Strategy
```python
# Reinitialize model for each training period
_, W, K = X_train[0].shape
model_params = {k: v for k, v in params.items() if k not in training_params}
policy = TransformerPolicy(K=K, **model_params, max_len=MAX_LEN)
```

**Rationale**: Prevents knowledge leakage between time periods and ensures fair comparison of different parameter combinations.

## Output Structure

### Directory Structure
```
walk_forward_results/
├── grid_search_summary.json          # Real-time progress
├── final_summary_report.json         # Complete results
├── complete_results.json             # All detailed results
├── individual_results/               # Per-parameter results
│   ├── param_combination_001.json
│   ├── param_combination_002.json
│   └── ...
└── checkpoints/                      # Model checkpoints
    ├── param_1_step_1.pt
    ├── param_1_step_2.pt
    └── ...
```

### Results JSON Structure

#### Individual Parameter Results
```json
{
  "param_combination": 1,
  "parameters": {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "epochs_per_step": 50,
    // ... other parameters
  },
  "walk_forward_steps": [
    {
      "step": 1,
      "train_start": "2010-01-01",
      "train_end": "2014-01-01",
      "val_start": "2014-01-01",
      "val_end": "2015-01-01",
      "train_periods": 208,
      "val_periods": 52,
      "val_sharpe": 0.234,
      "final_wealth_port": 1.456,
      "final_wealth_ew": 1.123,
      "training_logs": [...],
      "detailed_portfolio_data": [...]
    }
  ],
  "summary": {
    "mean_sharpe": 0.187,
    "std_sharpe": 0.045,
    "num_steps": 3
  }
}
```

#### Final Summary Report
```json
{
  "experiment_info": {
    "total_combinations": 58320,
    "total_steps": 174960,
    "date_ranges": 3,
    "initial_train_years": 4,
    "validation_years": 1,
    "step_years": 2
  },
  "best_parameters": {
    "combination_id": 12345,
    "parameters": {...},
    "mean_sharpe": 0.312,
    "std_sharpe": 0.023,
    "num_steps": 3
  },
  "top_10_combinations": [...],
  "parameter_ranges": {...}
}
```

## Performance Metrics

### Primary Metrics
- **Excess Sharpe Ratio**: `(μ_portfolio - μ_benchmark) / σ_excess`
- **Final Wealth**: Cumulative returns over validation period
- **Stability**: Sharpe ratio standard deviation across time periods

### Secondary Metrics
- **Training Convergence**: Loss reduction over epochs
- **Portfolio Turnover**: Rebalancing frequency and magnitude
- **Asset Coverage**: Percentage of assets with non-zero weights

## Computational Considerations

### Resource Requirements
- **Memory**: ~2-8GB per parameter combination (depending on model size)
- **Storage**: ~50MB per parameter combination for checkpoints
- **Time**: ~5-15 minutes per parameter combination (GPU-dependent)

### Parallelization Strategy
```python
# Recommended: Run multiple parameter combinations in parallel
# Each combination is independent and can be distributed across GPUs/machines
```

### Checkpoint Strategy
```python
# Save checkpoints for:
# 1. Best performing models
# 2. Regular intervals for recovery
# 3. Final models from each time period
if SAVE_CHECKPOINTS:
    torch.save(policy.state_dict(), f"param_{param_idx+1}_step_{step+1}.pt")
```

## Usage Examples

### Basic Walk-Forward Validation
```python
from rl.cli.walk_forward_validation import run_walk_forward_validation

# Run complete experiment
results = run_walk_forward_validation()

# Access best parameters
best_params = results["best_parameters"]
print(f"Best Sharpe: {best_params['mean_sharpe']:.4f}")
```

### Custom Parameter Search
```python
# Modify parameter ranges at the top of the file
D_MODEL_OPTIONS = [64, 128, 256]      # Larger models
EPOCHS_PER_STEP_OPTIONS = [100, 200]  # More training
LEARNING_RATE_OPTIONS = [1e-5, 5e-5]  # Finer learning rates
```

### Analyzing Results
```python
import json

# Load final results
with open("walk_forward_results/final_summary_report.json", "r") as f:
    results = json.load(f)

# Print top 5 parameter combinations
for i, combo in enumerate(results["top_10_combinations"][:5]):
    print(f"{i+1}. Sharpe: {combo['mean_sharpe']:.4f}, "
          f"Std: {combo['std_sharpe']:.4f}")
```

## Best Practices

### Experimental Design
1. **Start Small**: Begin with reduced parameter ranges for initial testing
2. **Monitor Resources**: Track memory usage and disk space
3. **Early Stopping**: Implement convergence checks to avoid wasted computation
4. **Result Monitoring**: Check intermediate results to identify parameter ranges

### Validation Robustness
1. **Multiple Seeds**: Run experiments with different random seeds
2. **Cross-Validation**: Consider k-fold cross-validation within time periods
3. **Statistical Significance**: Test if Sharpe improvements are significant
4. **Economic Relevance**: Evaluate if Sharpe gains justify implementation costs

### Production Considerations
1. **Model Selection**: Choose parameter combinations robust across time periods
2. **Overfitting Checks**: Validate on held-out future data
3. **Transaction Costs**: Include realistic trading costs in evaluation
4. **Risk Management**: Implement position limits and stop-loss rules

## Troubleshooting

### Common Issues

#### Memory Errors
```python
# Reduce batch size or model size
D_MODEL_OPTIONS = [32, 64]      # Smaller models
HIDDEN_OPTIONS = [64, 128]      # Smaller hidden layers
```

#### Slow Training
```python
# Optimize computational settings
EPOCHS_PER_STEP_OPTIONS = [20, 30]  # Fewer epochs
TRAIN_LENGTH_OPTIONS = [6, 8]      # Shorter episodes
```

#### Poor Validation Performance
```python
# Check for overfitting
DROPOUT_OPTIONS = [0.1, 0.2, 0.3]  # More regularization
LEARNING_RATE_OPTIONS = [1e-5, 5e-5]  # Lower learning rates
```

## Dependencies

- `torch>=1.9.0`: PyTorch for model training
- `numpy>=1.21.0`: Numerical computations
- `pandas>=1.3.0`: Data manipulation
- `pathlib`: File system operations
- `json`: Results serialization
- `itertools`: Parameter combination generation

## Integration with Other Modules

This script orchestrates the entire experimental pipeline by integrating:

- **Data Loading**: `rl.envs.build_environment()`
- **Model Architecture**: `rl.models.TransformerPolicy`
- **Training Algorithm**: `rl.algo.train_policy()`
- **Evaluation**: `rl.algo.evaluate_episode()`
- **Reward Function**: `rl.models.rewards.RewardFunction`

## References

- **Walk-Forward Analysis**: Robust method for simulating real trading
- **Grid Search**: Exhaustive hyperparameter optimization
- **Time Series Cross-Validation**: Preventing temporal data leakage
- **Financial Risk Management**: Sharpe ratio and wealth preservation</contents>
</xai:function_call">rl/cli/wfv_readme.md