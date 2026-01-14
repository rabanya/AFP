# Reinforcement Learning for Portfolio Management

A minimal RL package for portfolio management experiments using transformer-based policies to maximize excess Sharpe ratio.

## Project Structure

### Root Directory
- **`pyproject.toml`**: Project configuration file defining dependencies, scripts, and build settings
- **`eda.ipynb`**: Exploratory data analysis notebook for data preprocessing and feature engineering

### `rl/` - Main Package

#### `algo/` - Training and Evaluation Algorithms
- **`__init__.py`**: Exports `train_policy` and `evaluate_episode` functions
- **`trainer.py`**: Contains the `train_policy()` function that trains RL policies using excess Sharpe ratio maximization
  - `train_policy(policy, X_train, R_train, epochs, length, lr, device, seed, M_feat_list, risk_free_rate)`: Main training loop with gradient descent and learning rate scheduling
- **`evaluator.py`**: Contains the `evaluate_episode()` function for policy evaluation
  - `evaluate_episode(policy, X_val, R_val, t0, length, device, M_feat_list, risk_free_rate, ids_list, dates_list, detailed)`: Evaluates policy performance and returns Sharpe ratio, wealth curves, and detailed portfolio data

#### `cli/` - Command Line Interfaces
- **`walk_forward_validation.py`**: Comprehensive walk-forward validation script with grid search hyperparameter tuning
  - `generate_parameter_combinations()`: Creates grid search parameter combinations
  - `get_date_ranges(data, date_col, initial_train_years, validation_years, step_years)`: Generates rolling validation date ranges
  - `load_and_prepare_data(data_dir, date_col, train_start, train_end, val_start, val_end)`: Loads and preprocesses data for training/validation periods
  - `run_walk_forward_validation()`: Main function orchestrating the complete walk-forward validation process

#### `configs/` - Configuration Files
- **`env.yaml`**: Environment configuration specifying data columns and factors
- **`model.yaml`**: Model architecture hyperparameters
- **`train.yaml`**: Training configuration (epochs, learning rate, device, etc.)
- **`eval.yaml`**: Evaluation configuration and metrics

#### `envs/` - Environment Builders
- **`__init__.py`**: Exports `build_environment` and `build_environment_from_df` functions
- **`stock_env.py`**: Stock market environment construction utilities
  - `build_environment_from_df(df, factor_cols, ret_col, id_col, date_col, window, min_obs_in_window, stochastic, rebalance_prob, max_change, min_portfolio_size, max_portfolio_size)`: Builds environment from pandas DataFrame with optional stochastic portfolio rebalancing
  - `build_environment(data_dir, split, ret_col, id_col, date_col, window, min_obs_in_window, start_date, end_date, stochastic, rebalance_prob, max_change, min_portfolio_size, max_portfolio_size)`: Builds environment from parquet data files

#### `models/` - Neural Network Models
- **`__init__.py`**: Exports `TransformerPolicy` class
- **`rewards.py`**: Reward function implementations
  - `RewardFunction` class: Computes excess Sharpe ratio rewards
    - `__init__(self, risk_free_rate)`: Initialize with risk-free rate
    - `compute(self, portfolio_returns, equal_weight_returns, info)`: Calculate excess Sharpe ratio
- **`transformer_policy.py`**: Transformer-based portfolio policy
  - `PositionalEncoding` class: Sinusoidal positional encoding for transformer models
    - `__init__(self, d_model, max_len)`: Initialize positional encoding
    - `forward(self, x)`: Add positional encoding to input
  - `TransformerPolicy` class: Main policy model using transformer encoder and cross-attention
    - `__init__(self, K, d_model, nhead, num_layers, hidden, dropout, activation, norm_first, ff_hidden_mult, max_len)`: Initialize transformer policy
    - `forward(self, X_t, M_feat)`: Forward pass generating portfolio weights

#### `utils/` - Utility Functions
- **`__init__.py`**: Exports data processing, loading, and seeding utilities
- **`data_loader.py`**: Parquet data loading utilities
  - `load_parquet_data(data_dir, split, start_date, end_date)`: Load and filter parquet data files
- **`data_processing.py`**: Data preprocessing utilities
  - `winsorize_cs(x, p)`: Winsorize series at quantile thresholds
  - `zscore_cs(x)`: Z-score normalize a series
- **`seeding.py`**: Random seeding utilities for reproducibility
  - `set_seed(seed)`: Set random seeds for Python, NumPy, and PyTorch

### `tests/` - Unit Tests
- **`unit/test_env.py`**: Environment building tests
  - `test_build_environment()`: Tests environment construction with synthetic data
- **`unit/test_models.py`**: Model forward pass tests
  - `test_transformer_policy_forward()`: Tests policy forward pass shapes and constraints
  - `test_transformer_policy_different_sizes()`: Tests policy with varying input sizes
- **`unit/test_per_factor_masks.py`**: Per-factor masking functionality tests
  - `test_per_factor_masks()`: Tests per-factor mask behavior with missing data
  - `test_backward_compatibility()`: Tests compatibility with older API

## Key Features

1. **Transformer-based Portfolio Policy**: Uses self-attention and cross-attention mechanisms to process sequential factor data and generate portfolio weights

2. **Excess Sharpe Ratio Optimization**: Directly optimizes for risk-adjusted returns through reinforcement learning

3. **Per-factor Masking**: Handles missing factor data at the individual feature level rather than dropping entire observations

4. **Stochastic Portfolio Rebalancing**: Simulates realistic portfolio dynamics with probabilistic rebalancing decisions

5. **Walk-forward Validation**: Comprehensive backtesting framework with rolling validation windows and hyperparameter grid search

6. **Modular Design**: Clean separation of concerns with dedicated modules for algorithms, environments, models, and utilities

## Dependencies

- `torch>=1.9.0`: Deep learning framework
- `numpy>=1.21.0`: Numerical computing
- `pandas>=1.3.0`: Data manipulation
- `pyarrow>=10.0.0`: Parquet file support
- `gymnasium>=0.28.0`: Reinforcement learning environments
- `hydra-core>=1.2.0`: Configuration management
- `matplotlib>=3.5.0`: Plotting utilities
- `seaborn>=0.11.0`: Statistical visualization

## Installation

```bash
pip install -e .
```

## Usage

### Training
```bash
rl-train
```

### Evaluation
```bash
rl-eval
```

### Walk-forward Validation with Grid Search
```python
from rl.cli.walk_forward_validation import run_walk_forward_validation
results = run_walk_forward_validation()
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- Environment settings in `rl/configs/env.yaml`
- Model architecture in `rl/configs/model.yaml`
- Training parameters in `rl/configs/train.yaml`
- Evaluation metrics in `rl/configs/eval.yaml`

## Testing

Run unit tests with pytest:
```bash
pytest tests/
```

## Data Format

The project expects parquet-formatted panel data with columns for:
- Date information (`date`)
- Asset identifiers (`ticker`, `permno`)
- Return data (`ret_next`)
- Factor features (various financial indicators)