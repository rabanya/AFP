# Reinforcement Learning for Portfolio Management

A minimal RL package for portfolio management experiments using transformer-based policies to maximize excess Sharpe ratio.

## Project Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │ -> │  Data Processing │ -> │  Environment    │
│                 │    │                  │    │                 │
│ • CRSP (daily   │    │ • EDA Notebook   │    │ • Factor Data   │
│   prices)       │    │ • Weekly         │    │ • Return Data   │
│ • Compustat     │    │   Resampling     │    │ • Masking       │
│   (fundamentals)│    │ • Feature Eng.   │    │ • Windowing     │
│ • Fama-French   │    │ • Normalization  │    │                 │
│   (factors)     │    │                  │    └─────────────────┘
└─────────────────┘    └──────────────────┘            │
                                                       │
                                                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Model Init    │ -> │    Training      │ -> │   Evaluation    │
│                 │    │                  │    │                 │
│ • Transformer   │    │ • Policy Grad.   │    │ • Sharpe Ratio  │
│   Policy        │    │ • Excess Sharpe  │    │ • Wealth Curves │
│ • Reward Func.  │    │ • Rolling Windows│    │ • Risk Metrics  │
│ • Hyperparams   │    │ • LR Scheduling  │    │ • Backtesting   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                                                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Walk-forward    │ -> │ Hyperparameter   │ -> │   Results       │
│ Validation      │    │ Tuning           │    │                 │
│                 │    │                  │    │ • Performance   │
│ • Rolling       │    │ • Grid Search    │    │   Analysis      │
│   Windows       │    │ • Cross-         │    │ • Risk-Return   │
│ • OOS Testing   │    │   Validation     │    │   Plots         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Project Structure

### Root Directory
- **`pyproject.toml`**: Project configuration file defining dependencies, scripts, and build settings
- **`eda.ipynb`**: Exploratory data analysis notebook for data preprocessing and feature engineering
  - **Dataset Statistics**: ~300+ US stocks (AAPL, TSLA, ORCL, UBER, etc.) from 2010-2024
  - **Final Dataset**: 10,137 weekly observations after processing
  - **Data Sources**: Combined CRSP (prices/returns) and Compustat (fundamentals) databases
  - **Processing**: Daily data resampled to weekly frequency with Friday close convention

### `rl/` - Main Package

#### `algo/` - Training and Evaluation Algorithms
- **`algo_readme.md`**: Comprehensive documentation of training and evaluation algorithms
- **`__init__.py`**: Exports `train_policy` and `evaluate_episode` functions
- **`trainer.py`**: Contains the `train_policy()` function that trains RL policies using excess Sharpe ratio maximization
  - `train_policy(policy, X_train, R_train, epochs, length, lr, device, seed, M_feat_list, risk_free_rate)`: Main training loop with gradient descent and learning rate scheduling
- **`evaluator.py`**: Contains the `evaluate_episode()` function for policy evaluation
  - `evaluate_episode(policy, X_val, R_val, t0, length, device, M_feat_list, risk_free_rate, ids_list, dates_list, detailed)`: Evaluates policy performance and returns Sharpe ratio, wealth curves, and detailed portfolio data

#### `cli/` - Command Line Interfaces
- **`wfv_readme.md`**: Comprehensive documentation of walk-forward validation and grid search
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

### Environment Building System

The environment builders create reinforcement learning environments from financial time-series data, handling data preprocessing, missing values, and realistic portfolio dynamics simulation.

#### Core Functions

##### `build_environment_from_df()`
**Purpose**: Builds RL environment from pandas DataFrame with advanced portfolio simulation

**Key Features**:
- **Panel Data Processing**: Converts long-form DataFrame to time-series arrays
- **Stochastic Portfolio Selection**: Simulates realistic portfolio rebalancing decisions
- **Per-Factor Masking**: Handles missing data at individual feature level
- **Asset Filtering**: Ensures minimum historical data availability

**Parameters**:
- `df`: Input DataFrame with stock data
- `factor_cols`: List of financial factor columns to use as features
- `ret_col`: Target return column name (default: "ret_next")
- `id_col`: Asset identifier column (default: "permno")
- `date_col`: Date column name (default: "date")
- `window`: Historical window size in periods (default: 48)
- `min_obs_in_window`: Minimum observations required per asset (default: 1)

**Stochastic Portfolio Parameters**:
- `stochastic`: Enable realistic portfolio dynamics (default: False)
- `rebalance_prob`: Probability of rebalancing each period (default: 0.3)
- `max_change`: Maximum stocks to add/remove per rebalancing (default: 2)
- `min_portfolio_size`: Minimum assets in portfolio (default: 3)
- `max_portfolio_size`: Maximum assets in portfolio (default: 15)

**Returns**: `(X_train, R_train, ids_t, dates, M_feat_list)`
- `X_train`: List of feature arrays `(N_t, W, K)` - assets × window × factors
- `R_train`: List of return arrays `(N_t,)` - next-period returns
- `ids_t`: List of asset ID arrays for each time step
- `dates`: List of dates for each time step
- `M_feat_list`: List of per-factor mask arrays `(N_t, W, K)` - missing data indicators

##### `build_environment()`
**Purpose**: High-level wrapper that loads data from parquet files and builds environment

**Automatic Factor Detection**:
```python
# Automatically identifies factor columns
factor_cols = [col for col in df.columns
               if col not in [ret_col, id_col, date_col]]
```

**File-Based Loading**:
- `data_dir`: Directory containing parquet files
- `split`: Specific file to load (e.g., "processed_weekly_panel.parquet")
- `start_date`/`end_date`: Optional date range filtering

#### Data Processing Pipeline

##### 1. **Panel Data Transformation**
```python
# Convert long-form DataFrame to 3D arrays
for col in factor_cols:
    factor = df.pivot_table(index=date_col, columns=id_col,
                           values=col, aggfunc="last")
    factor_panels.append(factor.to_numpy(dtype=np.float32))

FACTORS = np.stack(factor_panels, axis=-1)  # (T, N, K)
RET = df.pivot_table(index=date_col, columns=id_col,
                    values=ret_col, aggfunc="last").to_numpy()  # (T, N)
```

##### 2. **Rolling Window Extraction**
```python
for t in range(window, T_full):
    win = FACTORS[t - window : t, :, :]  # Historical window (W, N, K)
    r_t = RET[t, :]                      # Next-period returns (N,)
```

##### 3. **Asset Filtering & Validation**
```python
# Per-factor validity masks
valid_feat = ~np.isnan(win)              # (W, N, K)
valid_ts = valid_feat.any(axis=2)       # (W, N) - time valid if any factor
real_counts = valid_ts.sum(axis=0)      # (N,) - observations per asset
enough_hist = real_counts >= min_obs_in_window

# Final asset selection
tradable = ~np.isnan(r_t)                # Must have return data
keep = tradable & enough_hist            # Must have sufficient history
```

##### 4. **Per-Factor Masking**
```python
# Preserve missing data information
M_feat_t = valid_feat[:, keep, :].transpose(1, 0, 2)  # (N_t, W, K)

# Fill NaNs for model input while tracking validity
win_filled = win.copy()
win_filled[np.isnan(win_filled)] = 0
```

#### Stochastic Portfolio Simulation

**Realistic Portfolio Dynamics**: Simulates how real portfolio managers make decisions

##### Portfolio Initialization
```python
if current_portfolio is None:
    # Random initial portfolio size (3-15 stocks)
    initial_size = np.random.randint(min_portfolio_size,
                                   min(max_portfolio_size, len(available_stocks)) + 1)
    current_portfolio = np.random.choice(available_stocks,
                                       size=initial_size, replace=False)
```

##### Rebalancing Logic
```python
if np.random.random() < rebalance_prob:  # 30% chance per period
    # Decide: add, remove, or both
    action = np.random.choice(['add', 'remove', 'both'])

    # Add stocks (up to max_change)
    if action == 'add' and len(current_portfolio) < max_portfolio_size:
        available_to_add = np.setdiff1d(available_stocks, current_portfolio)
        new_stocks = np.random.choice(available_to_add, size=num_to_add, replace=False)
        current_portfolio = np.concatenate([current_portfolio, new_stocks])

    # Remove stocks (maintaining minimum size)
    if action == 'remove' and len(current_portfolio) > min_portfolio_size:
        stocks_to_remove = np.random.choice(current_portfolio, size=num_to_remove)
        current_portfolio = np.setdiff1d(current_portfolio, stocks_to_remove)
```

**Benefits of Stochastic Simulation**:
- **Realistic Turnover**: Mimics actual portfolio management behavior
- **Diverse Training**: Exposes model to various portfolio compositions
- **Generalization**: Prevents overfitting to specific stock combinations
- **Risk Management**: Teaches model to handle portfolio changes

#### Data Format Specification

**Input Data Requirements**:
- **Panel Structure**: Long-form DataFrame with date × asset × features
- **Required Columns**: Date, asset ID, return target, factor features
- **Data Types**: Mixed (dates, strings, floats) → converted to float32 arrays
- **Missing Values**: Handled via per-factor masking, not imputation

**Output Data Format**:
- **Time Series**: List of 3D arrays for sequential processing
- **Asset Universe**: Varies by time step (filtered by data availability)
- **Feature Engineering**: Ready for transformer-based processing
- **Masking**: Per-factor validity indicators for robust handling

#### Integration with RL Pipeline

**Training Integration**:
```python
# Load environment
X_train, R_train, ids_train, dates_train, M_feat_train = build_environment(
    data_dir="data/",
    split="processed_weekly_panel.parquet",
    window=48,
    stochastic=True  # Enable realistic portfolio dynamics
)

# Train policy
logs = train_policy(policy, X_train, R_train, M_feat_list=M_feat_train)
```

**Key Advantages**:
- **End-to-End**: Raw data → RL-ready environment
- **Scalable**: Handles large datasets efficiently
- **Flexible**: Supports various data formats and factor sets
- **Robust**: Comprehensive missing data and validation handling
- **Realistic**: Stochastic simulation of portfolio management

#### `models/` - Neural Network Models
- **`model_readme.md`**: Comprehensive documentation of model architectures and training
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

#### Test Structure and Coverage
The test suite ensures reliability and correctness of the portfolio management system through comprehensive unit testing.

##### **`unit/test_env.py`** - Environment Building Tests
- **`test_build_environment()`**: Validates environment construction from synthetic data
  - Tests data transformation pipeline
  - Verifies output format correctness
  - Checks parameter handling and edge cases

##### **`unit/test_models.py`** - Neural Network Model Tests
- **`test_transformer_policy_forward()`**: Validates TransformerPolicy forward pass
  - Tests weight generation and normalization (sum to 1)
  - Verifies positive weight constraints
  - Tests per-factor masking with missing data
- **`test_transformer_policy_different_sizes()`**: Tests scalability across asset universes
  - Validates performance with 1 to 100 assets
  - Ensures consistent behavior across different portfolio sizes

##### **`unit/test_per_factor_masks.py`** - Masking Functionality Tests
- **`test_per_factor_masks()`**: Tests per-factor mask behavior
  - Validates missing data handling at feature level
  - Ensures model robustness with incomplete data
- **`test_backward_compatibility()`**: Tests API consistency
  - Ensures compatibility with older interfaces
  - Validates parameter handling changes

### `configs/` - Configuration Management

#### Hydra-Based Configuration System
The project uses Hydra for flexible, hierarchical configuration management, enabling systematic hyperparameter tuning and reproducible experiments.

##### **`env.yaml`** - Environment Configuration
```yaml
# Environment configuration
env:
  ret_col: "ret_next"           # Target return column
  id_col: "permno"              # Asset identifier column
  date_col: "date"              # Date column
  window: 12                    # Historical window size (weeks)
  min_obs_in_window: 1          # Minimum observations per asset
  factor_cols:                  # Financial factor features
    - "log_mktcap"              # Market capitalization (size)
    - "ret_lag1"                # Lagged returns (momentum)
    - "mom_12_2"                # 12-month momentum
    - "vol_12"                  # 12-month volatility
    - "turnover"                # Trading volume turnover
    - "log_dollar_vol"          # Dollar trading volume
    - "amihud"                  # Amihud illiquidity measure
    - "log_adjprc"              # Adjusted price (valuation)
```

**Factor Selection Rationale**:
- **Size**: `log_mktcap` - Small vs large company effects
- **Momentum**: `ret_lag1`, `mom_12_2` - Short and medium-term trends
- **Risk**: `vol_12` - Volatility-based risk assessment
- **Liquidity**: `turnover`, `log_dollar_vol`, `amihud` - Trading costs and market depth
- **Valuation**: `log_adjprc` - Price-based valuation signals

##### **`model.yaml`** - Model Architecture Configuration
```yaml
# Model configuration
model:
  type: "transformer"           # Model type
  K: 8                         # Number of input features
  d_model: 64                  # Transformer embedding dimension
  nhead: 4                     # Attention heads
  num_layers: 2                # Transformer layers
  hidden: 64                   # Scoring network hidden size
  use_per_factor_mask: true    # Enable per-factor masking (2*K input)
```

**Architecture Design Decisions**:
- **`d_model=64`**: Balanced complexity for financial time-series
- **`nhead=4`**: Multi-perspective factor relationship learning
- **`num_layers=2`**: Sufficient depth for temporal hierarchies
- **`use_per_factor_mask=true`**: Advanced missing data handling

##### **`train.yaml`** - Training Configuration
```yaml
# Training configuration
train:
  epochs: 30                   # Training epochs per experiment
  length: 12                   # Episode length (weeks)
  lr: 1e-4                     # Learning rate
  device: "cpu"                # Training device
  seed: 42                     # Random seed for reproducibility
  grad_clip: 1.0               # Gradient clipping threshold
```

**Training Strategy**:
- **Epochs**: Balance between convergence and overfitting
- **Episode Length**: Capture meaningful market cycles
- **Learning Rate**: Conservative for stable convergence
- **Gradient Clipping**: Prevent optimization instability

##### **`eval.yaml`** - Evaluation Configuration
```yaml
# Evaluation configuration
eval:
  block_len: 12                # Evaluation window length
  device: "cpu"                # Evaluation device
  metrics:                     # Performance metrics to compute
    - "sharpe_excess"          # Excess Sharpe ratio
    - "final_wealth_port"      # Final portfolio wealth
    - "final_wealth_ew"        # Final equal-weight wealth
```

**Evaluation Framework**:
- **Out-of-Sample Testing**: Validates on unseen future data
- **Benchmark Comparison**: Always compares against equal-weighted
- **Multiple Metrics**: Risk-adjusted and absolute performance measures

### `utils/` - Utility Functions

#### Data Processing and Loading Utilities
Core utilities for data handling, preprocessing, and reproducibility in the portfolio management pipeline.

##### **`data_loader.py`** - Parquet Data Loading
```python
def load_parquet_data(
    data_dir: str,
    split: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
```

**Key Features**:
- **Flexible File Loading**: Supports direct files or directory-based splits
- **Date Range Filtering**: Optional temporal subsetting for experiments
- **Automatic Date Parsing**: Ensures datetime format for time-series operations
- **Error Handling**: Clear error messages for missing files

**Usage Patterns**:
```python
# Load specific file
df = load_parquet_data("data/", "processed_weekly_panel.parquet")

# Load with date filtering
df = load_parquet_data("data/", "train.parquet",
                      start_date="2010-01-01", end_date="2018-12-31")
```

##### **`data_processing.py`** - Data Preprocessing Utilities

###### `winsorize_cs(x, p=0.01)` - Outlier Handling
```python
def winsorize_cs(x: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsorize series at p and (1-p) quantiles."""
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lo, hi)
```

**Purpose**: Mitigates outlier impact on model training
- **Default 1%/99%**: Removes extreme values while preserving distribution
- **Configurable**: Adjustable quantile thresholds
- **Cross-Sectional**: Applied consistently across assets

###### `zscore_cs(x)` - Standardization
```python
def zscore_cs(x: pd.Series) -> pd.Series:
    """Z-score normalize series: (x - mean) / std."""
    return (x - x.mean()) / (x.std(ddof=0) + 1e-8)
```

**Purpose**: Standardizes features for neural network training
- **Zero Mean, Unit Variance**: Standard normal distribution
- **Numerical Stability**: Small epsilon prevents division by zero
- **Cross-Sectional**: Applied per time period across assets

##### **`seeding.py`** - Reproducibility Utilities

###### `set_seed(seed)` - Global Seed Setting
```python
def set_seed(seed: int) -> None:
    """Set random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```

**Comprehensive Seeding**:
- **Python random**: Standard library randomization
- **NumPy**: Array operations and sampling
- **PyTorch CPU**: Neural network operations
- **PyTorch GPU**: CUDA operations (if available)
- **Multi-GPU**: All available CUDA devices

**Reproducibility Guarantee**:
- **Deterministic Training**: Same results across runs
- **Experiment Consistency**: Enables fair comparison of configurations
- **Debugging Support**: Reproducible error conditions

#### Integration with Main Pipeline

**Data Loading → Processing → Training**:
```python
# 1. Load and filter data
df = load_parquet_data("data/", "train.parquet",
                      start_date="2010-01-01")

# 2. Preprocess features
df['winsorized_factor'] = df.groupby('date')['factor'].transform(
    lambda x: winsorize_cs(x, p=0.01)
)
df['standardized_factor'] = df.groupby('date')['factor'].transform(zscore_cs)

# 3. Ensure reproducibility
set_seed(42)

# 4. Train model
# ... training pipeline continues
```

#### Utility Design Principles

1. **Modularity**: Each function has single responsibility
2. **Robustness**: Comprehensive error handling and edge cases
3. **Performance**: Efficient operations on large financial datasets
4. **Consistency**: Standardized interfaces across the codebase
5. **Configurability**: Flexible parameters for different use cases

## Data Dictionary

The dataset combines CRSP and Compustat data with 40+ features covering fundamentals, prices, ratios, and risk factors:

### Core Identifiers & Time
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `date` | Date | datetime64 | Trading date | CRSP |
| `ticker` | Ticker Symbol | string | Stock ticker symbol | CRSP |
| `permno` | PERMNO | int64 | Permanent stock identifier | CRSP |
| `datadate` | Data Date | datetime64 | Fundamental data date | Compustat |
| `effective_date` | Effective Date | datetime64 | Data effective date | Compustat |

### Price & Volume Data
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `open` | Open Price | float64 | Opening price | CRSP |
| `high` | High Price | float64 | Daily high price | CRSP |
| `low` | Low Price | float64 | Daily low price | CRSP |
| `adj_close` | Adjusted Close | float64 | Split/dividend adjusted price | CRSP |
| `prcc` | Close Price | float64 | Closing price | CRSP |
| `volume` | Volume | float64 | Trading volume | CRSP |

### Return Data
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `ret` | Return | float64 | Stock return | CRSP |
| `excess_ret` | Excess Return | float64 | Return minus risk-free rate | Calculated |
| `log_ret_1d` | Log Return | float64 | Daily logarithmic return | Calculated |
| `ret_next` | Next Period Return | float64 | Target return for prediction | Calculated |

### Fundamental Data
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `at` | Total Assets | float64 | Total assets (book value) | Compustat |
| `ceq` | Common Equity | float64 | Common/ordinary equity | Compustat |
| `revt` | Revenue | float64 | Total revenue | Compustat |
| `oiadpq` | Operating Income Qtr | float64 | Operating income (quarterly) | Compustat |
| `ibq` | Income Before Extra Qtr | float64 | Income before extraordinary items | Compustat |
| `oancf` | Operating Cash Flow | float64 | Net cash from operations | Compustat |
| `capx` | Capital Expenditures | float64 | Capital expenditures | Compustat |
| `dltt` | Long-Term Debt | float64 | Long-term debt | Compustat |
| `lt` | Total Liabilities | float64 | Total liabilities | Compustat |
| `che` | Cash & Equivalents | float64 | Cash and short-term investments | Compustat |
| `csho` | Common Shares Out | float64 | Common shares outstanding | Compustat |

### Calculated Ratios
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `book_to_price` | Book-to-Price | float64 | Book equity / Market cap | Calculated |
| `earnings_to_price` | E/P Ratio | float64 | Earnings / Price | Calculated |
| `sales_to_price` | S/P Ratio | float64 | Revenue / Price | Calculated |
| `cf_to_price` | CF/P Ratio | float64 | Cash flow / Price | Calculated |
| `price_to_book` | P/B Ratio | float64 | Price / Book equity | Calculated |

### Market Data
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `shares_outstanding` | Shares Outstanding | float64 | Total shares outstanding | Calculated |
| `mkt_cap` | Market Cap | float64 | Market capitalization | Calculated |
| `mkt_cap_eow` | End-of-Week Market Cap | float64 | End-of-week market cap | Calculated |

### Fama-French Factors
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `mkt_rf` | Market Risk Premium | float64 | Market return - risk-free rate | Fama-French |
| `rf` | Risk-Free Rate | float64 | Risk-free rate (T-bill) | Fama-French |
| `smb` | Small Minus Big | float64 | Size factor | Fama-French |
| `hml` | High Minus Low | float64 | Value factor | Fama-French |
| `umd` | Up Minus Down | float64 | Momentum factor | Carhart |

### Normalized Factors
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `mkt_rf_znorm_w504` | Market Factor Z-norm | float64 | Z-score normalized market factor | Processed |
| `smb_znorm_w504` | SMB Z-norm | float64 | Z-score normalized size factor | Processed |
| `hml_znorm_w504` | HML Z-norm | float64 | Z-score normalized value factor | Processed |
| `umd_znorm_w504` | UMD Z-norm | float64 | Z-score normalized momentum | Processed |

### Risk Model Estimates
| Column | Full Name | Data Type | Description | Source |
|--------|-----------|-----------|-------------|--------|
| `alpha_cs` | Cross-sectional Alpha | float64 | Stock-specific alpha | Estimated |
| `beta_mkt_rf` | Market Beta | float64 | Beta to market factor | Estimated |
| `beta_smb` | Size Beta | float64 | Beta to size factor | Estimated |
| `beta_hml` | Value Beta | float64 | Beta to value factor | Estimated |
| `beta_umd` | Momentum Beta | float64 | Beta to momentum factor | Estimated |

### Data Processing Notes
- **Original Frequency**: Daily data (2010-2024)
- **Final Frequency**: Weekly (Friday close aggregation)
- **Companies**: ~300+ US stocks with continuous trading history
- **Fundamental Handling**: Forward-filled within tickers to maintain "as-of" values
- **Return Calculation**: Compounded simple returns over weekly periods
- **Missing Data**: Handled via per-factor masking in the RL framework

## Key Features

1. **Transformer-based Portfolio Policy**: Uses self-attention and cross-attention mechanisms to process sequential factor data and generate portfolio weights (see `rl/models/model_readme.md`)

2. **Excess Sharpe Ratio Optimization**: Directly optimizes for risk-adjusted returns through reinforcement learning

3. **Per-factor Masking**: Handles missing factor data at the individual feature level rather than dropping entire observations

4. **Stochastic Portfolio Rebalancing**: Simulates realistic portfolio dynamics with probabilistic rebalancing decisions

5. **Walk-forward Validation**: Comprehensive backtesting framework with rolling validation windows and hyperparameter grid search

6. **Modular Design**: Clean separation of concerns with dedicated modules for algorithms, environments, models, and utilities

## How the Transformer Model Makes Portfolio Decisions

### Factor-Driven Decision Making

The transformer model learns to allocate portfolio weights based on **8 key financial factors** that capture different aspects of stock performance:

| Factor | Description | Decision Influence |
|--------|-------------|-------------------|
| `log_mktcap` | Log market capitalization | Size-based allocation preferences |
| `ret_lag1` | Previous period return | Short-term momentum signals |
| `mom_12_2` | 12-month momentum | Medium-term trend strength |
| `vol_12` | 12-month volatility | Risk assessment and diversification |
| `turnover` | Trading volume turnover | Liquidity considerations |
| `log_dollar_vol` | Log dollar trading volume | Market impact assessment |
| `amihud` | Amihud illiquidity measure | Trading cost estimation |
| `log_adjprc` | Log adjusted price | Valuation and size signals |

### Learning Factor Relationships

**The model learns factor importance automatically** through attention mechanisms:

```python
# Multi-head attention discovers which factors matter most
attention_weights = softmax(Q @ K.T / sqrt(d_k))

# No hardcoded weights - transformer learns optimal combinations
# Different factor combinations work for different market conditions
```

**Example Decision Process**:
For a portfolio of AAPL, NVDA, and TSLA, the model might learn:
- **NVDA**: High weight when momentum factor shows strong upward trends
- **AAPL**: Balanced weight when volatility is moderate and liquidity is high
- **TSLA**: Lower weight when illiquidity measures indicate high trading costs

### Asset Allocation Mechanism

#### 1. **Sequential Processing**
```python
# Process 12-week historical sequences for each asset
X_t = historical_factors  # (N_assets, 12_weeks, 8_factors)
M_feat = missing_data_masks  # (N_assets, 12_weeks, 8_factors)
```

#### 2. **Cross-Asset Comparison**
```python
# Single portfolio query attends to all assets simultaneously
KV = asset_embeddings  # All assets as keys/values
Q = portfolio_query    # Single query representing portfolio perspective
attention_scores = cross_attention(Q, KV, KV)
```

#### 3. **Weight Generation**
```python
# Raw scores → probability distribution → portfolio weights
logits = asset_scores - max(asset_scores)  # Numerical stability
weights = softmax(logits)  # Sum to 1, all positive
```

### Decision Criteria Evolution

**The model discovers decision patterns through training**:
- **Momentum regimes**: Prioritizes momentum factors when trends are strong
- **High volatility periods**: Emphasizes low-volatility, liquid assets
- **Market stress**: Favors large-cap, stable companies
- **Recovery phases**: Identifies early momentum leaders

### Real-World Example

**Portfolio Allocation Scenario** (Hypothetical 2015-2020 S&P 500 data):

| Asset | Momentum (12-mo) | Volatility (12-mo) | Liquidity (Amihud) | Model Weight |
|-------|------------------|-------------------|-------------------|--------------|
| NVDA | +45% | 35% | 0.003 | **25%** (Strong momentum, manageable risk) |
| TSLA | +120% | 65% | 0.008 | **15%** (High momentum but expensive to trade) |
| AAPL | +15% | 25% | 0.001 | **35%** (Stable performer, low trading costs) |
| MSFT | +8% | 20% | 0.002 | **25%** (Consistent, liquid) |

**Decision Rationale**:
- **AAPL gets highest weight**: Balanced fundamentals, low volatility, excellent liquidity
- **NVDA gets significant allocation**: Strong momentum justifies risk
- **TSLA gets moderate weight**: High momentum offset by high trading costs and volatility
- **MSFT provides stability**: Consistent performance anchors the portfolio

### Factor Weight Learning

**The transformer learns context-dependent factor importance**:
- **Bull markets**: Momentum factors dominate (60% attention weight)
- **Bear markets**: Volatility and liquidity factors increase (40% attention weight)
- **High uncertainty**: Size factors gain importance for stability

### Continuous Adaptation

**Unlike static factor models, the transformer adapts its decision criteria**:
- **Learns new factor combinations** as market dynamics change
- **Discovers non-obvious interactions** between factors
- **Adjusts sensitivity** based on factor reliability and market conditions

This adaptive, data-driven approach allows the model to discover sophisticated allocation strategies that go beyond traditional factor investing paradigms.

## Restrictions and Limitations

While the transformer-based portfolio optimization approach offers sophisticated pattern recognition capabilities, there are important restrictions and limitations to consider, particularly regarding Sharpe ratio optimization:

### Sharpe Ratio as a Reward Function

**Sharpe ratio is inherently backward-looking**: It measures historical risk-adjusted performance but provides no guarantee of future outcomes. The model optimizes for past Sharpe ratios, assuming these predict future Sharpe ratios—a relationship that may not hold reliably.

**Potential issues with this approach**:
- **Overfitting to historical patterns**: The model might learn factor relationships that worked in the past but break down in different market conditions
- **Volatility chasing**: Optimization may favor portfolios with high volatility (providing higher potential Sharpe) even when such volatility represents genuine risk
- **Neglect of downside risk**: Sharpe ratio penalizes all volatility equally, but investors typically care more about downside losses than upside volatility
- **Transaction cost ignorance**: Frequent rebalancing to optimize historical Sharpe may incur significant trading costs not reflected in the reward
- **Maximum drawdown blindness**: A portfolio could achieve high Sharpe ratio despite catastrophic losses that make it unusable in practice

### Alternative Reward Functions

Consider these improved reward formulations for more robust portfolio optimization:

```python
# Sortino ratio (downside deviation only)
sortino = excess_returns.mean() / excess_returns[excess_returns < 0].std()

# Calmar ratio (return vs maximum drawdown)
calmar = excess_returns.mean() / abs(max_drawdown)

# Multi-objective with constraints
reward = w1 * sharpe + w2 / max_drawdown + w3 * (1 - turnover_penalty)
```

### Risk Management Enhancements

To address these limitations, consider adding:
- **Position size limits**: Maximum allocation per asset
- **Turnover penalties**: Transaction cost awareness
- **Drawdown constraints**: Maximum acceptable loss limits
- **Diversification requirements**: Minimum number of assets
- **Stress testing**: Performance evaluation under extreme market conditions

## Reward Function Design

### Advanced Reward Function Designs

Beyond the basic Sharpe ratio, modern portfolio optimization requires sophisticated reward functions that capture growth, risk, and leverage considerations. Here are research-backed approaches:

#### Multi-Objective Utility Functions

```python
def advanced_portfolio_reward(portfolio_returns, benchmark_returns, transaction_costs=None, leverage=None):
    """
    Multi-objective reward function capturing growth, risk, and leverage
    """
    # 1. Growth Component - Sortino Ratio (downside-focused)
    excess_returns = portfolio_returns - benchmark_returns
    downside_returns = excess_returns[excess_returns < 0]
    sortino_ratio = excess_returns.mean() / (downside_returns.std() + 1e-8)

    # 2. Risk Component - Maximum Drawdown Penalty
    cumulative = torch.cumprod(1 + portfolio_returns, dim=0)
    running_max = torch.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # 3. Leverage Component - Position Concentration Penalty
    if leverage is not None:
        concentration_penalty = torch.mean(leverage ** 2)  # Penalize concentrated positions
    else:
        concentration_penalty = 0

    # 4. Transaction Cost Component
    if transaction_costs is not None:
        turnover_penalty = transaction_costs.mean()
    else:
        turnover_penalty = 0

    # Multi-objective weighted combination
    w_growth, w_risk, w_leverage, w_costs = 0.4, 0.3, 0.2, 0.1

    reward = (w_growth * sortino_ratio +
              w_risk * (1 / (1 + abs(max_drawdown))) +  # Higher reward for lower drawdown
              w_leverage * (1 / (1 + concentration_penalty)) +  # Penalize concentration
              w_costs * (1 / (1 + turnover_penalty)))  # Penalize high costs

    return reward
```

#### Utility-Based Reward Functions

Drawing from behavioral finance and prospect theory:

```python
def prospect_theory_reward(portfolio_returns, benchmark_returns, lambda_gain=2.25, lambda_loss=2.25):
    """
    Prospect theory-based reward function
    - Concave for gains (risk aversion)
    - Convex for losses (risk seeking)
    """
    excess_returns = portfolio_returns - benchmark_returns

    # Value function from prospect theory
    gains = excess_returns[excess_returns >= 0]
    losses = excess_returns[excess_returns < 0]

    value = (gains ** 0.88).sum() - lambda_loss * (abs(losses) ** 0.88).sum()

    # Normalize by time period
    reward = value / len(portfolio_returns)

    return reward
```

#### Risk-Adjusted Growth Measures

```python
def omega_ratio_reward(portfolio_returns, benchmark_returns, threshold=0.0):
    """
    Omega ratio - probability-weighted ratio of gains vs losses
    """
    excess_returns = portfolio_returns - benchmark_returns

    gains = excess_returns[excess_returns > threshold]
    losses = excess_returns[excess_returns <= threshold]

    omega = gains.sum() / (abs(losses.sum()) + 1e-8)

    return omega
```

### Mathematical Optimization of Reward Functions

#### Bayesian Optimization Approach

```python
# Use Bayesian optimization to find optimal reward function weights
from skopt import gp_minimize
from skopt.space import Real

def optimize_reward_weights(historical_data, validation_data):
    """
    Optimize reward function weights using Bayesian optimization
    """

    def objective(weights):
        w_growth, w_risk, w_leverage, w_costs = weights

        # Train policy with these weights
        policy = train_with_weights(historical_data, weights)

        # Evaluate on validation data
        performance = evaluate_policy(policy, validation_data)

        # Return negative Sharpe (minimization problem)
        return -performance['sharpe']

    # Search space
    space = [
        Real(0.1, 0.8, name='w_growth'),
        Real(0.1, 0.8, name='w_risk'),
        Real(0.0, 0.5, name='w_leverage'),
        Real(0.0, 0.3, name='w_costs')
    ]

    # Constraint: weights must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]

    result = gp_minimize(objective, space, constraints=constraints, n_calls=50)

    return result.x
```

#### Reinforcement Learning for Reward Learning

```python
def learn_optimal_reward(historical_data, preference_data):
    """
    Learn reward function from human preferences or expert demonstrations
    """

    class RewardLearner(nn.Module):
        def __init__(self):
            super().__init__()
            self.reward_net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, portfolio_state):
            return self.reward_net(portfolio_state)

    # Use preference learning or inverse RL
    # Compare trajectory pairs and learn reward function
    # that explains expert preferences

    reward_learner = RewardLearner()
    optimizer = torch.optim.Adam(reward_learner.parameters())

    for preference_pair in preference_data:
        traj1, traj2 = preference_pair['preferred'], preference_pair['non_preferred']

        reward1 = sum(reward_learner(state) for state in traj1)
        reward2 = sum(reward_learner(state) for state in traj2)

        # Preference loss: preferred trajectory should have higher reward
        loss = torch.relu(reward2 - reward1 + margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Statistical Portfolio Optimization

```python
def statistical_reward_optimization(returns_data, risk_factors):
    """
    Use statistical methods to find optimal reward function parameters
    """

    # Estimate risk preferences from historical data
    mu = returns_data.mean(axis=0)  # Expected returns
    Sigma = returns_data.cov()       # Covariance matrix

    # Use Black-Litterman model or similar to incorporate views
    # Solve for optimal weights that maximize expected utility

    def utility_function(weights, risk_aversion=2.0):
        portfolio_return = weights @ mu
        portfolio_risk = weights @ Sigma @ weights.T
        return portfolio_return - risk_aversion * portfolio_risk

    # Optimize weights
    from scipy.optimize import minimize
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    bounds = [(0, 1) for _ in range(len(mu))]

    optimal_weights = minimize(
        lambda w: -utility_function(w),
        x0=np.ones(len(mu))/len(mu),
        constraints=constraints,
        bounds=bounds
    )

    return optimal_weights.x
```

### Potential Problems with Complex Reward Functions

#### Optimization Challenges
- **Multi-objective trade-offs**: Hard to balance competing objectives
- **Local optima**: Complex reward surfaces create optimization difficulties
- **Credit assignment**: Hard to attribute outcomes to specific decisions
- **Sparse rewards**: Complex functions may provide weak learning signals

#### Overfitting Risks
- **Data mining bias**: Optimizing too many parameters on historical data
- **Parameter sensitivity**: Small changes in weights can cause large behavioral changes
- **Market regime dependency**: Optimal weights may not generalize across market conditions

#### Computational Complexity
- **Training time**: More complex rewards require more training iterations
- **Numerical instability**: Complex combinations can cause gradient issues
- **Debugging difficulty**: Hard to diagnose why a complex reward function fails

#### Interpretability Loss
- **Black box rewards**: Hard to understand what the agent is optimizing
- **Counterintuitive behavior**: Agent may find unexpected ways to maximize complex rewards
- **Validation challenges**: Hard to verify if complex rewards align with investment goals

### Solutions and Best Practices

#### Hierarchical Reward Design
```python
class HierarchicalReward:
    def __init__(self):
        self.base_rewards = {
            'growth': SortinoReward(),
            'risk': DrawdownPenalty(),
            'costs': TransactionCostPenalty()
        }
        self.weights = {'growth': 0.5, 'risk': 0.3, 'costs': 0.2}

    def compute(self, trajectory):
        rewards = {}
        for name, reward_fn in self.base_rewards.items():
            rewards[name] = reward_fn.compute(trajectory)

        # Combine with learned weights
        total_reward = sum(self.weights[name] * rewards[name] for name in self.weights)

        return total_reward, rewards  # Return both total and components
```

#### Curriculum Learning
```python
# Start simple, add complexity gradually
curriculum_stages = [
    {'reward': 'sharpe_only', 'epochs': 50},
    {'reward': 'sharpe + drawdown', 'epochs': 100},
    {'reward': 'full_multi_objective', 'epochs': 200}
]

for stage in curriculum_stages:
    # Train with increasingly complex reward function
    policy = train_with_reward_type(stage['reward'], epochs=stage['epochs'])
```

#### Regularization and Constraints
```python
def regularized_reward(portfolio_returns, complexity_penalty=0.01):
    """
    Add regularization to prevent overfitting to complex rewards
    """
    base_reward = multi_objective_reward(portfolio_returns)

    # Add entropy regularization to encourage exploration
    entropy_bonus = entropy_of_portfolio_weights(portfolio_returns)

    # Add complexity penalty based on number of active constraints
    complexity_cost = complexity_penalty * count_active_constraints()

    return base_reward + entropy_bonus - complexity_cost
```

#### Validation and Monitoring
```python
def validate_reward_function(reward_fn, validation_data, out_of_sample_data):
    """
    Comprehensive validation of reward function design
    """

    # 1. In-sample performance
    in_sample_perf = evaluate_reward(reward_fn, validation_data)

    # 2. Out-of-sample robustness
    oos_perf = evaluate_reward(reward_fn, out_of_sample_data)

    # 3. Sensitivity analysis
    sensitivities = test_parameter_sensitivity(reward_fn, validation_data)

    # 4. Behavioral analysis
    behaviors = analyze_portfolio_behavior(reward_fn, validation_data)

    return {
        'in_sample': in_sample_perf,
        'out_of_sample': oos_perf,
        'sensitivity': sensitivities,
        'behavior': behaviors
    }
```

### Implementation Recommendations

#### Start Simple, Add Complexity Gradually
```python
# Phase 1: Single objective
reward = excess_sharpe_ratio

# Phase 2: Add risk management
reward = 0.7 * excess_sharpe_ratio + 0.3 * (1 / max_drawdown)

# Phase 3: Full specification
reward = (0.4 * sortino_ratio +
          0.3 * (1 / max_drawdown) +
          0.2 * (1 / concentration) +
          0.1 * (1 / transaction_costs))
```

#### Use Domain Knowledge
- **Financial theory**: Base rewards on established portfolio theory
- **Risk preferences**: Align with institutional investor requirements
- **Practical constraints**: Include real-world trading limitations

#### Continuous Monitoring and Adaptation
```python
# Regularly re-evaluate reward function performance
# Adjust weights based on market conditions
# Monitor for reward gaming behaviors
# Update based on new research and market developments
```

### Mathematical Foundation for Optimal Reward Design

The optimal reward function can be derived from **utility theory** and **stochastic control**:

#### Expected Utility Maximization
```
U(w) = E[R(w)] - λ * Var[R(w)] + γ * Skew[R(w)] - δ * Kurt[R(w)]
```

Where:
- `R(w)`: Portfolio returns given weights `w`
- `λ`: Risk aversion parameter
- `γ`: Preference for positive skewness
- `δ`: Penalty for extreme events

#### Dynamic Programming Solution
```python
# Bellman equation for optimal portfolio policy
V_t(s) = max_a [ r(s,a) + γ * E[V_{t+1}(s') | s,a] ]

# With reward function design
r(s,a) = utility(portfolio_return) - transaction_costs - risk_penalties
```

#### Reinforcement Learning for Reward Learning
```python
# Use inverse RL or preference learning
# Learn reward function that explains expert behavior
# Iteratively refine based on performance feedback
```

The key insight is that **reward function design is an iterative process** requiring domain expertise, mathematical rigor, and empirical validation. Start with theory-grounded simple rewards, then gradually add complexity while continuously validating performance.

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