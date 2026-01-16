# Portfolio Management Models

This directory contains the neural network models and reward functions for the reinforcement learning portfolio management system.

## Architecture Overview

The portfolio management system uses a transformer-based policy network that processes sequential financial factor data to generate optimal portfolio weights through reinforcement learning.

## Transformer Policy Architecture

The `TransformerPolicy` implements a sophisticated neural network for portfolio weight generation using transformer architecture with the following key components:

### Input Processing
- **Feature Projection**: Linear layer maps concatenated features + per-factor masks to transformer dimension (2K → d_model)
- **Missing Data Handling**: Per-factor masking allows fine-grained handling of missing values at individual feature level

### Temporal Sequence Processing
- **Positional Encoding**: Sinusoidal encoding adds temporal awareness to sequential factor data
- **Transformer Encoder**: Multi-head self-attention captures interdependencies between factors across time
- **Layer Architecture**: Configurable depth (num_layers), attention heads (nhead), and feed-forward dimensions

### Portfolio-Level Decision Making
- **Cross-Attention Mechanism**: Single learnable portfolio query attends over all asset representations
- **Context Integration**: Combines individual asset embeddings with global portfolio context
- **Asset Scoring**: Neural network produces raw scores for each asset's portfolio allocation

### Weight Generation
- **Softmax Normalization**: Ensures portfolio weights sum to 1
- **Temperature Scaling**: Controls concentration of weight distribution
- **Training Exploration**: Optional noise injection during training

## Detailed Component Breakdown

### 1. PositionalEncoding Class
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
    def forward(self, x):  # Add positional encoding to input
```

- **Purpose**: Adds temporal position information to sequential data
- **Implementation**: Sinusoidal positional encoding as in "Attention Is All You Need"
- **Parameters**: `d_model` (embedding dimension), `max_len` (maximum sequence length)

### 2. TransformerPolicy Class
```python
class TransformerPolicy(nn.Module):
    def __init__(self, K, d_model=64, nhead=4, num_layers=2, hidden=64,
                 dropout=0.1, activation="gelu", norm_first=True,
                 ff_hidden_mult=4, max_len=5000):
    def forward(self, X_t, M_feat):  # Generate portfolio weights
```

#### Key Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | - | Number of input features (factors) |
| `d_model` | 64 | Transformer embedding dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 2 | Transformer encoder layers |
| `hidden` | 64 | Scoring network hidden dimension |
| `dropout` | 0.1 | Dropout rate for regularization |
| `activation` | "gelu" | Activation function |
| `norm_first` | True | Pre-LN vs Post-LN architecture |
| `ff_hidden_mult` | 4 | Feed-forward dimension multiplier |
| `max_len` | 5000 | Maximum sequence length |

#### Forward Pass Flow
1. **Input Concatenation**: `X_t` (N, W, K) + `M_feat` (N, W, K) → (N, W, 2K)
2. **Projection**: Linear layer → (N, W, d_model)
3. **Positional Encoding**: Add temporal information
4. **Sequence Encoding**: Transformer encoder → (N, W, d_model)
5. **Asset Aggregation**: Mean pooling over time → (N, d_model)
6. **Cross-Attention**: Portfolio query attends to all assets
7. **Context Integration**: Combine asset + portfolio embeddings
8. **Scoring**: Neural network → asset scores
9. **Weight Generation**: Softmax normalization → portfolio weights

## Reward Function

### RewardFunction Class
```python
class RewardFunction:
    def __init__(self, risk_free_rate: float = 0.0):
    def compute(self, portfolio_returns, equal_weight_returns, info=None):
```

- **Objective**: Excess Sharpe ratio maximization
- **Formula**: `(μ_portfolio - μ_benchmark) / σ_excess`
- **Benchmark**: Equal-weighted portfolio
- **Risk-Adjustment**: Penalizes volatility while rewarding outperformance

## Training Mechanism

### Policy Gradient Training
- **Loss Function**: `-excess_sharpe_ratio` (maximization via minimization of negative)
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rate**: Adaptive scheduling (ReduceLROnPlateau)
- **Gradient Clipping**: Max norm = 1.0 for stability
- **Rolling Windows**: Multiple overlapping training episodes

### Exploration Strategy
- **Training Noise**: Gaussian noise (σ=0.01) added during training
- **Temperature Scaling**: Controls weight distribution concentration
- **Numerical Stability**: Logit centering before softmax

## Architecture Innovations

1. **Per-Factor Masking**: Handles missing data at individual feature level rather than entire observations
2. **Cross-Attention Design**: Portfolio-level decision making through global attention mechanism
3. **Context Integration**: Balances asset-specific and portfolio-level information
4. **Robust Training**: Gradient clipping, learning rate scheduling, and numerical stability measures
5. **Temporal Awareness**: Positional encoding for sequential financial data processing

## Model Configuration

Models are configured via Hydra YAML files:
- `rl/configs/model.yaml`: Architecture hyperparameters
- `rl/configs/train.yaml`: Training parameters
- `rl/configs/env.yaml`: Environment and data settings

## Usage Examples

### Initialize Policy
```python
from rl.models import TransformerPolicy

policy = TransformerPolicy(
    K=25,  # Number of factors
    d_model=64,
    nhead=4,
    num_layers=2
)
```

### Training Integration
```python
from rl.algo import train_policy

logs = train_policy(
    policy=policy,
    X_train=training_features,
    R_train=training_returns,
    epochs=20,
    lr=1e-3
)
```

## Dependencies

- `torch>=1.9.0`: PyTorch deep learning framework
- `numpy>=1.21.0`: Numerical computing
- `hydra-core>=1.2.0`: Configuration management

## References

- Vaswani et al. (2017): "Attention Is All You Need"
- Jiang et al. (2017): "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
- Sharpe (1966): "Mutual Fund Performance"