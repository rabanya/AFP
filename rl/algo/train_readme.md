# Training and Evaluation Algorithms

This directory contains the core algorithms for training and evaluating reinforcement learning policies in the portfolio management system.

## Overview

The portfolio management system uses **policy gradient methods** to train neural network policies that maximize excess Sharpe ratio over equal-weighted portfolios. The training process involves rolling time windows and robust optimization techniques.

## Policy Gradient Training (`trainer.py`)

### Core Algorithm: `train_policy()`

The `train_policy` function implements a sophisticated policy gradient algorithm that learns portfolio allocation strategies through reinforcement learning.

#### Training Objective
```python
# Maximize excess Sharpe ratio over equal-weighted benchmark
reward = excess_sharpe_ratio(portfolio_returns, equal_weight_returns)
loss = -reward  # Gradient ascent via descent of negative reward
```

#### Key Components

##### 1. **Rolling Time Windows**
```python
# Create overlapping training windows for robust learning
for start in range(0, T - length // 2 + 1, length):
    windows.append((start, start + length // 2))

# Within each window, sample random starting points
candidates = list(range(start, min(start + length // 2, T)))
t0 = random.choice(candidates)
t1 = min(t0 + length // 2, T)
```

- **Purpose**: Creates diverse training episodes from historical data
- **Benefits**: Reduces overfitting to specific time periods, improves generalization
- **Window Size**: `length // 2` determines episode duration

##### 2. **Policy Gradient Optimization**
```python
# Forward pass through episode
for t in range(t0, t1):
    w_t = policy(X_t, M_feat_t)  # Generate portfolio weights
    r_p = (w_t * R_t).sum()      # Portfolio return
    r_ew = R_t.mean()            # Equal-weight return
    port.append(r_p)
    ew.append(r_ew)

# Compute reward and optimize
reward = reward_fn.compute(port, ew)
loss = -reward
loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
opt.step()
```

- **Reinforcement Learning**: Policy generates actions (weights), environment provides rewards (excess returns)
- **Gradient Clipping**: Prevents exploding gradients (max norm = 1.0)
- **End-to-End Training**: Differentiable from inputs to Sharpe ratio

##### 3. **Adaptive Optimization**
```python
# Adam optimizer with regularization
opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)

# Learning rate scheduling based on performance
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=5
)
```

- **Weight Decay**: L2 regularization prevents overfitting
- **Adaptive LR**: Reduces learning rate when Sharpe improvement stalls
- **Performance-Based**: Uses Sharpe ratio as scheduling metric

### Training Dynamics

#### Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 20 | Number of training epochs |
| `length` | 48 | Total window length (episode = length // 2) |
| `lr` | 1e-3 | Initial learning rate |
| `weight_decay` | 1e-5 | L2 regularization strength |
| `clip_norm` | 1.0 | Maximum gradient norm |

#### Training Logs
```python
logs.append({
    "epoch": ep + 1,
    "window": (start, end),
    "loss": float(loss.detach().cpu()),
    "sharpe": float(reward.detach().cpu())
})
```

- **Loss**: Negative Sharpe ratio (lower is better)
- **Sharpe**: Excess Sharpe ratio (higher is better)
- **Window**: Training window boundaries for reproducibility

#### Progress Monitoring
```python
# Print progress every 5 epochs
if (ep + 1) % 5 == 0:
    print(f"Epoch {ep+1}/{epochs}, Mean Sharpe: {mean_train_sharpe:.4f}, LR: {opt.param_groups[0]['lr']:.6f}")
```

### Algorithm Flow

1. **Initialization**: Set seeds, initialize policy, optimizer, scheduler
2. **Window Creation**: Generate overlapping training windows
3. **Epoch Loop**:
   - For each window, sample random episode start
   - Generate portfolio weights through policy
   - Compute returns and excess Sharpe reward
   - Backpropagate and update parameters
   - Log training metrics
4. **Learning Rate Update**: Adjust LR based on epoch performance
5. **Progress Reporting**: Print summary statistics

## Policy Evaluation (`evaluator.py`)

### Core Algorithm: `evaluate_episode()`

The `evaluate_episode` function provides comprehensive evaluation of trained policies, computing performance metrics and generating detailed portfolio analytics.

#### Evaluation Metrics

##### 1. **Return Calculations**
```python
r_p = (w_t * R_t).sum().item()     # Portfolio return
r_ew = R_t.mean().item()           # Equal-weight benchmark
```

##### 2. **Wealth Tracking**
```python
cum_p *= (1.0 + r_p)   # Cumulative portfolio wealth
cum_ew *= (1.0 + r_ew) # Cumulative benchmark wealth
```

##### 3. **Performance Summary**
```python
sharpe_excess = reward_fn.compute(torch.tensor(port), torch.tensor(ew))
```

#### Return Values
```python
return port, ew, wealth_port, wealth_ew, sharpe_excess, detailed_results
```

- **`port`**: Array of portfolio returns for each time step
- **`ew`**: Array of equal-weighted returns for each time step
- **`wealth_port`**: Cumulative wealth curve for portfolio
- **`wealth_ew`**: Cumulative wealth curve for benchmark
- **`sharpe_excess`**: Excess Sharpe ratio over evaluation period
- **`detailed_results`**: Optional detailed portfolio compositions

### Detailed Evaluation Mode

When `detailed=True`, the evaluator provides comprehensive portfolio analytics:

```python
portfolio_details = {
    'time_step': t - t0,
    'date': dates_list[t] if dates_list else f"step_{t}",
    'asset_ids': ids_list[t].tolist() if ids_list else [],
    'portfolio_weights': w_t.cpu().numpy().tolist(),
    'returns': R_t.cpu().numpy().tolist(),
    'portfolio_return': r_p,
    'equal_weight_return': r_ew,
    'cumulative_wealth_port': cum_p,
    'cumulative_wealth_ew': cum_ew
}
```

### Evaluation Characteristics

#### Inference Mode
```python
policy.eval()  # Set to evaluation mode
with torch.no_grad():  # No gradient computation
    w_t = policy(X_t, M_feat_t)
```

- **No Training**: Disables dropout, batch norm updates
- **No Gradients**: Faster inference, reduced memory usage
- **Deterministic**: Consistent results across runs

#### Robustness Features
- **Mask Handling**: Graceful fallback when masks unavailable
- **Device Support**: Automatic tensor placement
- **Numerical Stability**: Proper handling of edge cases

## Algorithm Integration

### Training-Evaluation Pipeline
```python
# 1. Train policy
logs = train_policy(policy, X_train, R_train, epochs=20)

# 2. Evaluate on validation set
port_returns, ew_returns, wealth_p, wealth_ew, sharpe, details = evaluate_episode(
    policy, X_val, R_val, detailed=True
)

# 3. Analyze results
print(f"Excess Sharpe: {sharpe:.4f}")
print(f"Final Wealth: {wealth_p[-1]:.2f} vs {wealth_ew[-1]:.2f}")
```

### Key Design Principles

#### 1. **Financial Realism**
- **Rolling Windows**: Simulates real-world portfolio rebalancing
- **Transaction Costs**: Framework ready for cost incorporation
- **Risk Management**: Sharpe ratio focuses on risk-adjusted returns

#### 2. **Training Stability**
- **Gradient Clipping**: Prevents optimization instability
- **Learning Rate Scheduling**: Adapts to training progress
- **Regularization**: Weight decay prevents overfitting

#### 3. **Evaluation Rigor**
- **Benchmark Comparison**: Always compares against equal-weighted
- **Multiple Metrics**: Returns, wealth curves, risk-adjusted performance
- **Detailed Logging**: Optional granular portfolio analytics

#### 4. **Scalability**
- **GPU Support**: Automatic device placement
- **Memory Efficient**: No gradient computation during evaluation
- **Batch Processing**: Handles variable numbers of assets

## Usage Examples

### Basic Training
```python
from rl.algo import train_policy

# Train policy for 20 epochs
logs = train_policy(
    policy=policy,
    X_train=training_features,
    R_train=training_returns,
    epochs=20,
    lr=1e-3,
    device="cuda"
)
```

### Comprehensive Evaluation
```python
from rl.algo import evaluate_episode

# Evaluate with detailed logging
results = evaluate_episode(
    policy=policy,
    X_val=validation_features,
    R_val=validation_returns,
    detailed=True,
    dates_list=validation_dates,
    ids_list=asset_ids
)

port_ret, ew_ret, wealth_p, wealth_ew, sharpe, details = results
```

### Performance Analysis
```python
# Analyze training progress
for log in logs:
    print(f"Epoch {log['epoch']}: Sharpe={log['sharpe']:.4f}, Loss={log['loss']:.4f}")

# Analyze evaluation results
print(f"Excess Sharpe: {sharpe:.4f}")
print(f"Portfolio Growth: {wealth_p[-1]/wealth_p[0]:.2f}x")
print(f"Benchmark Growth: {wealth_ew[-1]/wealth_ew[0]:.2f}x")
```

## Dependencies

- `torch>=1.9.0`: PyTorch for neural network operations
- `numpy>=1.21.0`: Numerical computations
- `random`: Random sampling for training windows

## Integration Notes

- **Model Compatibility**: Works with any PyTorch nn.Module that outputs portfolio weights
- **Reward Function**: Uses standardized RewardFunction interface
- **Data Format**: Expects list of numpy arrays for time-series data
- **Masking Support**: Optional per-factor masking for missing data handling</contents>
</xai:function_call">rl/algo/algo_readme.md