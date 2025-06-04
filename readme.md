# Housing Market Simulation

This project simulates the housing market using Agent-Based Modeling (ABM) techniques and Reinforcement Learning (RL). It explores how individual preferences, behaviors, and learning strategies influence market dynamics through various bidding processes.

## Project Structure

### 1. Baseline Model (`baseline/`)
- Basic implementation of an ABM-based housing market model
- Simple bidding process without WTP considerations
- Serves as a reference implementation
- Key features:
  - Basic household and house agents
  - Simple market clearing mechanism
  - Basic price dynamics

### 2. Enhanced Models (`model/`)

#### a. Model with WTP (`model-rl/`)
- Enhanced ABM implementation with willingness to pay (WTP)
- Sophisticated bidding process
- Features:
  - WTP calculation based on multiple factors
  - Price range determination
  - Market dynamics simulation

#### b. Deep RL Model (`model-deep-rl/`)
- Advanced implementation using Deep Q-Learning
- Multiple variants available:
  - **Variant 1**: Basic DQN implementation using pytorch
  - **Variant 2**: Competitive bidding with two-round process using pytorch
  - **Variant 3**: Basic Deep RL simulation using tensorflow

Key features across variants:
- Neural network-based decision making
- Experience replay and target networks
- Market state tracking
- Performance metrics and visualization

#### c. Result Parser (`result-parser/`)
- Tools for analyzing simulation results
- Generates visualizations and statistics
- Processes market dynamics data

### 3. Replication (`replication/`)
- Contains replication studies and experiments
- Validation of model behavior
- Comparison with real-world data

## Key Features

### Market Dynamics
- House price appreciation/depreciation
- Market turnover rates
- Vacancy rates
- Price trends over time

### Household Behavior
- Willingness to Pay (WTP) calculation
- Price range determination
- Learning from market experiences (RL models)
- Competitive bidding strategies

### Analysis Tools
- Market state tracking
- Transaction logging
- Performance metrics
- Visualization tools

## Getting Started

1. **Setup**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run Simulations**
   ```python
   # Baseline model
   python baseline/model.py

   # RL model
   python model/model-rl/model.py

   # Deep RL model
   python model/model-deep-rl/variant__2/model.py
   ```

3. **Analyze Results**
   ```python
   # Parse and visualize results
   python result-parser/analyze.py
   ```

## Output Structure

```
results/
├── baseline/              # Baseline model results
├── model-rl/             # RL model results
├── model-deep-rl/        # Deep RL model results
│   ├── variant__1/
│   ├── variant__2/
│   └── variant__3/
└── market_state/         # Market statistics

output-analysis/
├── baseline/             # Baseline visualizations
├── model-rl/            # RL model visualizations
└── model-deep-rl/       # Deep RL model visualizations
```

## Dependencies

- Python 3.x
- PyTorch (for Deep RL models)
- Mesa (for ABM)
- Pandas
- NumPy
- Matplotlib

## Future Improvements

1. **Model Enhancements**
   - More sophisticated market dynamics
   - Additional RL algorithms
   - Enhanced bidding strategies

2. **Analysis Tools**
   - Real-time visualization
   - Advanced market analysis
   - Performance benchmarking

3. **Documentation**
   - Detailed API documentation
   - Usage examples
