# Housing Market Simulation with Deep Q-Learning

This file implements a housing market simulation that uses Deep Q-Learning (DQN) to model household bidding behavior in a competitive real estate market. The simulation incorporates both RL-based households and traditional simulation households to study market dynamics and bidding strategies.

## Overview

The simulation models a housing market where:
- Multiple households compete for available houses
- Houses have different characteristics and price points
- Households use either RL-based or traditional bidding strategies
- Market conditions evolve over time
- Competitive bidding determines house allocations

## Key Features

1. **Deep Q-Learning Implementation**
   - Neural network-based Q-learning for household decision making
   - Experience replay for stable learning
   - Target network for improved training stability
   - Epsilon-greedy exploration strategy

2. **Competitive Bidding System**
   - Multiple households can bid on the same house
   - Bids are processed in two rounds:
     - First round: All households place their bids
     - Second round: Houses are awarded to highest bidders
   - Tracks number of competing bids for each transaction

3. **Market Dynamics**
   - House price appreciation/depreciation
   - Market turnover rates
   - Vacancy rates
   - Price trends over time

4. **Household Behavior**
   - Willingness to Pay (WTP) calculation
   - Price range determination
   - Negotiation strategies
   - Learning from past experiences (for RL households)

## Model Components

### 1. DQN Network
```python
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)
```
- 3-layer neural network
- Input: State representation (bid, counter-offer)
- Output: Q-values for different actions

### 2. BiddingGame
- Manages the competitive bidding process
- Handles negotiation between households and sellers
- Tracks and processes all bids
- Determines winning bids based on price

### 3. HousingMarket
- Main simulation environment
- Manages housing stock
- Tracks market conditions
- Coordinates bidding process
- Records market statistics

## Key Parameters

```python
PERCENTAGE_ABOVE_WTP = 1.2    # Upper bound for house price
PERCENTAGE_BELOW_WTP = 0.8    # Lower bound for house price
ALPHA = 0.001                 # Learning rate
GAMMA = 0.9                   # Discount factor
EPSILON = 0.1                 # Exploration rate
BATCH_SIZE = 32               # Training batch size
REPLAY_MEMORY_SIZE = 1000     # Experience replay buffer size
```

## Usage

1. **Setup**
   ```python
   model = HousingMarket(years=2)  # Initialize for 2 years
   ```

2. **Run Simulation**
   ```python
   model.simulate()
   ```

3. **View Results**
   - Transaction logs in `./results/transactions.csv`
   - Bidding results in `./results/model-deep-rl/`
   - Market statistics in `./results/market_state/`
   - Visualization plots in `./output-analysis/`

## Output Analysis

The simulation generates several types of outputs:
1. **Transaction Data**
   - Household IDs
   - House IDs
   - Final prices
   - Number of competing bids

2. **Market Statistics**
   - Average house prices
   - Median house prices
   - Vacancy rates
   - Market composition

3. **RL Metrics**
   - Training losses
   - Episode rewards
   - Successful deals
   - Learning progress

## Dependencies

- Python 3.x
- PyTorch
- Mesa
- Pandas
- NumPy
- Matplotlib

## File Structure

```
model/DeepQRL-model/
├── model4.py           # Main simulation code
├── README.md          # This file
├── results/           # Output data
│   ├── model-deep-rl/
│   ├── market_state/
│   └── transactions.csv
└── output-analysis/   # Visualizations
```

## Future Improvements

1. Add more sophisticated market dynamics
2. Implement different types of RL agents
3. Add more house characteristics
4. Enhance visualization capabilities
5. Add more detailed market analysis tools

## Contributing

Feel free to submit issues and enhancement requests!