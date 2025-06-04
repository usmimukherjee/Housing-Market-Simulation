# Housing Market Simulation - Variant 2

This variant implements a housing market simulation with competitive bidding and Deep Q-Learning, focusing on a two-round bidding process where multiple households can compete for the same properties.

## Key Features

### 1. Competitive Bidding System
- **Two-Round Bidding Process**
  - First Round: All households place their bids on available houses
  - Second Round: Houses are awarded to the highest bidders
- **Bid Tracking**
  - Tracks all bids for each house
  - Records number of competing bids
  - Maintains bid history for analysis

### 2. Deep Q-Learning Implementation
- **Neural Network Architecture**
  ```python
  class DQN(nn.Module):
      def __init__(self, input_dim, output_dim):
          self.fc1 = nn.Linear(input_dim, 24)
          self.fc2 = nn.Linear(24, 24)
          self.fc3 = nn.Linear(24, output_dim)
  ```
- **Learning Components**
  - Experience replay buffer (size: 1000)
  - Target network for stable learning
  - Epsilon-greedy exploration (ε = 0.1)
  - Batch size: 32
  - Learning rate: 0.001
  - Discount factor: 0.9

### 3. Market Dynamics
- **House Price Dynamics**
  - Appreciation for occupied houses
  - Depreciation for houses on market
  - Price adjustments based on market conditions
- **Market Turnover**
  - 5% annual turnover rate
  - 8% of houses available for sale each year
- **Price Ranges**
  - Upper bound: 120% of WTP
  - Lower bound: 80% of WTP

### 4. Household Behavior
- **Willingness to Pay (WTP) Calculation**
  ```python
  wtp = (
      38.637 * birth_of_child
      - 12.309 * distance_to_CBD
      - 84.074 * distance_to_CBD
      - 66.961 * distance_to_business_center
      + 79.441 * distance_to_business_center
      - 55.490 * distance_to_school
      - 11.740 * distance_to_bus_stop
      - 49.657 * distance_to_park
      + 0.064 * population_density
      + 2.162 * owned_property_percentage * 1000
  )
  ```
- **Bidding Strategies**
  - RL households: DQN-based decision making
  - Traditional households: Simple increment-based strategy

## Model Components

### 1. BiddingGame
- Manages the competitive bidding process
- Handles price negotiations
- Tracks and processes all bids
- Determines winning bids

### 2. HousingMarket
- Main simulation environment
- Manages housing stock
- Tracks market conditions
- Coordinates bidding process
- Records market statistics

### 3. RLHousehold
- Implements DQN-based decision making
- Manages learning process
- Tracks performance metrics
- Handles price negotiations

## Usage

1. **Initialize the Model**
   ```python
   model = HousingMarket(years=2)
   ```

2. **Run Simulation**
   ```python
   model.simulate()
   ```

3. **Access Results**
   - Transaction data: `./results/transactions.csv`
   - Bidding results: `./results/model-deep-rl/`
   - Market statistics: `./results/market_state/`
   - Visualizations: `./output-analysis/`

## Output Analysis

### 1. Transaction Data
- Household and house IDs
- Final prices
- Number of competing bids
- Negotiation history

### 2. Market Statistics
- Average and median prices
- Vacancy rates
- Market composition
- Price trends

### 3. RL Metrics
- Training losses
- Episode rewards
- Successful deals
- Learning progress

### 4. Visualizations
- Market composition over time
- Price trends
- Market indicators
- Transaction volumes

## Dependencies

- Python 3.x
- PyTorch
- Mesa
- Pandas
- NumPy
- Matplotlib

## Key Parameters

```python
PERCENTAGE_ABOVE_WTP = 1.2    # Upper price bound
PERCENTAGE_BELOW_WTP = 0.8    # Lower price bound
ALPHA = 0.001                 # Learning rate
GAMMA = 0.9                   # Discount factor
EPSILON = 0.1                 # Exploration rate
BATCH_SIZE = 32               # Training batch size
REPLAY_MEMORY_SIZE = 1000     # Experience replay buffer
PERCENTAGE_HOUSES_FOR_SALE = 0.08  # Houses for sale
HOUSING_TURNOVER_RATE = 0.05  # Annual turnover
```

## Future Improvements

1. **Enhanced Bidding Strategies**
   - Implement more sophisticated negotiation tactics
   - Add time-based bidding strategies
   - Include market sentiment in decision making

2. **Market Dynamics**
   - Add more realistic price adjustment mechanisms
   - Implement market cycles
   - Include external economic factors

3. **RL Enhancements**
   - Implement different RL algorithms
   - Add multi-agent coordination
   - Improve exploration strategies

4. **Analysis Tools**
   - Add more detailed market analysis
   - Implement real-time visualization
   - Add performance benchmarking

## Contributing

Feel free to submit issues and enhancement requests! 