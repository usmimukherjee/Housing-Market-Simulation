# Housing Market DQN Simulation

## Overview

This simulation models a housing market using Deep Q-Networks (DQN) to train household agents to make optimal bidding decisions. The simulation includes multi-step negotiations, counter-offers, and realistic market dynamics.

## System Architecture

### Core Components

1. **DQN Agent** - Deep reinforcement learning agent that learns optimal bidding strategies
2. **Household Agent** - Represents individual households with specific preferences and budgets
3. **Bidding Game** - Manages the auction process and negotiations
4. **Housing Market** - Orchestrates the overall simulation environment

## Workflow

### Phase 1: Initialization

1. **Data Loading**
   - Load household data from `./inputs/HMTS.xlsx`
   - Load property data from `./inputs/Parcel.xlsx`
   - Initialize market conditions

2. **Agent Creation**
   - Create household agents with individual preferences
   - Calculate Willingness-to-Pay (WTP) using utility function
   - Initialize DQN agents for each household
   - Set budget ranges (±20% of WTP)

3. **Neural Network Setup**
   - State size: 12 features (WTP, house price, bid history, etc.)
   - Action space: 3 actions (accept, counter-bid, reject)
   - Network architecture: 4-layer fully connected with dropout

### Phase 2: Bidding Process

#### Step 1: Initial Bid Collection
- Each household evaluates eligible houses within their budget range
- Households select up to 5 houses using diversified strategies:
  - Price range quintiles
  - Mix of cheapest, most expensive, and median-priced options
  - Random selection for remaining slots
- DQN agents determine initial bid amounts (85-100% of asking price)

#### Step 2: Seller Evaluation
- Sellers evaluate all received bids
- Decision logic:
  - **Accept**: If highest bid ≥ 90% of asking price
  - **Counter-offer**: Offer 95% of asking price to 1-3 top bidders
  - **Reject**: If no viable offers

#### Step 3: Negotiation Handling
- Households use DQN to respond to counter-offers
- Possible actions:
  - Accept counter-offer
  - Make new counter-bid (1-3% higher)
  - Reject and withdraw
- Sellers select best final offer from responses

### Phase 3: Market Resolution

1. **Deal Finalization**
   - Ensure each household can only win one house
   - Prioritize deals by final price (highest first)
   - Update household ownership status

2. **Experience Recording**
   - Store state-action-reward transitions
   - Calculate rewards based on:
     - Base reward (+100) for winning
     - Savings bonus (up to +50)
     - Budget penalty (up to -20)
     - Loss penalty (-10)

3. **Model Training**
   - Train DQN networks using experience replay
   - Update target networks periodically
   - Apply epsilon decay for exploration

### Phase 4: Multi-Year Simulation

1. **Market Updates**
   - Increase house prices (+3% annually for inflation)
   - Adjust household WTP (+2% for income growth)
   - Update DQN target networks

2. **Performance Tracking**
   - Monitor success rates
   - Track average purchase prices
   - Analyze learning progress

## Key Features

### DQN Implementation
- **Experience Replay**: 10,000 memory buffer with batch training
- **Target Network**: Updated every 10 steps for stability
- **Epsilon-Greedy**: Exploration decay from 1.0 to 0.01
- **Neural Architecture**: 128-unit hidden layers with dropout

### Bidding Strategies
- **Conservative**: Lower exploration, careful bidding
- **Moderate**: Balanced approach with standard parameters
- **Aggressive**: High exploration, competitive bidding

### Negotiation Engine
- Multi-step negotiations (up to 3 rounds)
- Dynamic counter-offer generation
- Seller preference modeling
- Realistic market constraints

## File Structure

```
├── inputs/
│   ├── HMTS.xlsx          # Household preference data
│   └── Parcel.xlsx        # Property listings
├── results/
│   └── model-drl/
│       ├── household_eligibility.csv    # Generated WTP data
│       ├── bidding_results_dqn.csv     # Detailed results
│       └── dqn_models/                 # Saved neural networks
└── main.py                # Main simulation code
```

## Usage

### Basic Simulation
```python
# Run single year simulation
model = HousingMarket(years=1)
model.simulate(save_models=True, load_models=False)
```

### Multi-Year Simulation
```python
# Run 10-year simulation with market evolution
model = HousingMarket(years=10)
model.simulate(save_models=True, load_models=False)
```

### Performance Analysis
```python
# Analyze DQN learning performance
analyze_dqn_performance(households)

# Compare different bidding strategies
strategy_results = compare_bidding_strategies()
```

## Configuration Parameters

### DQN Parameters
- `LEARNING_RATE`: 0.001 (Adam optimizer)
- `GAMMA`: 0.99 (discount factor)
- `EPSILON_START`: 1.0 (initial exploration)
- `EPSILON_END`: 0.01 (minimum exploration)
- `EPSILON_DECAY`: 0.995 (decay rate)
- `MEMORY_SIZE`: 10,000 (replay buffer)
- `BATCH_SIZE`: 32 (training batch)
- `TARGET_UPDATE`: 10 (target network update frequency)

### Market Parameters
- `PERCENTAGE_ABOVE_WTP`: 1.2 (maximum budget multiplier)
- `PERCENTAGE_BELOW_WTP`: 0.8 (minimum budget multiplier)
- Annual inflation: 3%
- Annual income growth: 2%

## Output Analysis

### Key Metrics
- **Success Rate**: Percentage of households that successfully purchase homes
- **Average Purchase Price**: Mean final transaction price
- **Negotiation Steps**: Average number of rounds per transaction
- **Learning Progress**: Epsilon decay and memory utilization

### Generated Files
- `household_eligibility.csv`: WTP calculations and budget ranges
- `bidding_results_dqn.csv`: Detailed bidding history and outcomes
- `dqn_models/`: Trained neural network weights for each household

## Dependencies

```python
pandas
numpy
mesa
torch
warnings
collections
random
```

## Model Validation

The simulation includes several validation mechanisms:
- Constraint enforcement (one house per household)
- Budget compliance checking
- Negotiation step limits
- Market condition updates
- Neural network convergence monitoring

## Future Enhancements

- Multi-agent learning with opponent modeling
- Dynamic pricing strategies for sellers
- Geographic preference modeling
- Market crash/boom scenarios
- Real estate agent intermediaries


model 1 works, but need to test model 2