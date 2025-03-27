## Housing Market Model using Deep Reinforcement Learning

This is a simple implementation of agent based housing market model using deep reinforcement learning 
### Agents

- Household
    - id
    - wtp
    - range

- Household-RL
    - household neural network model for decision making

- House
    - id
    - price


- Housing Market
    - load household data
    - load parcels - housing stock
    - process household data - wtp and range
    - annualy update market conditions
        - house price - inflation - 3%
        - household - 2% increase
    - for each household - call bidding
    - save - household, range, housing stock in range

## Model
- Bidding
    - for each household
        - bid on each house in range
            - negotiate price until bid is equal to counter offer using trained model
                - increase bidding price -> 2% or 5% randomly
                - counter bid decrease -> 2% or 5% randomly
            - outputs
                - a house closer to lower bound
                - a house in mid range
                - a house closer to upper bound
                - 2% above upper bound


# Current Progress
- , need to setup outputs and visualizations, and modify the training and testing, need to optimize and validate