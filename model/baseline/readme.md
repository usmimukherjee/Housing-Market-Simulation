## Housing Market Bidding Model
This is a simple implementation of agent based housing market model

### Agents

- Household 
    - id
    - wtp
    - range

- House
    - id
    - price

- Housing Market
    - load household data
    - load parcels - housing stock
    - process household data - wtp and range
    - for each household - call bidding
    - save - household, range, housing stock in range

## Model
- Bidding
    - for each household
        - bid on each house in range
            - negotiate price until bid is equal to counter offer
                - increase bidding price -> 2% or 5% randomly
                - counter bid decrease -> 2% or 5% randomly
            - outputs
                - a house closer to lower bound
                - a house in mid range
                - a house closer to upper bound
                - 2% above upper bound
    