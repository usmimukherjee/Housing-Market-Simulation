import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
from collections import defaultdict, deque
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(0)

PERCENTAGE_ABOVE_WTP = 1.2
PERCENTAGE_BELOW_WTP = 0.8

# Deep RL Parameters
ALPHA = 0.001
GAMMA = 0.9
EPSILON = 0.1

class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []
    
    def load_data(self, household_data_path="../../results/household_eligibility.csv", houses_path='../../inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.houses = pd.read_excel(houses_path)

    def run(self, households):
        self.load_data()
        
        for household in tqdm(households.values(), desc="Processing Households"):
            state = np.array([[household.wtp]])
            
            # Batch Prediction Instead of Calling it Every Loop Iteration
            q_values = household.model.predict(state, verbose=0)
            action = np.random.choice(3) if np.random.rand() < EPSILON else np.argmax(q_values)
            reward = np.random.choice([1, -1])
            
            # Use deque for efficient memory handling
            if not hasattr(household, 'memory'):
                household.memory = deque(maxlen=100)  # Fixed-size buffer
            
            household.memory.append((state, action, reward, state))
            
            if len(household.memory) > 10:
                self.train(household.model, random.sample(household.memory, 10))

        for household in tqdm(households.values(), desc="Bidding for Households"):
            results = self.rl_bidding(household)
            
            if results:
                house_list = sorted(results.items(), key=lambda x: x[1])
                bin_size = max(1, len(house_list) // 4)
                
                # Optimize house selection using list slicing
                house_choices = [
                    random.choice(house_list[i * bin_size:(i + 1) * bin_size])
                    for i in range(4) if house_list[i * bin_size:(i + 1) * bin_size]
                ]
                
                chosen_house, chosen_price = random.choice(house_choices)
                household.chosen_house = chosen_house
                household.chosen_house_price = chosen_price
                household.house_choices = dict(house_choices)
                
                # Vectorized update
                self.household_data.loc[household.unique_id, ['house_choices', 'chosen_house', 'chosen_house_price']] = [
                    str(household.house_choices), chosen_house, chosen_price
                ]
                
                household.successful_deals += 1
                household.deals.append(chosen_house)

        # Save results once instead of inside the loop
        pd.DataFrame(self.bidding_results).to_csv("../../results/bidding_results.csv", index=False)
    
    def negotiate_price(self, household, counter_offer):
        bid = household.lower_bound
        negotiation_prices = [(bid, counter_offer)]
        state = np.array([[household.wtp]])

        while bid <= counter_offer:
            action = np.argmax(household.model.predict(state, verbose=0))
            bid *= 1 + [0.02, 0.05, 0.1][action]

            if bid >= counter_offer:
                return counter_offer, negotiation_prices
            
            counter_offer *= 1 - np.random.uniform(0.02, 0.05)
            negotiation_prices.append((bid, counter_offer))
        
        return min(bid, counter_offer), negotiation_prices
    
    def rl_bidding(self, household):
        eligible_houses = self.houses[(self.houses['PUBVALUE'] > household.lower_bound) & (self.houses['PUBVALUE'] <= household.upper_bound)]
        
        results = {}
        for _, house in tqdm(eligible_houses.iterrows(), total=len(eligible_houses), desc=f"Negotiating Prices for Household {household.unique_id}", leave=False):
            final_price, negotiation_prices = self.negotiate_price(household, house['PUBVALUE'])
            results[house['OBJECTID']] = final_price
            
            self.bidding_results.append({
                'household_id': household.unique_id,
                'house_id': house['OBJECTID'],
                'initial_bid': household.lower_bound,
                'final_price': final_price,
                'counter_offer': household.wtp,
                'negotiation_steps': len(negotiation_prices),
                'negotiation_prices': str(negotiation_prices)
            })
        
        return results
    
    def train(self, model, batch):
        states, actions, rewards, next_states = zip(*batch)
        states, next_states = np.vstack(states), np.vstack(next_states)

        # Batch Prediction Instead of Iterative Calls
        q_next = model.predict(next_states, verbose=0)
        q_current = model.predict(states, verbose=0)

        targets = np.array(rewards) + GAMMA * np.max(q_next, axis=1)
        q_current[np.arange(len(actions)), actions] = targets

        # Single batch training instead of multiple iterations
        model.fit(states, q_current, epochs=1, verbose=0)