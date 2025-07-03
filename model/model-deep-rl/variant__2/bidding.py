import pandas as pd  
import numpy as np 
import mesa 
import os 
import csv 
import random  
import warnings 
import time 
from collections import defaultdict  
from mesa.time import RandomActivation  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from household_rl import RLHousehold

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Constants
PERCENTAGE_ABOVE_WTP = 1.2  
PERCENTAGE_BELOW_WTP = 0.8  
ALPHA = 0.001 
GAMMA = 0.9  
EPSILON = 0.1  
NEGOTIATION_PENALTY_FACTOR = 0.05 
NUM_HOUSE_SAMPLE = 15  
INTERMEDIATE_REWARD = 0.5
FINAL_SUCCESS_REWARD = 1.0
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000
TARGET_UPDATE_FREQ = 10
PERCENTAGE_HOUSES_FOR_SALE = 0.08  # 8% of houses are for sale each year
HOUSING_TURNOVER_RATE = 0.05  # 5% annual turnover of occupied houses

# Class to simulate the bidding process
class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []
        self.transactions = []
        
    def load_data(self, household_data_path="./results/household_eligibility.csv", houses_path='./inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.household_data.set_index("household_id", inplace=True)
        self.houses = pd.read_excel(houses_path)
    
    def reset_for_new_year(self):
        """Reset bidding results while preserving transactions"""
        self.bidding_results = []
    
    def negotiate_price(self, household, house_value):
        bid = household.lower_bound
        counter_offer = house_value
        negotiation_prices = []
        state = np.array([bid, counter_offer], dtype=np.float32)
        steps = 0
        total_reward = 0
        done = False

        while not done and steps < 100:  # Add max steps to prevent infinite loops
            # Epsilon-greedy action selection
            if isinstance(household, RLHousehold):
                if np.random.rand() < EPSILON:
                    action = np.random.choice(3)  # Exploration
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        q_values = household.dqn_model(state_tensor)
                        action = torch.argmax(q_values).item()
            else:
                # For regular households, use a simple strategy
                if bid < counter_offer:
                    action = np.random.choice([0, 1])  # Small or medium increment
                else:
                    action = 2  # Large increment

            increment = [0.02, 0.05, 0.1][action]
            new_bid = bid + bid * increment
            new_counter = counter_offer - counter_offer * np.random.uniform(0.02, 0.05)

            # Calculate reward with intermediate rewards
            reward = 0
            if new_bid > bid and new_counter < counter_offer:
                # Reward for mutual compromise
                reward += INTERMEDIATE_REWARD
            
            if new_bid >= new_counter:
                # Final success reward
                reward += FINAL_SUCCESS_REWARD - steps * NEGOTIATION_PENALTY_FACTOR
                done = True
            else:
                # Penalty for continued negotiation
                reward -= NEGOTIATION_PENALTY_FACTOR

            next_state = np.array([new_bid, new_counter], dtype=np.float32)
            
            # Store experience in replay memory and train only for RL households
            if isinstance(household, RLHousehold):
                household.memory.push(state, action, reward, next_state, done)
                total_reward += reward
                
                # Train the model
                if len(household.memory) >= BATCH_SIZE:
                    loss = household.train_model()
                    household.current_episode_loss += loss
            
            # Update state
            state = next_state
            bid = new_bid
            counter_offer = new_counter
            negotiation_prices.append((bid, counter_offer))
            steps += 1

        final_price = (bid + counter_offer) / 2 if done else None
        
        if done:
            if isinstance(household, RLHousehold):
                household.current_episode_reward += total_reward
            household.successful_deals += 1
            household.deals.append(final_price)
        
        return final_price, negotiation_prices

    def sample_houses(self, eligible_houses):
        if len(eligible_houses) <= NUM_HOUSE_SAMPLE:
            return eligible_houses
        prices = eligible_houses['PUBVALUE']
        quantiles = prices.quantile([0.33, 0.66])
        low = eligible_houses[prices <= quantiles[0.33]].sample(min(5, len(eligible_houses[prices <= quantiles[0.33]])))
        mid = eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])].sample(min(5, len(eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])])))
        high = eligible_houses[prices > quantiles[0.66]].sample(min(5, len(eligible_houses[prices > quantiles[0.66]])))
        return pd.concat([low, mid, high])

    def rl_bidding(self, household, houses_df):
        eligible = houses_df[
            (houses_df['PUBVALUE'] > household.lower_bound) &
            (houses_df['PUBVALUE'] <= household.upper_bound)
        ]
        if eligible.empty:
            return {}
            
        sampled = self.sample_houses(eligible)
        results = {}
        for _, house in sampled.iterrows():
            final_price, negotiation = self.negotiate_price(household, house['PUBVALUE'])
            if final_price is not None:
                results[house['OBJECTID']] = final_price
                self.bidding_results.append({
                    'household_id': household.unique_id,
                    'house_id': house['OBJECTID'],
                    'initial_bid': household.lower_bound,
                    'final_price': final_price,
                    'counter_offer': house['PUBVALUE'],
                    'negotiation_steps': len(negotiation),
                    'negotiation_prices': str(negotiation)
                })
        return results

    def run(self, households, model):
        """Run bidding process with market clearing"""
        # Get current houses for sale from the market model
        houses_df = model.get_houses_for_sale()
        print(f"\nNumber of houses available for sale: {len(houses_df)}")
        
        # Reset bidding results for the new year
        self.reset_for_new_year()
        
        # Track transactions
        transactions = []
        
        # Iterate over every type of households (e.g., 'rl', 'simulation')
        for household_type, household_dict in households.items():
            print(f"\nProcessing {household_type} households...")
            # Shuffle households to randomize bidding order
            household_list = list(household_dict.values())
            random.shuffle(household_list)
            
            for household in household_list:
                # Skip if household already has a house
                if hasattr(household, 'chosen_house') and household.chosen_house is not None:
                    continue
                    
                print(f"\nStarting bids for Household ID: {household.unique_id}")
                print(f"WTP: ${household.wtp:.2f}")
                print(f"Price range: ${household.lower_bound:.2f} - ${household.upper_bound:.2f}")
                
                results = self.rl_bidding(household, houses_df)
                
                if results:
                    # Household selects a house most affordable
                    sorted_list = sorted(results.items(), key=lambda x: x[1])
                    chosen_house, chosen_price = sorted_list[0] 
                    
                    # Record transaction
                    transactions.append({
                        'household_id': household.unique_id,
                        'house_id': chosen_house,
                        'final_price': chosen_price,
                        'year': model.current_year + 1
                    })
                    
                    # Update household status
                    household.chosen_house = chosen_house
                    household.chosen_house_price = chosen_price
                    
                    print(f"Household {household.unique_id} chose house {chosen_house} at ${chosen_price:.2f}")
                    
                    # Remove purchased house from market
                    houses_df = houses_df[houses_df['OBJECTID'] != chosen_house]
                    
                    # Update house status in the model
                    for house in model.housing_stock:
                        if house.id == chosen_house:
                            house.status = "occupied"
                            house.years_occupied = 0
                            print(f"Updated house {chosen_house} status to occupied")
                            break
                else:
                    print(f"No suitable houses found for Household {household.unique_id}")
        
        # Save bidding results
        if self.bidding_results:
            bidding_df = pd.DataFrame(self.bidding_results)
            os.makedirs("./results/model-deep-rl/variant__2", exist_ok=True)
            bidding_df.to_csv(f"./results/model-deep-rl/variant__2/bidding_results_year_{model.current_year + 1}.csv", index=False)
        
        # Save transaction log
        if transactions:
            pd.DataFrame(transactions).to_csv("./results/model-deep-rl/variant__2/transactions.csv", index=False)
        
        return len([t for t in transactions if t['year'] == model.current_year + 1])