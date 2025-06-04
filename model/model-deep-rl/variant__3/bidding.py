import pandas as pd  
import numpy as np 
import mesa 
import os 
import tensorflow as tf 
from tensorflow import keras
from keras import layers  
from mesa.time import RandomActivation  
import csv 
import random  
import warnings 
import time 
from collections import defaultdict  
import torch
ignore_warnings = warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(0)

# Constants used in negotiation and RL
PERCENTAGE_ABOVE_WTP = 1.2  
PERCENTAGE_BELOW_WTP = 0.8  
ALPHA = 0.001 
GAMMA = 0.9  
EPSILON = 0.1  
NEGOTIATION_PENALTY_FACTOR = 0.05 
NUM_HOUSE_SAMPLE = 15  
NEGOTIATION_PENALTY_FACTOR = 0.05
INTERMEDIATE_REWARD = 0.5
FINAL_SUCCESS_REWARD = 1.0
BATCH_SIZE = 32

# Class to simulate the bidding process between households and houses
class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []
        
    def load_data(self, household_data_path="./results/household_eligibility.csv", houses_path='./inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.household_data.set_index("household_id", inplace=True)
        self.houses = pd.read_excel(houses_path)
    
    def negotiate_price(self, household, house_value):
        bid = household.lower_bound
        counter_offer = house_value
        negotiation_prices = []
        state = np.array([bid, counter_offer], dtype=np.float32)
        steps = 0
        total_reward = 0

        while bid <= counter_offer:
            # Epsilon-greedy action selection
            if np.random.rand() < EPSILON:
                action = np.random.choice(3)  # Exploration
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = household.model(state_tensor)
                    action = torch.argmax(q_values).item()

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
                final = True
            else:
                # Penalty for continued negotiation
                reward -= NEGOTIATION_PENALTY_FACTOR
                final = False

            next_state = np.array([new_bid, new_counter], dtype=np.float32)
            
            # Store experience in replay memory
            household.memory.push(state, action, reward, next_state)
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

            if final:
                break

        final_price = min(bid, counter_offer)
        household.current_episode_reward += total_reward
        
        # Additional reward for successful deal closure
        if final:
            household.successful_deals += 1
            household.deals.append(final_price)
        
        return final_price, negotiation_prices

    def sample_houses(self, eligible_houses):
        if len(eligible_houses) <= NUM_HOUSE_SAMPLE:
            return eligible_houses
        prices = eligible_houses['PUBVALUE']
        quantiles = prices.quantile([0.33, 0.66])
        low = eligible_houses[prices <= quantiles[0.33]].sample(5)
        mid = eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])].sample(5)
        high = eligible_houses[prices > quantiles[0.66]].sample(5)
        return pd.concat([low, mid, high])

    def rl_bidding(self, household, houses_df):
        eligible = houses_df[
            (houses_df['PUBVALUE'] > household.lower_bound) &
            (houses_df['PUBVALUE'] <= household.upper_bound)
        ]
        sampled = self.sample_houses(eligible)
        results = {}
        for _, house in sampled.iterrows():
            final_price, negotiation = self.negotiate_price(household, house['PUBVALUE'])
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

    def run(self, households):
        self.load_data()
        rl_households = list(households.values())

        # Perform actual bidding
        for household in rl_households:
            print(f"Starting bids for Household ID: {household.unique_id}")
            results = self.rl_bidding(household, self.houses)
            if results:
                sorted_list = sorted(results.items(), key=lambda x: x[1])
                bins = max(1, len(sorted_list) // 4)
                house_choices = [
                    random.choice(sorted_list[i * bins:(i + 1) * bins])
                    for i in range(4) if sorted_list[i * bins:(i + 1) * bins]
                ]

                chosen_house, chosen_price = random.choice(house_choices)
                chosen_houses = {h_id: price for h_id, price in house_choices}
                remaining_houses = {h: p for h, p in chosen_houses.items() if h != chosen_house}

                household.chosen_house = chosen_house
                household.chosen_house_price = chosen_price
                household.house_choices = remaining_houses

                print(f"Household {household.unique_id} chose house {chosen_house} at ${chosen_price:.2f}")
                print("Other three choices:", remaining_houses)
            else:
                print(f"No suitable houses found for Household {household.unique_id}")

        pd.DataFrame(self.bidding_results).to_csv("./results/model-deep-rl/variant__3/bidding_results.csv", index=False)