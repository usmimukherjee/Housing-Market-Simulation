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

ignore_warnings = warnings.filterwarnings("ignore")
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
        for household in households.values():
            state = np.array([household.wtp]).reshape(1, -1)
            if np.random.rand() < EPSILON:
                action = np.random.choice(3) 
            else:
                action = np.argmax(household.model.predict(state)) 
            
            reward = np.random.choice([1, -1]) 
            next_state = state  
            
            household.memory.append((state, action, reward, next_state))
            
            # Train
            if len(household.memory) > 10:
                batch = random.sample(household.memory, 10)
                self.train(household.model, batch)

        for household in households.values():
            for household_id, household in households.items():
                print(f"Starting bids for Household ID: {household.unique_id} with price range ${household.lower_bound:.2f} - ${household.upper_bound:.2f}")
                results = self.rl_bidding(household)
                if results:
                    # Sort houses by price
                    house_list = sorted(results.items(), key=lambda x: x[1])  
                    house_ids, house_prices = zip(*house_list)

                    # Divide into four price bins
                    bin_size = max(1, len(house_ids) // 4)  # Avoid division by zero
                    house_choices = [(random.choice(house_list[i * bin_size:(i + 1) * bin_size])) for i in range(4) if house_list[i * bin_size:(i + 1) * bin_size]]

                    # Extract house IDs and prices
                    chosen_houses = {h_id: price for h_id, price in house_choices}

                    # Randomly choose one final house
                    chosen_house, chosen_price = random.choice(house_choices)
                    household.chosen_house = chosen_house
                    household.chosen_house_price = chosen_price

                    # Store choices in household data
                    household.house_choices = chosen_houses
                    self.household_data.loc[household.unique_id, ['house_choices', 'chosen_house', 'chosen_house_price']] = [
                        str(chosen_houses), chosen_house, chosen_price
                    ]

                    # Track successful deals
                    household.successful_deals += 1
                    household.deals.append(chosen_house)

                    # Print results
                    print(f"Household {household.unique_id} chose house {chosen_house} at ${chosen_price:.2f}")
                    remaining_houses = {h: p for h, p in chosen_houses.items() if h != chosen_house}
                    print("Other three choices:", remaining_houses)
                    
                    household.chosen_house = chosen_house
                    household.house_choices = remaining_houses
                else:
                    print(f"No suitable houses found for Household {household.unique_id}")

            pd.DataFrame(self.bidding_results).to_csv("../../results/bidding_results.csv", index=False)
    
    def negotiate_price(self, household, counter_offer):
        bid = household.lower_bound
        counter_offer = np.random.uniform(household.lower_bound, household.upper_bound)
        state = np.array([household.wtp]).reshape(1, -1)
        negotiation_prices = [(bid, counter_offer)]
        results = {}
        while bid <= counter_offer:
            action = np.argmax(household.model.predict(state))  
            bid_increase = bid * [0.02, 0.05, 0.1][action]  
            bid += bid_increase

            if bid >= counter_offer:
                return counter_offer, negotiation_prices  

            counter_offer -= counter_offer * np.random.uniform(0.02, 0.05)
            negotiation_prices.append((bid, counter_offer))
            final_price = min(bid, counter_offer)
        return final_price, negotiation_prices
    
    def rl_bidding(self, household):
        eligible_houses = self.houses[
            (self.houses['PUBVALUE'] > household.lower_bound) & 
            (self.houses['PUBVALUE'] <= household.upper_bound)
        ]
        results = {}
        for index, house in eligible_houses.iterrows():
            final_price, negotiation_prices = self.negotiate_price(household, house['PUBVALUE'])
            results[house['OBJECTID']] = final_price
            self. bidding_results.append({
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
        for state, action, reward, next_state in batch:
            target = reward + GAMMA * np.max(model.predict(next_state))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)