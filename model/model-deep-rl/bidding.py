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

# Set random seed for reproducibility
np.random.seed(0)

## To do:
# Pass bidding and counter offer in the state during model training and prediction
# How does the DRL maximize the reward over time? Is the total reward stored somewhere?
# Does it go back to the model for loss calculation and back propagation? add loss calculation
# Reward the intermediate behaviour if both parties are willing to compromise. You can also reward big at the end of a successful episode.
# hat is, given a certain state, the action involving the rewarding behaviour can be represented as class labels.

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

# Class to simulate the bidding process between households and houses
class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []
    # Load household eligibility data and available houses
    def load_data(self, household_data_path="./results/household_eligibility.csv", houses_path='./inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.household_data.set_index("household_id", inplace=True)
        self.houses = pd.read_excel(houses_path)
    
    def negotiate_price(self, household, house_value):
        bid = household.lower_bound
        counter_offer = house_value
        negotiation_prices = []
        state = np.array([[bid, counter_offer]])
        steps = 0

        while bid <= counter_offer:
            action_values = household.model.predict(state, verbose=0)
            action = np.argmax(action_values)
            increment = [0.02, 0.05, 0.1][action]
            new_bid = bid + bid * increment
            new_counter = counter_offer - counter_offer * np.random.uniform(0.02, 0.05)

            intermediate_reward = 0
            if new_bid > bid and new_counter < counter_offer:
                intermediate_reward = INTERMEDIATE_REWARD

            final = False
            if new_bid >= new_counter:
                final = True
                reward = 1 - steps * NEGOTIATION_PENALTY_FACTOR
            else:
                reward = intermediate_reward - NEGOTIATION_PENALTY_FACTOR

            next_state = np.array([[new_bid, new_counter]])
            household.memory.append((state, action, reward, next_state))

            state = next_state
            bid = new_bid
            counter_offer = new_counter
            negotiation_prices.append((bid, counter_offer))
            steps += 1

            if final:
                break

        final_price = min(bid, counter_offer)
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

    def train_model(self, household):
        if len(household.memory) >= 10:
            batch = random.sample(household.memory, 10)
            states = np.vstack([x[0] for x in batch])
            actions = [x[1] for x in batch]
            rewards = [x[2] for x in batch]
            next_states = np.vstack([x[3] for x in batch])

            q_values_next = household.model.predict(next_states, verbose=0)
            q_values = household.model.predict(states, verbose=0)

            for i in range(10):
                q_values[i][actions[i]] = rewards[i] + GAMMA * np.max(q_values_next[i])

            history = household.model.fit(states, q_values, epochs=1, verbose=0)
            household.update_metrics(np.mean(rewards), history.history['loss'][0])

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
    

    # Main method to run the simulation for a given set of households
    def run(self, households):
        self.load_data()
        rl_households = list(households.values())

        # Loop through each household to take RL actions and train
        for household in rl_households:
            # current state is the household's willingness to pay and the house value
            state = np.array([[household.wtp, household.lower_bound]])  # Initial state
            if np.random.rand() < EPSILON:
                action = np.random.choice(3)  # Exploration: choose random action
            else:
                state = np.reshape(state, (1, 2))
                prediction = household.model.predict(state, verbose=0)  # Exploitation: use model
                action = np.argmax(prediction)

            reward = np.random.choice([1, -1])  # Random reward (placeholder logic)
            household.memory.append((state, action, reward, state))  # Store experience

            # Train model if enough experiences are collected
            if len(household.memory) > 10:
                batch = random.sample(household.memory, 10)
                states = np.vstack([b[0] for b in batch])
                next_states = np.vstack([b[3] for b in batch])
                predictions_next = household.model.predict(next_states, verbose=0)
                targets = []
                for i, (s, a, r, ns) in enumerate(batch):
                    target = household.model.predict(s, verbose=0)
                    target[0][a] = r + GAMMA * np.max(predictions_next[i])  # Q-learning update
                    targets.append(target[0])
                # Train the model on the updated targets
                history = household.model.fit(states, np.array(targets), epochs=1, verbose=0)
                household.update_metrics(reward, history.history['loss'][0])  # Log metrics

        # After training, perform actual bidding
        for household in rl_households:
            print(f"Starting bids for Household ID: {household.unique_id}")
            results = self.rl_bidding(household, self.houses)
            if results:
                sorted_list = sorted(results.items(), key=lambda x: x[1])  # Sort by final price
                bins = max(1, len(sorted_list) // 4)
                # Pick 1 house from each quartile of the sorted list
                house_choices = [
                    random.choice(sorted_list[i * bins:(i + 1) * bins])
                    for i in range(4) if sorted_list[i * bins:(i + 1) * bins]
                ]

                # Randomly select one house among the 4 choices
                chosen_house, chosen_price = random.choice(house_choices)
                chosen_houses = {h_id: price for h_id, price in house_choices}
                remaining_houses = {h: p for h, p in chosen_houses.items() if h != chosen_house}

                # Store decision in household attributes
                household.chosen_house = chosen_house
                household.chosen_house_price = chosen_price
                household.house_choices = remaining_houses
                household.successful_deals += 1
                household.deals.append(chosen_house)

                print(f"Household {household.unique_id} chose house {chosen_house} at ${chosen_price:.2f}")
                print("Other three choices:", remaining_houses)
            else:
                print(f"No suitable houses found for Household {household.unique_id}")

        # Save results to CSV
        pd.DataFrame(self.bidding_results).to_csv("./results/model-deep-rl/bidding_results.csv", index=False)