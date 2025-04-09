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
ALPHA = 0.001 
GAMMA = 0.9 
EPSILON = 0.1  
NEGOTIATION_PENALTY_FACTOR = 0.05
NUM_HOUSE_SAMPLE = 15

class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []

    def load_data(self, household_data_path="../../results/household_eligibility.csv", houses_path='../../inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.household_data.set_index("household_id", inplace=True)
        self.houses = pd.read_excel(houses_path)

    def run(self, households):
        self.load_data()
        rl_households = list(households.values())

        for household in rl_households:
            state = np.array([[household.wtp]])
            if np.random.rand() < EPSILON:
                action = np.random.choice(3)
            else:
                prediction = household.model.predict(state, verbose=0)
                action = np.argmax(prediction)

            reward = np.random.choice([1, -1])
            household.memory.append((state, action, reward, state))

            if len(household.memory) > 10:
                batch = random.sample(household.memory, 10)
                states = np.vstack([b[0] for b in batch])
                next_states = np.vstack([b[3] for b in batch])
                predictions_next = household.model.predict(next_states, verbose=0)
                targets = []
                for i, (s, a, r, ns) in enumerate(batch):
                    target = household.model.predict(s, verbose=0)
                    target[0][a] = r + GAMMA * np.max(predictions_next[i])
                    targets.append(target[0])
                household.model.fit(states, np.array(targets), epochs=1, verbose=0)

        for household in rl_households:
            print(f"Starting bids for Household ID: {household.unique_id}")
            results = self.rl_bidding(household)
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
                household.successful_deals += 1
                household.deals.append(chosen_house)

                print(f"Household {household.unique_id} chose house {chosen_house} at ${chosen_price:.2f}")
                remaining_houses = {h: p for h, p in chosen_houses.items() if h != chosen_house}
                print("Other three choices:", remaining_houses)

            else:
                    print(f"No suitable houses found for Household {household.unique_id}")

        pd.DataFrame(self.bidding_results).to_csv("../../results/model-deep-rl/bidding_results.csv", index=False)


    def sample_houses(self, eligible_houses):
        if len(eligible_houses) <= NUM_HOUSE_SAMPLE:
            return eligible_houses
        prices = eligible_houses['PUBVALUE']
        quantiles = prices.quantile([0.33, 0.66])
        low = eligible_houses[prices <= quantiles[0.33]].sample(5)
        mid = eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])].sample(5)
        high = eligible_houses[prices > quantiles[0.66]].sample(5)
        return pd.concat([low, mid, high])

    def negotiate_price(self, household, counter_offer):
        bid = household.lower_bound
        counter_offer = np.random.uniform(household.lower_bound, household.upper_bound)
        state = np.array([household.wtp]).reshape(1, -1)
        negotiation_prices = [(bid, counter_offer)]
        results = {}
        steps = 0
        while bid <= counter_offer:
            action = np.argmax(household.model.predict(state, verbose=0))
            bid += bid * [0.02, 0.05, 0.1][action]
            if bid >= counter_offer:
                break
            counter_offer -= counter_offer * np.random.uniform(0.02, 0.05)
            negotiation_prices.append((bid, counter_offer))
            steps += 1
        final_price = min(bid, counter_offer)
        reward = -steps * NEGOTIATION_PENALTY_FACTOR
        household.memory.append((state, action, reward, state))
        return final_price, negotiation_prices

    def rl_bidding(self, household):
        eligible = self.houses[
            (self.houses['PUBVALUE'] > household.lower_bound) &
            (self.houses['PUBVALUE'] <= household.upper_bound)
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
                'counter_offer': household.wtp,
                'negotiation_steps': len(negotiation),
                'negotiation_prices': str(negotiation)
            })
        return results