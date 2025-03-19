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

    def run(self, households):
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
            print(f"\nHousehold {household.unique_id} begins bidding...")
            final_price, negotiation_prices = self.rl_bidding(household)
            print(f"Final agreed price: ${final_price:.2f}")
            print(f"Negotiation history: {negotiation_prices}")
    
    def rl_bidding(self, household):
        bid = household.lower_bound
        counter_offer = np.random.uniform(household.lower_bound, household.upper_bound)
        state = np.array([household.wtp]).reshape(1, -1)
        negotiation_prices = [(bid, counter_offer)]

        while bid <= counter_offer:
            action = np.argmax(household.model.predict(state))  
            bid_increase = bid * [0.02, 0.05, 0.1][action]  
            bid += bid_increase

            if bid >= counter_offer:
                return counter_offer, negotiation_prices  

            counter_offer -= counter_offer * np.random.uniform(0.02, 0.05)
            negotiation_prices.append((bid, counter_offer))
        
        return min(bid, counter_offer), negotiation_prices

    def train(self, model, batch):
        for state, action, reward, next_state in batch:
            target = reward + GAMMA * np.max(model.predict(next_state))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)