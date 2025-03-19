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
PERCENTAGE_ABOVE_WTP = 1.2
PERCENTAGE_BELOW_WTP = 0.8

class Household(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)
        self.unique_id = unique_id
        self.wtp = self.calculate_wtp(household_data.loc[unique_id])
        household_data.loc[unique_id, 'wtp'] = self.wtp
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        self.successful_deals = 0

    def range_calculation(self, wtp):
        lower_bound = wtp * PERCENTAGE_BELOW_WTP
        upper_bound = wtp * PERCENTAGE_ABOVE_WTP
        return lower_bound, upper_bound
    
    def calculate_wtp(self, data):
        y_values = np.ones(10)
        self.wtp = (
            38.637 * data['birth of a child (x1)'] * y_values[0]
            - 12.309 * data.get('distance to CBD (x3) (km)', 0) * y_values[1]
            - 84.074 * data['distance to CBD (x3) (km)'] * y_values[2]
            - 66.961 * data.get('distance to nearest business center (x5) (km)', 0)
            + 79.441 * data['distance to nearest business center (x5) (km)'] * y_values[4]
            - 55.490 * data['distance to nearest school (x6) (km)'] * y_values[5]
            - 11.740 * data['distacne to nearest bus stop(km) (x7)']
            - 49.657 * data['distance to nearest park area (x8) (km)']
            + 0.064 * data['population density (x9)']
            + 2.162 * data['% owned property (x10)'] * 1000
        )
        return float(self.wtp)

    def rule_based_buying(self, houses):
        eligible_houses = [house for house in houses.values() if self.lower_bound <= house.price <= self.upper_bound]
        
        if eligible_houses:
            chosen_house = min(eligible_houses, key=lambda x: x.price)
            negotiation_prices = self.negotiate_price(chosen_house)
            
            chosen_house.status = "occupied"
            self.successful_deals += 1
            print(f"Household {self.unique_id} successfully bought House {chosen_house.id} for ${chosen_house.price:.2f}")
        print(f"Negotiation steps: {negotiation_prices}")
        
    def negotiate_price(self, house):
        """Simulates bidding negotiations using rule-based or RL-based strategies."""
        bid = self.lower_bound
        counter_offer = house.price
        negotiation_prices = [(bid, counter_offer)]
        
        while bid <= counter_offer:
            bid += bid * random.uniform(0.02, 0.1)  # Increase bid by 2%-10%
            
            if bid >= counter_offer:
                return negotiation_prices  # Deal is closed
            
            counter_offer -= counter_offer * random.uniform(0.02, 0.05)  # Seller adjusts price down
            negotiation_prices.append((bid, counter_offer))
        
        return negotiation_prices


    @staticmethod
    def save_eligibility_data(households, filename="../../results/household_eligibility.csv"):
        """Save household eligibility data (wtp, lower_bound, upper_bound) to a CSV file."""
        df = pd.DataFrame.from_dict(
            {
                h.unique_id: {"wtp": h.wtp, "lower_bound": h.lower_bound, "upper_bound": h.upper_bound}
                for h in households.values()
            },
            orient="index"
        )
        df.index.name = "household_id"
        df.to_csv(filename)
