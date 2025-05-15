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
from household import Household
from house import House
from household_rl import RLHousehold
from bidding import BiddingGame
import matplotlib.pyplot as plt

class HousingMarket(mesa.Model):
    def __init__(self, years=10):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.housing_stock = []
        self.household_data = {}
        self.current_year = 0
        self.total_years = years
        self.init_data()
        self.init_housing_stock()
        self.episode = 0

    def init_data(self):
        self.household_data = pd.read_excel('../../inputs/HMTS.xlsx')
        self.parcel_data = pd.read_excel('../../inputs/Parcel.xlsx')

    def init_housing_stock(self):
        self.housing_stock = [House(row) for _, row in self.parcel_data.iterrows()]

    def update_market_conditions(self):
        inflation_rate = 0.03
        for house in self.housing_stock:
            house.price *= (1 + inflation_rate)
        self.household_data['wtp'] *= (1 + 0.02)

    def run_year(self, households, rl_game):
        self.episode += 1
        print(f"Running Episode {self.episode}")
        rl_game.run(households['rl'])
        
        # End episode for all RL households
        for household in households['rl'].values():
            household.end_episode()
            
        # Plot metrics for the first household (as a sample)
        if self.episode % 1 == 0:  # Plot every 1 episodes
            first_household = next(iter(households['rl'].values()))
            first_household.plot_metrics(save_path=f'../../output-analysis/model-deep-rl/metrics_episode_{self.episode}.png')

    def simulate(self):
        households = {
            'rl': {household_id: RLHousehold(household_id, self, self.household_data) 
                for household_id in self.household_data.index},
            'simulation': {household_id: Household(household_id, self, self.household_data)
                        for household_id in self.household_data.index}
        }

        rl_game = BiddingGame()

        for year in range(self.total_years):
            print(f"Year {year + 1} Simulation:")
            self.run_year(households, rl_game)
            self.update_market_conditions()