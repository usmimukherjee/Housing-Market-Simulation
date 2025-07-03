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
        self.household_data = pd.read_excel('./inputs/HMTS.xlsx')
        self.parcel_data = pd.read_excel('./inputs/Parcel.xlsx')

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
        
        # End episode for all RL households and reset metrics
        for household in households['rl'].values():
            household.end_episode()
            
        # Plot metrics for the first household
        if self.episode % 1 == 0:
            first_household = next(iter(households['rl'].values()))
            save_path = f'./output-analysis/model-deep-rl/variant__3/metrics_episode_{self.episode}.png'
            first_household.plot_metrics(save_path=save_path)
            print(f"Saved metrics visualization for episode {self.episode}")

    def simulate(self):
        # Create directories for outputs
        os.makedirs("./results/model-deep-rl/variant__3", exist_ok=True)
        os.makedirs("./output-analysis/model-deep-rl/variant__3", exist_ok=True)
        
        households = {
            'rl': {household_id: RLHousehold(household_id, self, self.household_data) 
                for household_id in self.household_data.index},
            'simulation': {household_id: Household(household_id, self, self.household_data)
                        for household_id in self.household_data.index}
        }

        rl_game = BiddingGame()

        for year in range(self.total_years):
            print(f"\n{'='*50}")
            print(f"Year {year + 1} Simulation")
            print(f"{'='*50}")
            self.run_year(households, rl_game)
            self.update_market_conditions()
            
            # Save intermediate results
            self.save_simulation_state(year, households)

    def save_simulation_state(self, year, households):
        # Save household eligibility data
        Household.save_eligibility_data(
            {**households['rl'], **households['simulation']},
            f"./results/model-deep-rl/variant__3/household_eligibility_year_{year+1}.csv"
        )
        
        # Save RL agent metrics
        rl_metrics = []
        for h in households['rl'].values():
            rl_metrics.append({
                'household_id': h.unique_id,
                'episode': year + 1,
                'total_reward': h.episode_rewards[-1] if h.episode_rewards else 0,
                'total_loss': h.episode_losses[-1] if h.episode_losses else 0,
                'successful_deals': h.successful_deals
            })
        pd.DataFrame(rl_metrics).to_csv(
            f"./results/model-deep-rl/variant__3/rl_metrics_year_{year+1}.csv", 
            index=False
        )