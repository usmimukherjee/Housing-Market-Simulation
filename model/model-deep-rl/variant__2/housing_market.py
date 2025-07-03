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
from bidding import BiddingGame
from house import House
from household_rl import RLHousehold
from household import Household

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
        self.percentage_houses_for_sale = PERCENTAGE_HOUSES_FOR_SALE
        self.housing_turnover_rate = HOUSING_TURNOVER_RATE
        self.bidding_game = BiddingGame()

    def init_data(self):
        self.household_data = pd.read_excel('./inputs/HMTS.xlsx')
        self.parcel_data = pd.read_excel('./inputs/Parcel.xlsx')

    def init_housing_stock(self):
        self.housing_stock = [House(row) for _, row in self.parcel_data.iterrows()]

    def update_market_conditions(self):
        inflation_rate = 0.03
        # Update house prices based on market dynamics
        for house in self.housing_stock:
            # Appreciation based on years occupied and market conditions
            if house.status == "occupied":
                appreciation_rate = 0.02 + (house.years_occupied * 0.005)
                house.price *= (1 + appreciation_rate)
            elif house.status == "for sale":
                # Depreciation if on market too long
                house.price *= 0.99

    def run_year(self, households):
        self.episode += 1
        print(f"\n{'='*50}")
        print(f"Running Year {self.current_year + 1} (Episode {self.episode})")
        print(f"{'='*50}")
        
        # Reset market for new year
        self.update_market_conditions()
        self.reset_houses_for_sale()
        
        # Record initial market state
        year_stats = self.record_market_state(self.current_year + 1)
        
        # Print market state information
        print(f"\nMarket State for Year {self.current_year + 1}:")
        print(f"Total houses: {year_stats['total_houses']}")
        print(f"Houses for sale: {year_stats['for_sale']}")
        print(f"Occupied houses: {year_stats['occupied']}")
        print(f"Vacant houses: {year_stats['vacant']}")
        
        # Run bidding process
        print("\nStarting bidding process...")
        transactions = self.bidding_game.run(households, self)
        print(f"Completed transactions: {transactions}")
        
        # End episode for all RL households and reset metrics
        for household in households['rl'].values():
            household.end_episode()
            
        # Plot metrics for the first household
        if self.episode % 1 == 0 and households['rl']:
            first_household = next(iter(households['rl'].values()))
            save_path = f'./output-analysis/model-deep-rl/variant__2/metrics_episode_{self.episode}.png'
            first_household.plot_metrics(save_path=save_path)
            print(f"Saved metrics visualization for episode {self.episode}")
        
        # Update year stats
        year_stats.update({
            'transactions_completed': transactions,
            'avg_house_price': np.mean([h.price for h in self.housing_stock if h.price > 0]),
            'median_house_price': np.median([h.price for h in self.housing_stock if h.price > 0]),
            'successful_households': sum(1 for h in households['rl'].values() if h.chosen_house is not None)
        })
        
        # Print year-end statistics
        print(f"\nYear {self.current_year + 1} Statistics:")
        print(f"Average house price: ${year_stats['avg_house_price']:.2f}")
        print(f"Median house price: ${year_stats['median_house_price']:.2f}")
        print(f"Successful households: {year_stats['successful_households']}")
        
        # Reset households for next year (except those who got houses)
        self.reset_households(households)
        
        # Increment year counter
        self.current_year += 1
        
        return year_stats

    def simulate(self):
        # Create directories for outputs
        os.makedirs("./results/model-deep-rl/variant__2", exist_ok=True)
        os.makedirs("./output-analysis/model-deep-rl/variant__2", exist_ok=True)
        os.makedirs("./results/model-deep-rl/variant__2/market_state", exist_ok=True)
        
        households = {
            'rl': {household_id: RLHousehold(household_id, self, self.household_data) 
                  for household_id in self.household_data.index},
            'simulation': {household_id: Household(household_id, self, self.household_data)
                         for household_id in self.household_data.index}
        }

        # Track market statistics over time
        market_stats = []
        
        while self.current_year < self.total_years:
            year_stats = self.run_year(households)
            market_stats.append(year_stats)
            
            # Save intermediate results
            self.save_simulation_state(self.current_year - 1, households)
        
        # Save final market statistics
        pd.DataFrame(market_stats).to_csv("./results/model-deep-rl/variant__2/market_state/market_stats.csv", index=False)
        
        # Plot market dynamics
        self.plot_market_dynamics(market_stats)
        
        print("\nSimulation completed successfully!")

    def reset_houses_for_sale(self):
        """Set a percentage of houses as available for sale at the start of each year"""
        # Convert occupied houses to potential sellers based on turnover rate
        for house in self.housing_stock:
            if house.status == "occupied":
                if random.random() < self.housing_turnover_rate:
                    house.status = "for sale"
                else:
                    house.years_occupied += 1
            elif house.status == "vacant":
                # Always put vacant houses on the market
                house.status = "for sale"
        
        # Ensure at least the required percentage is for sale
        current_for_sale = sum(1 for h in self.housing_stock if h.status == "for sale")
        target_for_sale = int(len(self.housing_stock) * self.percentage_houses_for_sale)
        
        if current_for_sale < target_for_sale:
            # Convert additional vacant houses to for sale
            vacant_houses = [h for h in self.housing_stock if h.status == "vacant"]
            if vacant_houses:
                additional_needed = target_for_sale - current_for_sale
                additional_for_sale = random.sample(vacant_houses, min(additional_needed, len(vacant_houses)))
                for house in additional_for_sale:
                    house.status = "for sale"

    def get_houses_for_sale(self):
        """Return DataFrame of houses currently for sale"""
        for_sale = []
        for house in self.housing_stock:
            if house.status == "for sale":
                for_sale.append({
                    'OBJECTID': house.id,
                    'PUBVALUE': house.price,
                    'original_price': house.original_price,
                    'status': house.status
                })
        return pd.DataFrame(for_sale)

    def record_market_state(self, year):
        """Record current market state for analysis"""
        market_data = []
        for house in self.housing_stock:
            market_data.append({
                'year': year,
                'house_id': house.id,
                'status': house.status,
                'price': house.price,
                'original_price': house.original_price,
                'years_occupied': house.years_occupied
            })
        
        # Create directory if needed
        os.makedirs("./results/model-deep-rl/variant__2/market_state", exist_ok=True)
        pd.DataFrame(market_data).to_csv(f"./results/model-deep-rl/variant__2/market_state/year_{year}.csv", index=False)
        
        # Calculate market statistics
        total_houses = len(self.housing_stock)
        for_sale = sum(1 for h in self.housing_stock if h.status == "for sale")
        occupied = sum(1 for h in self.housing_stock if h.status == "occupied")
        vacant = sum(1 for h in self.housing_stock if h.status == "vacant")
        
        return {
            'year': year,
            'total_houses': total_houses,
            'for_sale': for_sale,
            'occupied': occupied,
            'vacant': vacant,
            'for_sale_pct': for_sale / total_houses if total_houses > 0 else 0,
            'vacancy_rate': vacant / total_houses if total_houses > 0 else 0
        }

    def save_simulation_state(self, year, households):
        # Save household eligibility data
        Household.save_eligibility_data(
            {**households['rl'], **households['simulation']},
            f"./results/model-deep-rl/variant__2/household_eligibility_year_{year+1}.csv"
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
            f"./results/model-deep-rl/variant__2/rl_metrics_year_{year+1}.csv", 
            index=False
        )

    def reset_households(self, households):
        """Reset household states for a new year"""
        for household_type in ['rl', 'simulation']:
            for household in households[household_type].values():
                # Reset chosen house and price
                household.chosen_house = None
                household.chosen_house_price = None
                
                # Update WTP based on market conditions
                household.wtp *= (1 + np.random.uniform(0.01, 0.03))
                household.lower_bound = household.wtp * PERCENTAGE_BELOW_WTP
                household.upper_bound = household.wtp * PERCENTAGE_ABOVE_WTP
                
                # Reset successful deals counter for the new year
                household.successful_deals = 0
                household.deals = []

    def plot_market_dynamics(self, market_stats):
        """Visualize market dynamics over time"""
        df = pd.DataFrame(market_stats)
        
        plt.figure(figsize=(14, 10))
        
        # Market composition plot
        plt.subplot(2, 2, 1)
        plt.stackplot(df['year'], 
                     df['for_sale'], 
                     df['occupied'], 
                     df['vacant'],
                     labels=['For Sale', 'Occupied', 'Vacant'])
        plt.title('Housing Market Composition')
        plt.xlabel('Year')
        plt.ylabel('Number of Houses')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Price trends
        plt.subplot(2, 2, 2)
        plt.plot(df['year'], df['avg_house_price'], 'b-', label='Average Price')
        plt.plot(df['year'], df['median_house_price'], 'r--', label='Median Price')
        plt.title('House Price Trends')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Market rates
        plt.subplot(2, 2, 3)
        plt.plot(df['year'], df['for_sale_pct'], 'g-', label='For Sale %')
        plt.plot(df['year'], df['vacancy_rate'], 'm-', label='Vacancy Rate')
        plt.title('Market Indicators')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        
        # Transactions
        plt.subplot(2, 2, 4)
        plt.bar(df['year'], df['transactions_completed'], color='orange')
        plt.title('Completed Transactions')
        plt.xlabel('Year')
        plt.ylabel('Number of Sales')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("./output-analysis/model-deep-rl/variant__2/market_dynamics.png")
        plt.close()