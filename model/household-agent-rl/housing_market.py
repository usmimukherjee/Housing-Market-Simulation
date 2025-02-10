import pandas as pd
import numpy as np
import mesa
from mesa.time import RandomActivation
from household import Household
from bidding import BiddingGame
from house import House

class HousingMarket(mesa.Model):
    def __init__(self, years=10):
        super().__init__()
        # self.schedule = RandomActivation(self)
        self.housing_stock = []
        self.household_data = {}
        self.current_year = 0
        self.total_years = years
        self.init_data()
        self.init_housing_stock()
    
    def step(self):
        self.agents.shuffle().do("step")
    
    def init_data(self):
        """Load initial household and housing data."""
        self.household_data = pd.read_excel('../../inputs/HMTS.xlsx')
        self.parcel_data = pd.read_excel('../../inputs/Parcel.xlsx')

    def init_housing_stock(self):
        """Initialize the housing stock from parcel data."""
        self.housing_stock = [House(row) for _, row in self.parcel_data.iterrows()]

    def update_market_conditions(self):
        """Simulate market changes for each year."""
        # Increase house prices due to inflation (+3% annually)
        inflation_rate = 0.03
        for house in self.housing_stock:
            house.price *= (1 + inflation_rate)

        # Adjust household WTP (+2% annually to simulate income increases)
        self.household_data['wtp'] *= (1 + 0.02)  


    def run_year(self, households, game):
        """Run the housing market for one year."""
        game.run(households)

    def simulate(self):
        """Simulate the housing market over multiple years."""
        households = {
            household_id: Household(household_id, self, self.household_data)
            for household_id in self.household_data.index
        }

        game = BiddingGame()

        for year in range(self.total_years):
            print(f"Year {year + 1} Simulation:")
            self.run_year(households, game)
            self.update_market_conditions() 

        self.evaluate_simulation(households)

    def evaluate_simulation(self, households):
        """Evaluate the performance of the households and market."""
        success_count = 0
        total_negotiations = 0
        total_prices = 0
        total_steps = 0

        for household in households.values():
            success_count += household.successful_deals
            total_negotiations += len(household.q_table)
            total_prices += sum(state[1] for state in household.deals) 
            total_steps += sum(len(actions) for actions in household.q_table.values())

        print("\nSimulation Results:")
        print(f"Total Successful Deals: {success_count}")
        print(f"Average Negotiated Price: ${total_prices / success_count if success_count > 0 else 0:.2f}")
        print(f"Average Steps per Negotiation: {total_steps / total_negotiations if total_negotiations > 0 else 0:.2f}")