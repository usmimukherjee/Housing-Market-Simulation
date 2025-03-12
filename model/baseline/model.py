import pandas as pd
import numpy as np
import mesa
import os
from mesa.time import RandomActivation
import csv
import random
import warnings
import time
ignore_warnings = warnings.filterwarnings("ignore")
np.random.seed(0)

PERCENTAGE_ABOVE_WTP = 1.2
PERCENTAGE_BELOW_WTP = 0.8

class Household(mesa.Agent):
    
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)
        self.unique_id = unique_id
        self.wtp = self.calculate_wtp(household_data.loc[unique_id])
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        # self.eligible_houses = []

    def range_calculation(self, wtp):
        # Calculates the lower and upper bounds of the price range for a household based on WTP
        lower_bound = wtp * PERCENTAGE_BELOW_WTP
        upper_bound = wtp * PERCENTAGE_ABOVE_WTP
        return lower_bound, upper_bound

    def calculate_wtp(self, data):
            # Calculates WTP using a model based on various household attributes and weights
            y_values = [1] * 10 
            self.wtp = (38.637 * data['birth of a child (x1)'] * y_values[0]
                        - 12.309 * data.get('distance to CBD (x3) (km)', 0) * y_values[1]
                        - 84.074 * data['distance to CBD (x3) (km)'] * y_values[2]
                        - 66.961 * data.get('distance to nearest business center (x5) (km)', 0)
                        + 79.441 * data['distance to nearest business center (x5) (km)'] * y_values[4]
                        - 55.490 * data['distance to nearest school (x6) (km)'] * y_values[5]
                        - 11.740 * data['distacne to nearest bus stop(km) (x7)']
                        - 49.657 * data['distance to nearest park area (x8) (km)']
                        + 0.064 * data['population density (x9)']
                        + 2.162 * data['% owned property (x10)'] * 1000)
            return self.wtp
    
class House:
    next_id = 1
    def __init__(self, data):
        self.id = data['OBJECTID']
        self.status = "vacant"
        self.price = data['PUBVALUE']

class HousingMarket(mesa.Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.housing_stock = []
        self.init_data()
        self.init_housing_stock()
        self.process_households()
        self.log_data()

    def init_data(self):
        # Load data from Excel files
        self.household_data = pd.read_excel('../../inputs/HMTS.xlsx')
        self.parcel_data = pd.read_excel('../../inputs/Parcel.xlsx')

    def init_housing_stock(self):
        # Initialize the housing stock from parcel data
        self.housing_stock = [House(row) for _, row in self.parcel_data.iterrows()]

    def process_households(self):
        # Process each household, calculate WTP and bounds, and add to schedule
        for household_id in self.household_data.index:
            household = Household(household_id, self, self.household_data)
            self.schedule.add(household)
            household.step()

    def log_data(self):
        # Log data about the households and their eligibility
        with open('../../results/household_eligibility.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['household_id', 'wtp', 'lower_bound', 'upper_bound'])
            for household in self.schedule.agents:
                writer.writerow([household.unique_id, household.wtp, household.lower_bound, household.upper_bound])


class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []

    def load_data(self, household_data_path="../../results/household_eligibility.csv", houses_path='../../inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.houses = pd.read_excel(houses_path)

    def negotiate_price(self, household, house_value):
        bid = float(household['lower_bound'])  # Explicitly convert to Python float
        counter_offer = float(house_value)  # Explicitly convert to Python float
        negotiation_prices = []
        
        while bid < counter_offer:
            negotiation_prices.append((bid, counter_offer))
            bid_increase = bid * random.uniform(0.02, 0.05)
            bid += bid_increase
            
            if bid >= counter_offer:
                break
            
            counter_offer_decrease = counter_offer * random.uniform(0.02, 0.05)
            counter_offer -= counter_offer_decrease
        
        negotiation_prices.append((bid, counter_offer))
        final_price = min(bid, counter_offer)
        return final_price, negotiation_prices


    def bid_on_houses(self, household):
        eligible_houses = self.houses[
            (self.houses['PUBVALUE'] > household['lower_bound']) & 
            (self.houses['PUBVALUE'] <= household['upper_bound'])
        ]
        
        for _, house in eligible_houses.iterrows():
            final_price, negotiation_prices = self.negotiate_price(household, house['PUBVALUE'])
            self.bidding_results.append({
                'household_id': household['household_id'],
                'house_id': house['OBJECTID'],
                'initial_bid': household['lower_bound'],
                'final_price': final_price,
                'counter_offer': house['PUBVALUE'],
                'negotiation_steps': len(negotiation_prices),
                'negotiation_prices': str(negotiation_prices)  
            })

    def run(self):
        self.load_data()
        for _, household in self.household_data.iterrows():
            self.bid_on_houses(household)
        self.save_results()

    def save_results(self):
        with open('../../results/model-baseline/bidding_results.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['household_id', 'house_id', 'initial_bid', 'final_price', 'counter_offer', 'negotiation_steps', 'negotiation_prices'])
            writer.writeheader()
            writer.writerows(self.bidding_results)
if __name__ == "__main__":
    model = HousingMarket()
    game = BiddingGame()
    game.run()
