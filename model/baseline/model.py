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
        # Calculates the lower and upper bounds of the price range for a house based on WTP
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
        self.household_data = pd.read_excel('../inputs/HMTS.xlsx')
        self.parcel_data = pd.read_excel('../inputs/Parcel.xlsx')

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
        with open('../results/household_eligibility.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['household_id', 'wtp', 'lower_bound', 'upper_bound'])
            for household in self.schedule.agents:
                writer.writerow([household.unique_id, household.wtp, household.lower_bound, household.upper_bound])

class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []

    def load_data(self, household_data_path="../results/household_eligibility.csv", houses_path='../inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.houses = pd.read_excel(houses_path)

    def negotiate_price(self, household, house_value):
        # Negotiates between household's bid and seller's counter-offer until an agreement is reached.
        bid = household['lower_bound']
        counter_offer = house_value
        negotiation_prices = []  # Track negotiation prices
        
        while bid < counter_offer:
            negotiation_prices.append((bid, counter_offer))
            bid_increase = bid * random.uniform(0.02, 0.05)
            bid += bid_increase
            if bid >= counter_offer:
                break
            counter_offer_decrease = counter_offer * random.uniform(0.02, 0.05)
            counter_offer -= counter_offer_decrease

        negotiation_prices.append((bid, counter_offer))  # Final negotiation prices
        final_price = min(bid, counter_offer)
        return final_price, negotiation_prices

    def bid_on_houses(self, household):
        # Processes bids for all eligible houses for a given household
        eligible_houses = self.houses[
            (self.houses['PUBVALUE'] > household['lower_bound']) & 
            (self.houses['PUBVALUE'] <= household['upper_bound'])
        ]
        results = {}
        for index, house in eligible_houses.iterrows():
            final_price, negotiation_prices = self.negotiate_price(household, house['PUBVALUE'])
            results[house['OBJECTID']] = final_price
            print(f"Negotiation Prices for House ID {house['OBJECTID']}: {negotiation_prices}")
            print(f"Final negotiated price for House ID {house['OBJECTID']}: ${final_price}")

        # Add houses slightly beyond the upper bound
        beyond_upper_bound_houses = self.houses[
            (self.houses['PUBVALUE'] > household['upper_bound']) &
            (self.houses['PUBVALUE'] <= household['upper_bound'] * 1.2)  # Up to 20% above the upper bound
        ]
        for index, house in beyond_upper_bound_houses.iterrows():
            final_price, negotiation_prices = self.negotiate_price(household, house['PUBVALUE'])
            results[house['OBJECTID']] = final_price
            print(f"Negotiation Prices for House ID {house['OBJECTID']} (beyond upper bound): {negotiation_prices}")
            print(f"Final negotiated price for House ID {house['OBJECTID']}: ${final_price}")

        return results

    def choose_four_options(self, bidding_results, household):
        # Select four houses based on specified criteria
        sorted_houses = sorted(bidding_results.items(), key=lambda x: x[1])  # Sort by price
        options = {}
        lower_bound = household['lower_bound']
        upper_bound = household['upper_bound']

        # Select house close to lower bound
        options['Close to Lower Bound'] = next(
            (house for house in sorted_houses if house[1] >= lower_bound), None
        )

        # Select house close to mid of range
        mid_range = (lower_bound + upper_bound) / 2
        options['Mid Range'] = next(
            (house for house in sorted_houses if house[1] >= mid_range), None
        )

        # Select house close to upper bound
        options['Close to Upper Bound'] = next(
            (house for house in sorted_houses if upper_bound * 0.9 <= house[1] <= upper_bound), None
        )

        # Select house 10–20% beyond upper bound
        options['Beyond Upper Bound'] = next(
            (house for house in sorted_houses if upper_bound < house[1] <= upper_bound * 1.2), None
        )

        # Remove None entries
        options = {k: v for k, v in options.items() if v is not None}

        print("Chosen Options:")
        for key, value in options.items():
            print(f"{key}: House ID {value[0]} with price ${value[1]}")

        return options

    def run(self):
        # Runs the bidding process for each household
        self.load_data()
        for index, household in self.household_data.iterrows():
            print(f"Starting bids for Household ID: {household['household_id']} with price range ${household['lower_bound']} - ${household['upper_bound']}")
            
            start_time = time.time()  # Start timing
            
            results = self.bid_on_houses(household)
            
            end_time = time.time()  # End timing
            time_taken = end_time - start_time
            print(f"Time taken for Household ID {household['household_id']}: {time_taken:.2f} seconds")
            
            if results:
                chosen_options = self.choose_four_options(results, household)
                self.bidding_results.append((household['household_id'], chosen_options))
            else:
                print(f"No suitable house found for Household ID: {household['household_id']}")
            
            self.save_results()
            break 

    def save_results(self):
        # Saves the final bidding results to a CSV file
        with open('../results/BiddingResults.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Household ID', 'Options'])
            for result in self.bidding_results:
                writer.writerow(result)

if __name__ == "__main__":
    model = HousingMarket()
    game = BiddingGame()
    game.run()
