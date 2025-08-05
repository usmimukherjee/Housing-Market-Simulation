import pandas as pd
import numpy as np
import random
import os
import time
import mesa
from collections import defaultdict

# Constants
INFLATION_RATE = 0.044
PURCHASING_POWER_INCREASE = 0.034
PERCENTAGE_ABOVE_WTP = 1.2
PERCENTAGE_BELOW_WTP = 0.8

SNLR_BY_YEAR = {
    2015: 0.54,
    2016: 0.57,
    2017: 0.59,
    2018: 0.62,
    2019: 0.65,
    2020: 0.68,
    2021: 0.72,
    2022: 0.75,
    2023: 0.70,
    2024: 0.65,
    2025: 0.63  
}

class House:
    def __init__(self, data):
        self.id = data['Assessment Account Number']
        self.price = data['Assessed Value']

class Household(mesa.Agent):
    def __init__(self, unique_id, model, data_row):
        super().__init__(model)
        self.unique_id = unique_id
        self.data = data_row
        self.wtp = self.calculate_wtp(data_row)
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        self.chosen_house = None
        self.successful_deals = 0
        self.deals = []

    def calculate_wtp(self, data):
        y_values = np.ones(10)
        return float(
            38.637 * data.get('birth of a child (x1)', 0) * y_values[0] -
            12.309 * data.get('distance to CBD (x3) (km)', 0) * y_values[1] -
            84.074 * data.get('distance to CBD (x3) (km)', 0) * y_values[2] -
            66.961 * data.get('distance to nearest business center (x5) (km)', 0) +
            79.441 * data.get('distance to nearest business center (x5) (km)', 0) * y_values[4] -
            55.490 * data.get('distance to nearest school (x6) (km)', 0) * y_values[5] -
            11.740 * data.get('distacne to nearest bus stop(km) (x7)', 0) -
            49.657 * data.get('distance to nearest park area (x8) (km)', 0) +
            0.064 * data.get('population density (x9)', 0) +
            2.162 * data.get('% owned property (x10)', 0) * 1000
        )

    def range_calculation(self, wtp):
        return wtp * PERCENTAGE_BELOW_WTP, wtp * PERCENTAGE_ABOVE_WTP

class BiddingGame:
    def __init__(self, scenario, demand_shock_year=None, demand_multiplier=1.0):
        self.bidding_results = []
        self.scenario = scenario
        self.demand_shock_year = demand_shock_year
        self.demand_multiplier = demand_multiplier


    def negotiate_price(self, household, house_value, current_year=0):
        """Rule-based negotiation: simple iterative price matching with seller flexibility."""
        bid = household.lower_bound
        counter_offer = house_value
        negotiation_prices = []
        steps = 0
        done = False

        BID_INCREMENT = np.random.uniform(0.01, 0.05) 
        MAX_STEPS = 50       

        year = 2010 + current_year
        snlr = SNLR_BY_YEAR.get(year, 0.6)
        seller_flexibility =  np.random.uniform(0.01, 0.05) * (1 - snlr) * (1 + 0.05)  

        while not done and steps < MAX_STEPS:
            negotiation_prices.append((bid, counter_offer))

            if bid >= counter_offer:
                done = True
                agreed_price = bid
                final_price = agreed_price
                household.successful_deals += 1
                household.deals.append(final_price)
                return final_price, negotiation_prices

            # Buyer increases bid
            bid += bid * BID_INCREMENT

            # Seller reduces price
            counter_offer -= counter_offer * seller_flexibility
            steps += 1

        # If no deal reached
        return None, negotiation_prices



    def run(self, households, houses, year):
        base_sample_size = int(0.4 * len(households))
        if year == self.demand_shock_year:
            sample_size = int(base_sample_size * self.demand_multiplier)
        else:
            sample_size = base_sample_size

        sub_households = dict(random.sample(list(households.items()), sample_size))

        sub_houses = random.sample(houses, int(0.4 * len(houses)))
        cleared_households = set()
        cleared_houses = set()

        for h_id, household in sub_households.items():
            eligible = [h for h in sub_houses if household.lower_bound <= h.price <= household.upper_bound]
            eligible = sorted(eligible, key=lambda h: h.price, reverse=True)[:10]

            for house in eligible:
                final_price, negotiation = self.negotiate_price(household, house.price)
                self.bidding_results.append({
                    'year': year,
                    'household_id': household.unique_id,
                    'house_id': house.id,
                    'initial_bid': household.lower_bound,
                    'final_price': final_price,
                    'counter_offer': house.price,
                    'negotiation_steps': len(negotiation),
                    'negotiation_prices': str(negotiation)
                })
                cleared_households.add(h_id)
                cleared_houses.add(house.id)
                break

        pd.DataFrame(self.bidding_results).to_csv(
            f'./results-scenario/demand-shock/baseline/bidding_results_year_{year}.csv',
            index=False
        )

class HousingMarket(mesa.Model):
    def __init__(self, years=10, freeze_year=None, demand_shock_year=None, demand_multiplier=1.0):
        super().__init__()
        self.years = years
        self.current_year = 0
        self.household_data = pd.read_excel('./inputs/HMTS.xlsx')
        self.model_households = {}
        self.housing_stock = []
        self.freeze_year = freeze_year
        self.demand_shock_year = demand_shock_year
        self.demand_multiplier = demand_multiplier
        self.scenario = "demandshock" if demand_shock_year else "baseline"

    def update_market_conditions(self):
            inflation = INFLATION_RATE
            for house in self.housing_stock:
                house.price *= (1 + inflation)
    def init_housing_stock(self, year):
        if year <= 2025:
            load_year = self.freeze_year if self.freeze_year and year >= self.freeze_year else year
        else:
            load_year = 2025  # reuse final available year

        path = f'./inputs/parcel-yearwise/parcel_data_{load_year}.csv'
        parcel_df = pd.read_csv(path, low_memory=False)

        self.housing_stock = [House(row) for _, row in parcel_df.iterrows()]

    def simulate(self):
        self.model_households = {
            hid: Household(hid, self, self.household_data.loc[hid])
            for hid in self.household_data.index
        }
        self.save_eligibility()
        game = BiddingGame(
            scenario=self.scenario,
            demand_shock_year=self.demand_shock_year,
            demand_multiplier=self.demand_multiplier
        )


        # Extend simulation to include 2025 (11 years from 2015)
        for year_offset in range(self.years + 1):  # +1 to include 2025
            self.current_year = 2010 + year_offset
            print(f"\n=== Running Year: {self.current_year} ===")
            self.init_housing_stock(self.current_year)
            self.update_market_conditions()
            game.run(self.model_households, self.housing_stock, year=self.current_year)

            # Update household parameters annually
            for h in self.model_households.values():
                h.chosen_house = None
                h.wtp *= (1 + PURCHASING_POWER_INCREASE)
                h.lower_bound, h.upper_bound = h.range_calculation(h.wtp)

    def save_eligibility(self):
        df = pd.DataFrame.from_dict(
            {h.unique_id: {"wtp": h.wtp, "lower_bound": h.lower_bound, "upper_bound": h.upper_bound}
             for h in self.model_households.values()},
            orient="index"
        )
        df.index.name = "household_id"
        df.to_csv(f"./results-scenario/demand-shock/baseline/household_eligibility.csv")


if __name__ == "__main__":
    print("Running demand shock housing market model...")
    model = HousingMarket(
        years=15,
        demand_shock_year=2020,       
        demand_multiplier=2.0         
    )
    model.simulate()

