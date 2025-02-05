import pandas as pd
import random

class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []

    def load_data(self, household_data_path="../../results/household_eligibility.csv", houses_path='../../inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.houses = pd.read_excel(houses_path)

    def negotiate_price(self, household, house_value):
        """Reinforcement learning-based negotiation."""
        bid = household.lower_bound
        counter_offer = house_value
        state = (bid, counter_offer)  # Initial state
        negotiation_prices = []

        while bid < counter_offer:
            action = household.choose_action(state)  # Choose action using RL policy
            bid_increase = bid * [0.02, 0.05, 0.10][action]  # Action determines bid increment
            bid += bid_increase

            if bid >= counter_offer:
                reward = 10  # Reward for successful deal
                household.update_q_table(state, action, reward, (bid, counter_offer))
                break

            counter_offer_decrease = counter_offer * random.uniform(0.02, 0.05)
            counter_offer -= counter_offer_decrease

            next_state = (bid, counter_offer)
            reward = -1  # Penalize for prolonged negotiation
            household.update_q_table(state, action, reward, next_state)
            state = next_state

            negotiation_prices.append((bid, counter_offer))

        final_price = min(bid, counter_offer)
        return final_price, negotiation_prices

    def bid_on_houses(self, household):
        eligible_houses = self.houses[
            (self.houses['PUBVALUE'] > household.lower_bound) & 
            (self.houses['PUBVALUE'] <= household.upper_bound)
        ]
        results = {}
        for index, house in eligible_houses.iterrows():
            final_price, negotiation_prices = self.negotiate_price(household, house['PUBVALUE'])
            results[house['OBJECTID']] = final_price
        return results

    def run(self, households):
        """Run the bidding process for all households."""
        self.load_data()
        for household_id, household in households.items():
            print(f"Starting bids for Household ID: {household.unique_id} with price range ${household.lower_bound:.2f} - ${household.upper_bound:.2f}")
            results = self.bid_on_houses(household)
            if results:
                print(f"Household {household.unique_id} successfully bid on houses: {results}")
            else:
                print(f"No suitable houses found for Household {household.unique_id}")