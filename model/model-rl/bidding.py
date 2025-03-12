import pandas as pd
import random
import numpy as np

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
        state = (bid, counter_offer)
        negotiation_prices = []

        while bid <= counter_offer:
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
            self.bidding_results.append({
                'household_id': household.unique_id,
                'house_id': house['OBJECTID'],
                'final_price': final_price,
                'negotiation_prices': negotiation_prices
            })
        return results

    def run(self, households):
        """Run the bidding process for all households."""
        self.load_data()
        for household_id, household in households.items():
            print(f"Starting bids for Household ID: {household.unique_id} with price range ${household.lower_bound:.2f} - ${household.upper_bound:.2f}")
            results = self.bid_on_houses(household)
            if results:
                # print(f"Household {household.unique_id} successfully bid on houses: {results}")
                # Sort houses by price
                house_list = sorted(results.items(), key=lambda x: x[1])  # [(house_id, price), ...]
                house_ids, house_prices = zip(*house_list)

                # Divide into four price bins
                bin_size = max(1, len(house_ids) // 4)  # Avoid division by zero
                house_choices = [(random.choice(house_list[i * bin_size:(i + 1) * bin_size])) for i in range(4) if house_list[i * bin_size:(i + 1) * bin_size]]

                # Extract house IDs and prices
                chosen_houses = {h_id: price for h_id, price in house_choices}

                # Randomly choose one final house
                chosen_house, chosen_price = random.choice(house_choices)
                household.chosen_house = chosen_house
                household.chosen_house_price = chosen_price

                # Store choices in household data
                household.house_choices = chosen_houses
                self.household_data.loc[household.unique_id, ['house_choices', 'chosen_house', 'chosen_house_price']] = [
                    str(chosen_houses), chosen_house, chosen_price
                ]

                # Track successful deals
                household.successful_deals += 1
                household.deals.append(chosen_house)

                # Print results
                print(f"Household {household.unique_id} chose house {chosen_house} at ${chosen_price:.2f}")
                remaining_houses = {h: p for h, p in chosen_houses.items() if h != chosen_house}
                print("Other three choices:", remaining_houses)
                
                household.chosen_house = chosen_house
                household.house_choices = remaining_houses
            else:
                print(f"No suitable houses found for Household {household.unique_id}")