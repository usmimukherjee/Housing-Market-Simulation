import pandas as pd
import random
import numpy as np
import time
import mesa
import warnings
import os
from collections import defaultdict

ignore_warnings = warnings.filterwarnings("ignore")
np.random.seed(0)

# Constant as per data ====
INFLATION_RATE = 0.044
PURCHASING_POWER_INCREASE = 0.034
PERCENTAGE_ABOVE_WTP = 1.2  
PERCENTAGE_BELOW_WTP = 0.8 
SNLR_BY_YEAR = {
    2015: 0.54,
    2016: 0.62,
    2017: 0.70,
    2018: 0.75,
    2019: 0.78,
    2020: 0.82,
    2021: 0.85,
    2022: 0.88,
    2023: 0.75,
    2024: 0.65,
    2025: 0.68
}

PURCHASING_POWER_INCREASE = 0.034 
# Hyperparameters for the Q-learning reward system ======
# Reward function
FINAL_SUCCESS_REWARD = 6.0      # encourages closing deals
INTERMEDIATE_REWARD = 1.5       # encourage convergence
NEGOTIATION_PENALTY_FACTOR = 0.1  # smaller penalty allows more negotiation

# Learning hyperparameters
ALPHA = 0.4                    # learn faster
GAMMA = 0.95                    # weigh future rewards more
EPSILON = 0.3                   # more exploration initially

#Agents and Game Classes ======

class BiddingGame:
    def __init__(self,model):
        self.model = model
        self.house_list = {}
        self.bidding_results = []
        self.last_year_median_price = 0

    def load_data(self, year, household_data_path="./results-scenario/combined-shock/rl/household_eligibility.csv", houses_path='./inputs/parcel-yearwise/'):
        self.household_data = pd.read_csv(household_data_path)
        
        # Handle year loading with proper bounds checking
        if year > 2025:
            print(f"Year {year} exceeds available data, using 2025 data with inflation adjustment")
            load_year = 2025
        else:
            load_year = year
            
        try:
            self.houses = pd.read_csv(f'{houses_path}/parcel_data_{load_year}.csv')
            # Apply inflation for years beyond available data
            if year > 2025:
                years_diff = year - 2025
                inflation_multiplier = (1 + INFLATION_RATE) ** years_diff
                self.houses['Assessed Value'] *= inflation_multiplier
                print(f"Applied {years_diff} years of inflation ({inflation_multiplier:.3f}x) to housing values")
        except FileNotFoundError:
            print(f"Warning: No data file for year {load_year}, using 2025 data")
            self.houses = pd.read_csv(f'{houses_path}/parcel_data_2025.csv')
            if year > 2025:
                years_diff = year - 2025
                inflation_multiplier = (1 + INFLATION_RATE) ** years_diff
                self.houses['Assessed Value'] *= inflation_multiplier

    def negotiate_price(self, household, house_value, current_year=0):
        """Reinforcement learning-based negotiation with DRL reward system."""
        bid = household.lower_bound
        counter_offer = house_value
        state = (bid, counter_offer)
        negotiation_prices = []
        steps = 0
        total_reward = 0
        done = False

        # Constants
        BID_INCREMENT_GROWTH_RATE = 0.05
        FINAL_PRICE_INFLATION_FLOOR = 1.04

        bid_increments = np.random.uniform(0.08, 0.12, size=3)  # Random bid increments between 8% and 10%

        while not done and steps < 50:
            action = household.choose_action(state)

            base_increment = bid_increments[action]
            increment = base_increment * ((1 + BID_INCREMENT_GROWTH_RATE) ** current_year)
            prev_bid = bid
            prev_counter = counter_offer
            bid += bid * increment

            # SNLR-adjusted seller flexibility
            year = 2010 + current_year

            # Adjust house_value based on market pressure
            if year in SNLR_BY_YEAR:
                adjusted_house_value = house_value * SNLR_BY_YEAR.get(year, 1.0)
            else:
                adjusted_house_value = house_value
            counter_offer = adjusted_house_value
            # Replace the existing seller flexibility calculation with:
            snlr = SNLR_BY_YEAR.get(year, 0.6)
            base_flexibility = 0.12  # Increased from 0.08
            seller_flexibility = base_flexibility * (1 - snlr) * (1 + 0.05 * current_year)
            counter_offer_decrease = counter_offer * seller_flexibility
            counter_offer -= counter_offer_decrease

            reward = 0
            if bid > prev_bid and counter_offer < prev_counter:
                reward += INTERMEDIATE_REWARD

            if bid >= counter_offer:
                agreed_price = (bid + counter_offer) / 2

                target_price = household.lower_bound * (1.03 ** current_year)
                if agreed_price >= target_price:
                    reward += 0.5

                inflation_target = household.lower_bound * (1.044 ** current_year)
                if agreed_price >= inflation_target:
                    reward += 0.3

                reward += FINAL_SUCCESS_REWARD - steps * NEGOTIATION_PENALTY_FACTOR
                done = True
            else:
                reward -= NEGOTIATION_PENALTY_FACTOR

            next_state = (bid, counter_offer)
            total_reward += reward
            household.update_q_table(state, action, reward, next_state)
            state = next_state
            negotiation_prices.append((bid, counter_offer))
            steps += 1

        if done:
            final_price = min(bid, counter_offer)
            min_price = household.lower_bound * (FINAL_PRICE_INFLATION_FLOOR ** current_year)
            final_price = max(final_price, min_price)
            household.successful_deals += 1
            household.deals.append(final_price)
        else:
            final_price = None

        return final_price, negotiation_prices

    def bid_on_houses(self, household, year):
        # For now, assume eligible_houses are all houses within the price range of household
        eligible_houses = self.houses[
            (self.houses['Assessed Value'] >= household.lower_bound) & 
            (self.houses['Assessed Value'] <= household.upper_bound)
        ]
        # Limit to at most 15 eligible houses
        eligible_houses = eligible_houses.sort_values(by='Assessed Value', ascending=False).head(15)

        results = {}
        for index, house in eligible_houses.iterrows():
            final_price, negotiation_prices = self.negotiate_price(household, house['Assessed Value'])
            if final_price is not None:  # Only add successful negotiations
                household.chosen_house = house['Assessment Account Number']
                results[house['Assessment Account Number']] = final_price
                current_year = 2010 + year
                self.bidding_results.append({
                'year': current_year,
                'household_id': household.unique_id,
                'house_id': house['Assessment Account Number'],
                'initial_bid': household.lower_bound,
                'final_price': final_price,
                'counter_offer': house['Assessed Value'],
                'negotiation_steps': len(negotiation_prices),
                'negotiation_prices': str(negotiation_prices)  
            })

        return results

    def run(self, households, max_cycles=2, year=0):  # Reduced from 3 to 2 cycles for better performance
        """Market clearing process with sub-market cycles based on Farooq and Miller (2012)."""
        simulation_year = 2010 + year      
        print(f"Starting market clearing for year {simulation_year}")
        self.load_data(year=simulation_year)
        
        SUBMARKET_FRACTION = 0.45
        num_households = int(len(households) * SUBMARKET_FRACTION)
        num_houses = int(len(self.houses) * SUBMARKET_FRACTION)

        all_households = households.copy()
        available_houses = self.houses.copy()
        cycle = 0
        
        # Cycle count (max_cycles) controls how long the market clearing lasts per year
        while cycle < max_cycles:
            print(f"\n--- Sub-Market Cycle {cycle + 1}/{max_cycles} ---")
            # Randomly sample sub-market participants
            sub_households = dict(random.sample(list(all_households.items()), num_households))
            sub_houses = available_houses.sample(n=num_houses).copy()

            cleared_households = set()
            cleared_houses = set()

            # Replace house pool for this round
            self.houses = sub_houses

            # Add safety counter to prevent infinite loops
            safety_counter = 0
            max_iterations = 100  # Reduced from 1000 for better performance
            
            while (len(cleared_households) < num_households // 2) and (len(cleared_houses) < num_houses // 2) and safety_counter < max_iterations:
                safety_counter += 1
                
                # Process only a subset of households per iteration to improve performance
                households_to_process = list(sub_households.items())[:min(20, len(sub_households))]
                
                for h_id, household in households_to_process:
                    if len(cleared_households) >= num_households // 2 or len(cleared_houses) >= num_houses // 2:
                        break
                        
                    result = self.bid_on_houses(household, year)
                    if result:
                        cleared_households.add(h_id)
                        chosen_house = household.chosen_house
                        cleared_houses.add(chosen_house)

                        # Remove matched house from list
                        self.houses = self.houses[self.houses['Assessment Account Number'] != chosen_house]
                        sub_households.pop(h_id)

                        # Remove house from global availability
                        available_houses = available_houses[available_houses['Assessment Account Number'] != chosen_house]

                # Break if no progress was made in this iteration
                if safety_counter > 1 and len(cleared_households) == 0 and len(cleared_houses) == 0:
                    print("No market clearing progress, breaking cycle")
                    break

            print(f"Cleared {len(cleared_households)} households and {len(cleared_houses)} houses in this cycle.")

            # === Replenish Sub-market ===
            remaining_households = {k: v for k, v in all_households.items() if k not in cleared_households}
            remaining_houses = available_houses[~available_houses['Assessment Account Number'].isin(cleared_houses)]

            replenish_n_households = num_households - len(cleared_households)
            replenish_n_houses = num_houses - len(cleared_houses)

            if len(remaining_households) >= replenish_n_households:
                new_sub_households = dict(random.sample(list(remaining_households.items()), replenish_n_households))
            else:
                new_sub_households = remaining_households  # whatever remains

            if len(remaining_houses) >= replenish_n_houses:
                new_sub_houses = remaining_houses.sample(n=replenish_n_houses)
            else:
                new_sub_houses = remaining_houses

            sub_households.update(new_sub_households)
            self.houses = pd.concat([self.houses, new_sub_houses])
            available_houses = pd.concat([available_houses, new_sub_houses])

            # === Optional: Exit or refresh for households ===
            for household in all_households.values():
                if household.successful_deals == 0:
                    household.failed_attempts = getattr(household, 'failed_attempts', 0) + 1
                    if household.failed_attempts >= 3:
                        print(f"Household {household.unique_id} is exiting market after 3 failed attempts.")
                        continue  # this agent gives up
                    elif len(household.house_choices) < 2:
                        household.refresh_options()  # refresh bounds
                else:
                    household.failed_attempts = 0  # reset if succeeded

            # === Optional: Inject New Households (Migration) ===
            for i in range(5):
                new_id = max(all_households.keys()) + 1 + i
                sample_row = self.household_data.sample(1)
                # Fix: Pass the model instance instead of None
                new_household = Household(new_id, self.model, sample_row)
                all_households[new_id] = new_household

            cycle += 1

        # Final export of bidding results
        os.makedirs("./results-scenario/combined-shock/rl/", exist_ok=True)
        pd.DataFrame(self.bidding_results).to_csv(f"./results-scenario/combined-shock/rl/bidding_results_year_{simulation_year}.csv", index=False)
        # After final export
        df = pd.DataFrame(self.bidding_results)
        if not df.empty:
            self.model.last_year_median_price = df['final_price'].median()
        # reset bidding results for next year
        self.bidding_results.clear() 


class House:
    def __init__(self, data, current_year=0):
        self.id = data['Assessment Account Number']
        self.status = "vacant"
        self.years_occupied = 0
        # Use the current year's inflation rate
        year = 2010 + current_year
        inflation_rate =  INFLATION_RATE
        self.price = data['Assessed Value'] * (1 + inflation_rate)
    


class Household(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)  # Mesa Agent only needs model parameter
        self.unique_id = unique_id
        
        # Fix: Handle case where household_data might be a DataFrame row
        if isinstance(household_data, pd.DataFrame):
            if len(household_data) > 0:
                data_row = household_data.iloc[0]
            else:
                raise ValueError("Empty household_data DataFrame")
        elif isinstance(household_data, pd.Series):
            data_row = household_data
        else:
            data_row = household_data
            
        self.wtp = self.calculate_wtp(data_row)
        
        # Set the WTP value for this household in the model's household_data
        if hasattr(model, 'household_data') and unique_id in model.household_data.index:
            # Initialize WTP column in model's household_data if it doesn't exist
            if 'wtp' not in model.household_data.columns:
                model.household_data['wtp'] = 0
            model.household_data.loc[unique_id, 'wtp'] = self.wtp
            
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.chosen_house = None
        self.house_choices = {}
        self.successful_deals = 0
        self.deals = []
        
        # Add metrics tracking like DRL
        self.episode_rewards = []
        self.current_episode_reward = 0

    def refresh_options(self):
        """Refresh WTP bounds slightly if household has too few options."""
        self.lower_bound *= random.uniform(0.98, 1.02)
        self.upper_bound *= random.uniform(0.98, 1.02)


    def range_calculation(self, wtp):
        lower_bound = wtp * PERCENTAGE_BELOW_WTP
        upper_bound = wtp * PERCENTAGE_ABOVE_WTP
        return lower_bound, upper_bound

    def calculate_wtp(self, data):
        y_values = np.ones(10)
        
        # Fix: Use .get() method with default values for potentially missing columns
        self.wtp = (
            38.637 * data.get('birth of a child (x1)', 0) * y_values[0]
            - 12.309 * data.get('distance to CBD (x3) (km)', 0) * y_values[1]
            - 84.074 * data.get('distance to CBD (x3) (km)', 0) * y_values[2]
            - 66.961 * data.get('distance to nearest business center (x5) (km)', 0)
            + 79.441 * data.get('distance to nearest business center (x5) (km)', 0) * y_values[4]
            - 55.490 * data.get('distance to nearest school (x6) (km)', 0) * y_values[5]
            - 11.740 * data.get('distacne to nearest bus stop(km) (x7)', 0)
            - 49.657 * data.get('distance to nearest park area (x8) (km)', 0)
            + 0.064 * data.get('population density (x9)', 0)
            + 2.162 * data.get('% owned property (x10)', 0) * 1000
        )
        return float(self.wtp)

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-value for a state-action pair using the Bellman equation."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + GAMMA * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += ALPHA * td_error
        
        # Track episode reward
        self.current_episode_reward += reward

    def end_episode(self):
        """End episode and record metrics like RL"""
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0

    @staticmethod
    def save_eligibility_data(households, filename="./results-scenario/combined-shock/rl/household_eligibility.csv"):
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


class HousingMarket(mesa.Model):
    def __init__(self, years=10, freeze_year=2020, demand_shock_year=2020, demand_multiplier=2.0):
        super().__init__()
        self.housing_stock = []
        self.household_data = {}
        self.current_year = 0
        self.total_years = years
        self.freeze_year = freeze_year
        self.demand_shock_year = demand_shock_year
        self.demand_multiplier = demand_multiplier
        self.init_data()
        self.init_housing_stock()


    
    def step(self):
        self.agents.shuffle().do("step")

    
    def init_data(self,year=2015, houses_path='./inputs/parcel-yearwise/'):
        """Load initial household and housing data."""
        self.household_data = pd.read_excel('./inputs/HMTS.xlsx')
        
        # Determine which year's housing data to load
        current_simulation_year = year + self.current_year
        if self.freeze_year and current_simulation_year >= self.freeze_year:
            print(f"Supply shock in effect — using frozen housing stock from {self.freeze_year}")
            load_year = self.freeze_year  # Use frozen year's data
        else:
            load_year = current_simulation_year  # Use current year's data
            
        # Handle years beyond available data
        if load_year > 2025:
            print(f"Year {load_year} exceeds available data, using 2025 data with inflation adjustment")
            actual_load_year = 2025
        else:
            actual_load_year = load_year
            
        try:
            self.parcel_data = pd.read_csv(f'{houses_path}/parcel_data_{actual_load_year}.csv')
            # Apply inflation for years beyond available data
            if load_year > 2025:
                years_diff = load_year - 2025
                inflation_multiplier = (1 + INFLATION_RATE) ** years_diff
                self.parcel_data['Assessed Value'] *= inflation_multiplier
                print(f"Applied {years_diff} years of inflation ({inflation_multiplier:.3f}x) to housing values")
        except FileNotFoundError:
            print(f"Warning: No data file for year {actual_load_year}, using 2025 data")
            self.parcel_data = pd.read_csv(f'{houses_path}/parcel_data_2025.csv')
            if load_year > 2025:
                years_diff = load_year - 2025
                inflation_multiplier = (1 + INFLATION_RATE) ** years_diff
                self.parcel_data['Assessed Value'] *= inflation_multiplier


    def init_housing_stock(self):
        """Initialize the housing stock from parcel data."""
        # Always use current year for House initialization (prices will be adjusted)
        self.housing_stock = [House(row, self.current_year) for _, row in self.parcel_data.iterrows()]

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


    def run_year(self, households, game):
        """Run the housing market for one year."""
        print(f"\n{'='*50}")
        print(f"Running Year {self.current_year + 1}")
        print(f"{'='*50}")
        start_time = time.time()
        game.run(households, max_cycles=2, year=self.current_year)  # Reduced from 3 to 2 cycles
        # End episode for all households to track metrics
        for household in households.values():
            household.end_episode()
        end_time = time.time()
        print(f"Time taken for Year {self.current_year + 1}: {end_time - start_time:.2f} seconds")

    def evaluate_simulation(self, households):
        """Evaluate the performance of the households and market."""
        success_count = 0
        total_negotiations = 0
        total_prices = 0
        total_steps = 0
        total_rewards = 0

        for household in households.values():
            success_count += household.successful_deals
            total_negotiations += len(household.q_table)
            if household.successful_deals > 0:
                total_prices += getattr(household, 'chosen_house_price', 0) or 0
            total_steps += sum(len(household.q_table.get(state, [])) for state in household.q_table)
            total_rewards += sum(household.episode_rewards)

        print("\nSimulation Results:")
        print(f"Total Successful Deals: {success_count}")
        print(f"Average Negotiated Price: ${total_prices / success_count if success_count > 0 else 0:.2f}")
        print(f"Average Steps per Negotiation: {total_steps / total_negotiations if total_negotiations > 0 else 0:.2f}")
        print(f"Average Episode Reward: {total_rewards / len(households) if len(households) > 0 else 0:.2f}")

    def simulate(self):
        print(f"Initializing simulation for {self.total_years} years...")

        all_households = {
            household_id: Household(household_id, self, self.household_data.loc[household_id])
            for household_id in self.household_data.index
        }

        Household.save_eligibility_data(all_households)

        game = BiddingGame(self)

        for year in range(self.total_years):
            print(f"\n{'='*60}")
            print(f"SIMULATION YEAR {year + 1}/{self.total_years}")
            print(f"{'='*60}")
            self.current_year = year
            self.init_data()
            self.init_housing_stock()

            # Apply demand shock by increasing the number of active households
            base_ids = list(all_households.keys())
            is_shock_year = (2015 + year == self.demand_shock_year)
            sample_size = int(len(base_ids) * (self.demand_multiplier if is_shock_year else 1.0))
            sampled_ids = random.sample(base_ids, min(sample_size, len(base_ids)))

            print(f"{'>>>' if is_shock_year else '   '} Year {2024 + year}: Sampling {len(sampled_ids)} households (Demand Multiplier: {self.demand_multiplier if is_shock_year else 1.0})")

            households = {hid: all_households[hid] for hid in sampled_ids}

            self.run_year(households, game)
            self.update_market_conditions()

            for household in all_households.values():
                household.chosen_house = None
                household.chosen_house_price = None
                household.successful_deals = 0
                household.deals = []
                household.wtp *= (1 + np.random.uniform(0.01, 0.03))
                household.lower_bound = household.wtp * PERCENTAGE_BELOW_WTP
                household.upper_bound = household.wtp * PERCENTAGE_ABOVE_WTP

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        self.evaluate_simulation(all_households)


if __name__ == "__main__":
    print("Starting Housing Market Simulation with Demand Shock (Year: 2020)")
    model = HousingMarket(years=15, freeze_year=2020, demand_shock_year=2020, demand_multiplier=2.0)
    model.simulate()
