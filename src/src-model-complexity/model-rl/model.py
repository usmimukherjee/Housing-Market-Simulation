import pandas as pd
import random
import numpy as np
import time
import mesa
import warnings
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

INFLATION_RATE_BY_YEAR = {
    2010:0.0178, 2011:0.291, 2012:0.0152, 2013:0.0094,2014:0.0194,2015:0.013,
    2016: 0.013, 2017: 0.026, 2018:0.016, 2019:0.037,2020:0.006, 2021:0.041,2022:0.074, 2023:0.043, 2024:0.044,
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

    def load_data(self, year, household_data_path="./results-complexity/rl/household_eligibility.csv", houses_path='./inputs/parcel-yearwise/'):
        self.household_data = pd.read_csv(household_data_path)
        self.houses = pd.read_csv(f'{houses_path}/parcel_data_{year}.csv')
        

    def negotiate_price(self, household, house_value, current_year=0):
        """Reinforcement learning-based negotiation with RL reward system."""
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

    def run(self, households, max_cycles=3, year=0):
        """Market clearing process with sub-market cycles based on Farooq and Miller (2012)."""
        simulation_year = 2010 + year
        self.load_data(year=simulation_year)
        
        SUBMARKET_FRACTION = 0.45
        num_households = int(len(households) * SUBMARKET_FRACTION)
        num_houses = int(len(self.houses) * SUBMARKET_FRACTION)

        all_households = households.copy()
        available_houses = self.houses.copy()
        cycle = 0
        
        # Cycle count (max_cycles) controls how long the market clearing lasts per year
        while cycle < max_cycles:
            print(f"\n--- Sub-Market Cycle {cycle + 1} ---")
            # Randomly sample sub-market participants
            sub_households = dict(random.sample(list(all_households.items()), num_households))
            sub_houses = available_houses.sample(n=num_houses).copy()

            cleared_households = set()
            cleared_houses = set()

            # Replace house pool for this round
            self.houses = sub_houses

            while (len(cleared_households) < num_households // 2) and (len(cleared_houses) < num_houses // 2):
                for h_id, household in list(sub_households.items()):
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

                    # Stop if sub-market shrunk enough
                    if len(cleared_households) >= num_households // 2 or len(cleared_houses) >= num_houses // 2:
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
        pd.DataFrame(self.bidding_results).to_csv(f"./results-complexity/rl/bidding_results_year_{simulation_year}.csv", index=False)
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
        # Use the current year's inflation rate
        year = 2010 + current_year
        inflation_rate = INFLATION_RATE_BY_YEAR.get(year, INFLATION_RATE)
        self.price = data['Assessed Value'] * (1 + inflation_rate) 
    


class Household(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)  # Mesa Agent only needs model parameter
        self.unique_id = unique_id
        
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
        
        # Add metrics tracking like RL
        self.episode_rewards = []
        self.current_episode_reward = 0
        # --- Additions for tracking ---
        self.action_counts = np.zeros(3, dtype=int)  # Track action selection per episode
        self.q_variances = []  # Track Q-value variance per episode
        self.yearly_stats = []  # Store stats for CSV output

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
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.q_table[state])
        self.action_counts[action] += 1  # Track action selection
        return action
        
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
        # --- Add Q-value variance tracking ---
        all_qs = np.concatenate(list(self.q_table.values())) if self.q_table else np.array([0,0,0])
        q_var = np.var(all_qs)
        self.q_variances.append(q_var)
        self.current_episode_reward = 0
        # Optionally reset action counts if you want per-episode stats
        # self.action_counts = np.zeros(3, dtype=int)

    @staticmethod
    def save_eligibility_data(households, filename="./results-complexity/rl/household_eligibility.csv"):
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
    def __init__(self, years=10):
        super().__init__()
        self.housing_stock = []
        self.household_data = {}
        self.current_year = 0
        self.total_years = years
        self.init_data()
        self.init_housing_stock()
    
    def step(self):
        self.agents.shuffle().do("step")

    
    def init_data(self,year=2016,houses_path='./inputs/parcel-yearwise/'):
        """Load initial household and housing data."""
        self.household_data = pd.read_excel('./inputs/HMTS.xlsx')
        self.parcel_data = pd.read_csv(f'{houses_path}/parcel_data_{year+self.current_year}.csv')

    def init_housing_stock(self):
        """Initialize the housing stock from parcel data."""
        self.housing_stock = [House(row, self.current_year) for _, row in self.parcel_data.iterrows()]

    def update_market_conditions(self):
        """Simulate market changes for each year."""
        year = 2010 + self.current_year
        current_inflation = INFLATION_RATE_BY_YEAR.get(year, INFLATION_RATE)
        
        # Increase house prices with variable inflation
        for house in self.housing_stock:
            house.price *= (1 + current_inflation)
        
        # Adjust household WTP with market conditions
        market_multiplier = 1.0 - (self.current_year * 0.02)  # Declining market
        self.household_data['wtp'] *= (1 + PURCHASING_POWER_INCREASE * market_multiplier)


    def run_year(self, households, game):
        """Run the housing market for one year."""
        print(f"\n{'='*50}")
        print(f"Running Year {self.current_year + 1}")
        print(f"{'='*50}")
        start_time = time.time()
        game.run(households, max_cycles=3,year=self.current_year)
        # End episode for all households to track metrics
        for household in households.values():
            household.end_episode()
        end_time = time.time()
        print(f"Time taken for Year {self.current_year + 1}: {end_time - start_time:.2f} seconds")
        # --- Save stats for this year ---
        self.save_yearly_stats(households, self.current_year)

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
        """Simulate the housing market over multiple years."""
        households = {
            household_id: Household(household_id, self, self.household_data.loc[household_id])
            for household_id in self.household_data.index
        }

        Household.save_eligibility_data(households)

        game = BiddingGame(self)

        for year in range(self.total_years):
            self.current_year = year
            self.run_year(households, game)
            self.update_market_conditions()
            
            for household in households.values():
                household.chosen_house = None
                household.chosen_house_price = None
                household.successful_deals = 0
                household.deals = []
                household.wtp *= (1 + np.random.uniform(0.01, 0.03))
                household.lower_bound = household.wtp * PERCENTAGE_BELOW_WTP
                household.upper_bound = household.wtp * PERCENTAGE_ABOVE_WTP

        self.evaluate_simulation(households)

    def save_yearly_stats(self, households, year):
        """Save Q-value variance, action selection, and cumulative rewards for each household for this year to a separate CSV file."""
        stats = []
        for h in households.values():
            # Use last episode's stats for this year
            q_var = h.q_variances[-1] if h.q_variances else 0
            total_reward = h.episode_rewards[-1] if h.episode_rewards else 0
            action_counts = h.action_counts.tolist()
            stats.append({
                'year': 2010 + year,
                'household_id': h.unique_id,
                'q_value_variance': q_var,
                'cumulative_reward': total_reward,
                'action_0_count': action_counts[0],
                'action_1_count': action_counts[1],
                'action_2_count': action_counts[2],
            })
            # Reset action counts for next year
            h.action_counts = np.zeros(3, dtype=int)
        df = pd.DataFrame(stats)
        out_path = f"./results-complexity/rl/yearly_rl_stats_{2010 + year}.csv"
        df.to_csv(out_path, index=False)

# Run the simulation
if __name__ == "__main__":
    print("Starting Housing Market Simulation with Modified RL Reward System")
    model = HousingMarket(years=15)
    model.simulate()