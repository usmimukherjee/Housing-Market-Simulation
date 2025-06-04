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

# Suppress warnings
warnings.filterwarnings("ignore")

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

# Define the DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

# Class to simulate the bidding process
class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []
        self.transactions = []
        self.competitive_bids = defaultdict(list)  # Track bids for each house
        
    def load_data(self, household_data_path="./results/household_eligibility.csv", houses_path='./inputs/Parcel.xlsx'):
        self.household_data = pd.read_csv(household_data_path)
        self.household_data.set_index("household_id", inplace=True)
        self.houses = pd.read_excel(houses_path)
    
    def reset_for_new_year(self):
        """Reset bidding results while preserving transactions"""
        self.bidding_results = []
        self.competitive_bids.clear()
    
    def negotiate_price(self, household, house_value):
        bid = household.lower_bound
        counter_offer = house_value
        negotiation_prices = []
        state = np.array([bid, counter_offer], dtype=np.float32)
        steps = 0
        total_reward = 0
        done = False

        while not done and steps < 100:  # Add max steps to prevent infinite loops
            # Epsilon-greedy action selection
            if isinstance(household, RLHousehold):
                if np.random.rand() < EPSILON:
                    action = np.random.choice(3)  # Exploration
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        q_values = household.dqn_model(state_tensor)
                        action = torch.argmax(q_values).item()
            else:
                # For regular households, use a simple strategy
                if bid < counter_offer:
                    action = np.random.choice([0, 1])  # Small or medium increment
                else:
                    action = 2  # Large increment

            increment = [0.02, 0.05, 0.1][action]
            new_bid = bid + bid * increment
            new_counter = counter_offer - counter_offer * np.random.uniform(0.02, 0.05)

            # Calculate reward with intermediate rewards
            reward = 0
            if new_bid > bid and new_counter < counter_offer:
                # Reward for mutual compromise
                reward += INTERMEDIATE_REWARD
            
            if new_bid >= new_counter:
                # Final success reward
                reward += FINAL_SUCCESS_REWARD - steps * NEGOTIATION_PENALTY_FACTOR
                done = True
            else:
                # Penalty for continued negotiation
                reward -= NEGOTIATION_PENALTY_FACTOR

            next_state = np.array([new_bid, new_counter], dtype=np.float32)
            
            # Store experience in replay memory and train only for RL households
            if isinstance(household, RLHousehold):
                household.memory.push(state, action, reward, next_state, done)
                total_reward += reward
                
                # Train the model
                if len(household.memory) >= BATCH_SIZE:
                    loss = household.train_model()
                    household.current_episode_loss += loss
            
            # Update state
            state = next_state
            bid = new_bid
            counter_offer = new_counter
            negotiation_prices.append((bid, counter_offer))
            steps += 1

        final_price = (bid + counter_offer) / 2 if done else None
        
        if done:
            if isinstance(household, RLHousehold):
                household.current_episode_reward += total_reward
            household.successful_deals += 1
            household.deals.append(final_price)
        
        return final_price, negotiation_prices

    def sample_houses(self, eligible_houses):
        if len(eligible_houses) <= NUM_HOUSE_SAMPLE:
            return eligible_houses
        prices = eligible_houses['PUBVALUE']
        quantiles = prices.quantile([0.33, 0.66])
        low = eligible_houses[prices <= quantiles[0.33]].sample(min(5, len(eligible_houses[prices <= quantiles[0.33]])))
        mid = eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])].sample(min(5, len(eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])])))
        high = eligible_houses[prices > quantiles[0.66]].sample(min(5, len(eligible_houses[prices > quantiles[0.66]])))
        return pd.concat([low, mid, high])

    def rl_bidding(self, household, houses_df):
        eligible = houses_df[
            (houses_df['PUBVALUE'] > household.lower_bound) &
            (houses_df['PUBVALUE'] <= household.upper_bound)
        ]
        if eligible.empty:
            return {}
            
        sampled = self.sample_houses(eligible)
        results = {}
        for _, house in sampled.iterrows():
            final_price, negotiation = self.negotiate_price(household, house['PUBVALUE'])
            if final_price is not None:
                # Store the bid in competitive bids
                self.competitive_bids[house['OBJECTID']].append({
                    'household_id': household.unique_id,
                    'final_price': final_price,
                    'negotiation': negotiation
                })
                results[house['OBJECTID']] = final_price
                self.bidding_results.append({
                    'household_id': household.unique_id,
                    'house_id': house['OBJECTID'],
                    'initial_bid': household.lower_bound,
                    'final_price': final_price,
                    'counter_offer': house['PUBVALUE'],
                    'negotiation_steps': len(negotiation),
                    'negotiation_prices': str(negotiation)
                })
        return results

    def run(self, households, model):
        """Run bidding process with market clearing"""
        # Get current houses for sale from the market model
        houses_df = model.get_houses_for_sale()
        print(f"\nNumber of houses available for sale: {len(houses_df)}")
        
        # Reset bidding results for the new year
        self.reset_for_new_year()
        
        # Track transactions
        transactions = []
        
        # First round: Collect all bids
        print("\nCollecting bids from all households...")
        for household_type, household_dict in households.items():
            print(f"\nProcessing {household_type} households...")
            household_list = list(household_dict.values())
            random.shuffle(household_list)
            
            for household in household_list:
                if hasattr(household, 'chosen_house') and household.chosen_house is not None:
                    continue
                    
                print(f"\nStarting bids for Household ID: {household.unique_id}")
                print(f"WTP: ${household.wtp:.2f}")
                print(f"Price range: ${household.lower_bound:.2f} - ${household.upper_bound:.2f}")
                
                self.rl_bidding(household, houses_df)
        
        # Second round: Process competitive bids
        print("\nProcessing competitive bids...")
        for house_id, bids in self.competitive_bids.items():
            if not bids:
                continue
                
            # Sort bids by final price (highest first)
            sorted_bids = sorted(bids, key=lambda x: x['final_price'], reverse=True)
            winning_bid = sorted_bids[0]
            
            # Record transaction
            transactions.append({
                'household_id': winning_bid['household_id'],
                'house_id': house_id,
                'final_price': winning_bid['final_price'],
                'year': model.current_year + 1,
                'competing_bids': len(bids)
            })
            
            # Update household status
            for household_type in households.values():
                if winning_bid['household_id'] in household_type:
                    household = household_type[winning_bid['household_id']]
                    household.chosen_house = house_id
                    household.chosen_house_price = winning_bid['final_price']
                    print(f"Household {household.unique_id} won house {house_id} at ${winning_bid['final_price']:.2f} with {len(bids)} competing bids")
                    break
            
            # Update house status in the model
            for house in model.housing_stock:
                if house.id == house_id:
                    house.status = "occupied"
                    house.years_occupied = 0
                    print(f"Updated house {house_id} status to occupied")
                    break
        
        # Save bidding results
        if self.bidding_results:
            bidding_df = pd.DataFrame(self.bidding_results)
            os.makedirs("./results/model-deep-rl/variant__1", exist_ok=True)
            bidding_df.to_csv(f"./results/model-deep-rl/variant__1/bidding_results_year_{model.current_year + 1}.csv", index=False)
        
        # Save transaction log
        if transactions:
            pd.DataFrame(transactions).to_csv("./results/model-deep-rl/variant__1/transactions.csv", index=False)
        
        return len([t for t in transactions if t['year'] == model.current_year + 1])

class House:
    def __init__(self, data):
        self.id = data['OBJECTID']
        self.status = "vacant"  # "vacant", "for sale", "occupied"
        self.price = data['PUBVALUE']
        self.original_price = data['PUBVALUE']  # Store original price
        self.years_occupied = 0

class RLHousehold(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)
        self.unique_id = unique_id
        self.wtp = self.calculate_wtp(household_data.loc[unique_id])
        self.lower_bound = self.wtp * PERCENTAGE_BELOW_WTP
        self.upper_bound = self.wtp * PERCENTAGE_ABOVE_WTP
        self.household_data = household_data
        self.successful_deals = 0
        self.chosen_house = None
        self.chosen_house_price = None
        self.house_choices = {}
        self.deals = []
        
        # DQN components
        self.dqn_model = DQN(2, 3)  # State: [bid, counter_offer], Actions: 3 options
        self.target_model = DQN(2, 3)
        self.target_model.load_state_dict(self.dqn_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=ALPHA)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        
        # Metrics
        self.episode_rewards = []       # Total reward per episode (year)
        self.episode_losses = []        # Total loss per episode (year)
        self.current_episode_reward = 0
        self.current_episode_loss = 0
        self.update_target_counter = 0

    def calculate_wtp(self, data):
        y_values = np.ones(10)
        wtp = (
            38.637 * data['birth of a child (x1)'] * y_values[0]
            - 12.309 * data.get('distance to CBD (x3) (km)', 0) * y_values[1]
            - 84.074 * data['distance to CBD (x3) (km)'] * y_values[2]
            - 66.961 * data.get('distance to nearest business center (x5) (km)', 0)
            + 79.441 * data['distance to nearest business center (x5) (km)'] * y_values[4]
            - 55.490 * data['distance to nearest school (x6) (km)'] * y_values[5]
            - 11.740 * data['distacne to nearest bus stop(km) (x7)']
            - 49.657 * data['distance to nearest park area (x8) (km)']
            + 0.064 * data['population density (x9)']
            + 2.162 * data['% owned property (x10)'] * 1000
        )
        return float(wtp)

    def train_model(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        # Sample batch from replay memory
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Compute current Q values
        current_q = self.dqn_model(states).gather(1, actions)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            next_q = next_q * (1 - dones)  # Zero out Q values for terminal states
        
        # Compute target Q values
        target_q = rewards + GAMMA * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_target_counter += 1
        if self.update_target_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.dqn_model.state_dict())
        
        return loss.item()

    def end_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_losses.append(self.current_episode_loss)
        self.current_episode_reward = 0
        self.current_episode_loss = 0

    def plot_metrics(self, save_path=None):
        plt.figure(figsize=(15, 6))
        
        # Reward plot
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, 'b-', label='Total Reward')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_losses, 'r-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()

class Household(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)
        self.unique_id = unique_id
        self.wtp = self.calculate_wtp(household_data.loc[unique_id])
        household_data.loc[unique_id, 'wtp'] = self.wtp
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        self.successful_deals = 0
        self.chosen_house = None
        self.chosen_house_price = None
        self.house_choices = {}
        self.deals = []

    def range_calculation(self, wtp):
        lower_bound = wtp * PERCENTAGE_BELOW_WTP
        upper_bound = wtp * PERCENTAGE_ABOVE_WTP
        return lower_bound, upper_bound
    
    def calculate_wtp(self, data):
        y_values = np.ones(10)
        wtp = (
            38.637 * data['birth of a child (x1)'] * y_values[0]
            - 12.309 * data.get('distance to CBD (x3) (km)', 0) * y_values[1]
            - 84.074 * data['distance to CBD (x3) (km)'] * y_values[2]
            - 66.961 * data.get('distance to nearest business center (x5) (km)', 0)
            + 79.441 * data['distance to nearest business center (x5) (km)'] * y_values[4]
            - 55.490 * data['distance to nearest school (x6) (km)'] * y_values[5]
            - 11.740 * data['distacne to nearest bus stop(km) (x7)']
            - 49.657 * data['distance to nearest park area (x8) (km)']
            + 0.064 * data['population density (x9)']
            + 2.162 * data['% owned property (x10)'] * 1000
        )
        return float(wtp)

    @staticmethod
    def save_eligibility_data(households, filename="./results/household_eligibility.csv"):
        df = pd.DataFrame.from_dict(
            {
                h.unique_id: {"wtp": h.wtp, "lower_bound": h.lower_bound, "upper_bound": h.upper_bound}
                for h in households.values()
            },
            orient="index"
        )
        df.index.name = "household_id"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename)

class HousingMarket(mesa.Model):
    def __init__(self, years=2):  # Changed default to 2 years
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
            save_path = f'./output-analysis/model-deep-rl/variant__1/metrics_episode_{self.episode}.png'
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
        os.makedirs("./results/model-deep-rl", exist_ok=True)
        os.makedirs("./output-analysis/model-deep-rl/variant__1", exist_ok=True)
        os.makedirs("./results/market_state", exist_ok=True)
        
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
        pd.DataFrame(market_stats).to_csv("./results/model-deep-rl/variant__1/market_state/market_stats.csv", index=False)
        
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
        os.makedirs("./results/model-deep-rl/variant__1/market_state", exist_ok=True)
        pd.DataFrame(market_data).to_csv(f"./results/model-deep-rl/variant__1/market_state/year_{year}.csv", index=False)
        
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
            f"./results/model-deep-rl/variant__1/household_eligibility_year_{year+1}.csv"
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
            f"./results/model-deep-rl/rl_metrics_year_{year+1}.csv", 
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
        plt.savefig("./output-analysis/market_dynamics.png")
        plt.close()

# Run the simulation
if __name__ == "__main__":
    print("Starting Housing Market Simulation with Deep RL")
    start_time = time.time()
    
    model = HousingMarket(years=5)
    model.simulate()
    
    duration = time.time() - start_time
    print(f"\nSimulation completed in {duration:.2f} seconds")