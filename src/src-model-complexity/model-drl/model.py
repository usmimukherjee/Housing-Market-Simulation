import pandas as pd  
import numpy as np 
import mesa 
import os 
import csv 
import random  
import warnings 
import time 
from collections import defaultdict  
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
PERCENTAGE_HOUSES_FOR_SALE = 0.08  
HOUSING_TURNOVER_RATE = 0.05  
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
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Simple feedforward neural network
class DQNSeller(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNSeller, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class RLSellerAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99, memory_size=10000, batch_size=32):
        self.dqn_model = DQNSeller(input_dim=2, output_dim=4)  # 4 concession actions
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.dqn_model(state_tensor)
        return torch.argmax(q_values).item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_tensor = torch.tensor(state_batch, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state_batch, dtype=torch.float32)
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32)
        action_tensor = torch.tensor(action_batch, dtype=torch.int64)
        done_tensor = torch.tensor(done_batch, dtype=torch.float32)

        q_values = self.dqn_model(state_tensor)
        current_q = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.dqn_model(next_state_tensor)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward_tensor + self.gamma * max_next_q * (1 - done_tensor)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# Class to simulate the bidding process
class BiddingGame:
    def __init__(self):
        self.house_list = {}
        self.bidding_results = []
        self.transactions = []
        # Track bids for each house
        self.competitive_bids = defaultdict(list) 
    
    def reset_for_new_year(self):
        """Reset bidding results while preserving transactions"""
        self.bidding_results = []
        self.competitive_bids.clear()
    
    def negotiate_price(self, household, house_value, current_year=0):
        bid = household.lower_bound
        counter_offer = house_value
        negotiation_prices = []
        state = np.array([bid / house_value, counter_offer / house_value], dtype=np.float32)
        steps = 0
        total_reward = 0
        done = False

        while not done and steps < 25: 
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
                if bid < counter_offer:
                    action = np.random.choice([0, 1])  # Small or medium increment
                else:
                    action = 2  # Large increment

            # More aggressive bid increments
            increment = [0.12, 0.22, 0.40][action]
            new_bid = bid + bid * increment

            # Seller flexibility logic
            concession_base = np.random.uniform(0.01, 0.03)
            gap_ratio = (counter_offer - bid) / counter_offer
            gap_sensitivity = np.exp(-5 * gap_ratio)  # Sharp drop as gap increases
            concession_rate = np.clip(concession_base * gap_sensitivity, 0.001, 0.03)

            # gap_ratio = (counter_offer - bid) / counter_offer
            # concession_rate = np.clip(np.random.uniform(0.005, 0.02) * (1 - gap_ratio), 0.002, 0.03)
            new_counter = counter_offer - counter_offer * concession_rate
            new_counter = max(new_counter, house_value * 0.95)


            # Intermediate reward for movement
            reward = 0
            if new_bid > bid and new_counter < counter_offer:
                reward += INTERMEDIATE_REWARD

            # Price realism reward
            estimated_price = (new_bid + new_counter) / 2
            price_ratio = estimated_price / house_value

            if 0.9 <= price_ratio <= 1.1:
                reward += 0.4
            elif price_ratio > 1.1:
                reward -= 0.2
            elif price_ratio < 0.7:
                reward -= 0.3

            if new_bid >= new_counter:
                reward += FINAL_SUCCESS_REWARD - steps * NEGOTIATION_PENALTY_FACTOR
                done = True
                if estimated_price <= house_value * 0.6:
                    reward -= 1.0
                elif estimated_price <= house_value * 0.8:
                    reward -= 0.5

            else:
                reward -= NEGOTIATION_PENALTY_FACTOR

            next_state = np.array([new_bid / house_value, new_counter / house_value], dtype=np.float32)

            if isinstance(household, RLHousehold):
                household.memory.push(state, action, reward, next_state, done)
                total_reward += reward
                if len(household.memory) >= BATCH_SIZE:
                    loss = household.train_model()
                    household.current_episode_loss += loss

            state = next_state
            bid = new_bid
            counter_offer = new_counter
            negotiation_prices.append((bid, counter_offer))
            steps += 1

        final_price = (bid + counter_offer) / 2 if done else None
        if final_price:
            floor = house_value * 0.9
            final_price = max(final_price, floor)


        if done:
            if isinstance(household, RLHousehold):
                household.current_episode_reward += total_reward
            household.successful_deals += 1
            household.deals.append(final_price)

        return final_price, negotiation_prices


    def sample_houses(self, eligible_houses):
        if len(eligible_houses) <= NUM_HOUSE_SAMPLE:
            return eligible_houses
        prices = eligible_houses['Assessed Value']
        quantiles = prices.quantile([0.33, 0.66])
        low = eligible_houses[prices <= quantiles[0.33]].sample(min(5, len(eligible_houses[prices <= quantiles[0.33]])))
        mid = eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])].sample(min(5, len(eligible_houses[(prices > quantiles[0.33]) & (prices <= quantiles[0.66])])))
        high = eligible_houses[prices > quantiles[0.66]].sample(min(5, len(eligible_houses[prices > quantiles[0.66]])))
        return pd.concat([low, mid, high])

    def run(self, households, model):
        """Run bidding process with market clearing using sub-market cycles"""
        # Get current houses for sale from the market model
        houses_df = model.get_houses_for_sale()
        print(f"\nNumber of houses available for sale: {len(houses_df)}")
        
        # Reset bidding results for the new year
        self.reset_for_new_year()
        
        # Market clearing parameters
        SUBMARKET_FRACTION = 0.3  # Submarket fraction (0.3) can be tuned based on demand/supply pressure
        MAX_CYCLES = 3  # Maximum number of sub-market cycles
        
        # Flatten all households into a single dictionary for easier processing
        all_households = {}
        for household_type, household_dict in households.items():
            all_households.update(household_dict)
        
        # Calculate sub-market sizes
        num_households = max(1, int(len(all_households) * SUBMARKET_FRACTION))
        num_houses = max(1, int(len(houses_df) * SUBMARKET_FRACTION))
        
        # Track all transactions across cycles
        all_transactions = []
        available_houses = houses_df.copy()
        remaining_households = all_households.copy()
        cycle = 0
        
        print(f"\nStarting market clearing with {MAX_CYCLES} cycles")
        print(f"Sub-market size: {num_households} households, {num_houses} houses")
        
        # Market clearing cycles
        while cycle < MAX_CYCLES and len(remaining_households) > 0 and len(available_houses) > 0:
            print(f"\n--- Sub-Market Cycle {cycle + 1} ---")
            
            # Sample sub-market participants
            sample_households = dict(random.sample(
                list(remaining_households.items()), 
                min(num_households, len(remaining_households))
            ))
            
            sample_houses = available_houses.sample(
                n=min(num_houses, len(available_houses))
            ).copy()
            
            print(f"Cycle {cycle + 1}: {len(sample_households)} households, {len(sample_houses)} houses")
            
            # Reset competitive bids for this cycle
            cycle_competitive_bids = {}
            cycle_bidding_results = []
            
            # First round: Collect all bids for this sub-market
            print(f"\nCollecting bids in cycle {cycle + 1}...")
            for household_id, household in sample_households.items():
                if hasattr(household, 'chosen_house') and household.chosen_house is not None:
                    continue
                    
                print(f"Processing Household ID: {household.unique_id}")
                print(f"WTP: ${household.wtp:.2f}")
                print(f"Price range: ${household.lower_bound:.2f} - ${household.upper_bound:.2f}")
                
                # Run RL bidding for this household on the sub-market houses
                self.rl_bidding_cycle(household, sample_houses, cycle_competitive_bids, cycle_bidding_results)
            
            # Second round: Process competitive bids for this cycle
            print(f"\nProcessing competitive bids in cycle {cycle + 1}...")
            cycle_transactions = []
            cleared_households = set()
            cleared_houses = set()
            
            for house_id, bids in cycle_competitive_bids.items():
                if not bids:
                    continue
                    
                # Sort bids by final price (highest first)
                sorted_bids = sorted(bids, key=lambda x: x['final_price'], reverse=True)
                winning_bid = sorted_bids[0]
                
                # Record transaction
                cycle_transactions.append({
                    'household_id': winning_bid['household_id'],
                    'house_id': house_id,
                    'final_price': winning_bid['final_price'],
                    'year': model.current_year,
                    'competing_bids': len(bids),
                    'cycle': cycle + 1
                })
                
                # Update household status
                household = remaining_households.get(winning_bid['household_id'])
                if household:
                    household.chosen_house = house_id
                    household.chosen_house_price = winning_bid['final_price']
                    print(f"Household {household.unique_id} won house {house_id} at ${winning_bid['final_price']:.2f} with {len(bids)} competing bids")
                    cleared_households.add(winning_bid['household_id'])
                    cleared_houses.add(house_id)
                
                # Update house status in the model
                for house in model.housing_stock:
                    if house.id == house_id:
                        house.status = "occupied"
                        house.years_occupied = 0
                        print(f"Updated house {house_id} status to occupied")
                        break
            
            # Add cycle results to overall results
            all_transactions.extend(cycle_transactions)
            self.bidding_results.extend(cycle_bidding_results)
            
            print(f"Cycle {cycle + 1} completed: {len(cleared_households)} households matched, {len(cleared_houses)} houses sold")
            
            # Remove cleared households and houses from remaining pools
            for household_id in cleared_households:
                remaining_households.pop(household_id, None)
            
            available_houses = available_houses[~available_houses.index.isin(cleared_houses)]
            
            # Exit early if sub-market cleared enough participants
            if len(cleared_households) >= num_households // 2 or len(cleared_houses) >= num_houses // 2:
                print(f"Sub-market cycle {cycle + 1} cleared sufficient participants, continuing...")
            
            # === Optional: Handle failed households ===
            for household_id, household in list(remaining_households.items()):
                if not hasattr(household, 'chosen_house') or household.chosen_house is None:
                    household.failed_attempts = getattr(household, 'failed_attempts', 0) + 1
                    if household.failed_attempts >= 3:
                        print(f"Household {household.unique_id} exiting market after 3 failed attempts")
                        remaining_households.pop(household_id, None)
                        continue
                    elif len(getattr(household, 'house_choices', {})) < 2:
                        # Refresh household options
                        household.lower_bound *= random.uniform(0.98, 1.02)
                        household.upper_bound *= random.uniform(0.98, 1.02)
                        print(f"Household {household.unique_id} refreshed price bounds")
                else:
                    household.failed_attempts = 0  # Reset if successful
            
            # === Optional: Inject new households (migration) ===
            if len(remaining_households) < num_households and cycle < MAX_CYCLES - 1:
                print(f"Injecting new households for cycle {cycle + 2}...")
                # Add some new households to simulate market entry
                for i in range(min(5, num_households - len(remaining_households))):
                    # Create new household based on existing household data
                    if hasattr(model, 'household_data') and len(model.household_data) > 0:
                        new_id = max(all_households.keys()) + 1 + i
                        sample_row = model.household_data.sample(1).iloc[0]
                        
                        # Create new household (assuming you have the appropriate constructor)
                        new_household = self.create_new_household(new_id, model, sample_row)
                        remaining_households[new_id] = new_household
                        all_households[new_id] = new_household
                        print(f"Added new household {new_id} to market")
            
            cycle += 1
        
        print(f"\nMarket clearing completed after {cycle} cycles")
        print(f"Total transactions: {len(all_transactions)}")
        print(f"Remaining households: {len(remaining_households)}")
        print(f"Remaining houses: {len(available_houses)}")
        
        # Save bidding results
        if self.bidding_results:
            bidding_df = pd.DataFrame(self.bidding_results)
            os.makedirs("./results-convergence/model-drl/", exist_ok=True)
            bidding_df.to_csv(f"./results-convergence/model-drl/bidding-results2/bidding_results_year_{model.current_year + 2010}.csv", index=False)
        
        # Save transaction log
        if all_transactions:
            transactions_df = pd.DataFrame(all_transactions)
            transactions_df.to_csv(f"./results-convergence/model-drl/transactions/transactions_year_{model.current_year + 2010}.csv", index=False)
            
            # Also save cumulative transactions
            cumulative_file = "./results-convergence/model-drl/transactions/transactions_cumulative.csv"
            if os.path.exists(cumulative_file):
                existing_df = pd.read_csv(cumulative_file)
                combined_df = pd.concat([existing_df, transactions_df], ignore_index=True)
            else:
                combined_df = transactions_df
            combined_df.to_csv(cumulative_file, index=False)
        
        return len([t for t in all_transactions if t['year'] == model.current_year + 1])

    def rl_bidding_cycle(self, household, houses_df, competitive_bids, bidding_results):
        """Modified RL bidding for sub-market cycles"""
        # Filter houses based on household's price range
        eligible_houses = houses_df[
            (houses_df['Assessed Value'] >= household.lower_bound) & 
            (houses_df['Assessed Value'] <= household.upper_bound)
        ]
        
        if eligible_houses.empty:
            print(f"No eligible houses for household {household.unique_id}")
            return
        
        print(f"Found {len(eligible_houses)} eligible houses for household {household.unique_id}")
        
        # Bid on eligible houses
        for _, house in eligible_houses.iterrows():
            house_id = house['Assessment Account Number']
            house_value = house['Assessed Value']
            
            # Run DRL negotiation
            final_price, negotiation_trace = self.negotiate_price(household, house_value)
            
            if final_price is not None:
                # Store bid details
                bid_info = {
                    'household_id': household.unique_id,
                    'house_id': house_id,
                    'final_price': final_price,
                    'negotiation_trace': negotiation_trace
                }
                
                # Add to competitive bids
                if house_id not in competitive_bids:
                    competitive_bids[house_id] = []
                competitive_bids[house_id].append(bid_info)
                
                # Add to bidding results
                bidding_results.append({
                    'year': getattr(self, 'current_year', 0) + 1,
                    'household_id': household.unique_id,
                    'house_id': house_id,
                    'initial_bid': household.lower_bound,
                    'final_price': final_price,
                    'counter_offer': house_value,
                    'negotiation_steps': len(negotiation_trace),
                    'negotiation_trace': str(negotiation_trace)
                })
                
                print(f"Household {household.unique_id} bid ${final_price:.2f} on house {house_id}")

    def create_new_household(self, household_id, model, sample_data):
        """Create a new household for market injection"""
        try:
            new_household = Household(household_id, model, sample_data)
            return new_household
        except Exception as e:
            print(f"Error creating new household: {e}")
            return None
    

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


class House:
    def __init__(self, data):
        self.id = data['Assessment Account Number']
        self.status = "vacant"  # "vacant", "for sale", "occupied"
        self.price = data['Assessed Value']
        self.original_price = data['Assessed Value']  # Store original price
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
        self.wtp = self.calculate_wtp(household_data)
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        self.successful_deals = 0
        self.chosen_house = None
        self.chosen_house_price = None
        self.house_choices = {}
        self.deals = []
        self.episode_rewards = []
        self.episode_losses = []
        self.failed_attempts = 0

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

    def end_episode(self):
        """End episode for RL tracking"""
        # For regular households, just track basic metrics
        if hasattr(self, 'episode_rewards'):
            self.episode_rewards.append(0)  # No RL rewards for regular households
        if hasattr(self, 'episode_losses'):
            self.episode_losses.append(0)  # No RL losses for regular households

    def plot_metrics(self, save_path=None):
        """Plot metrics for the household"""
        if not self.episode_rewards and not self.episode_losses:
            print("No metrics to plot for this household")
            return
            
        plt.figure(figsize=(12, 8))
        
        if self.episode_rewards:
            plt.subplot(2, 2, 1)
            plt.plot(self.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
        
        if self.episode_losses:
            plt.subplot(2, 2, 2)
            plt.plot(self.episode_losses)
            plt.title('Episode Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.bar(['Successful Deals'], [self.successful_deals])
        plt.title('Successful Deals')
        plt.ylabel('Count')
        
        plt.subplot(2, 2, 4)
        plt.bar(['WTP', 'Lower Bound', 'Upper Bound'], 
                [self.wtp, self.lower_bound, self.upper_bound])
        plt.title('Price Bounds')
        plt.ylabel('Price ($)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def save_eligibility_data(households, filename="./results-convergence/model-drl/household_eligibility.csv"):
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
    def __init__(self, years): 
        super().__init__()
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

    def step(self):
        self.agents.shuffle().do("step")

    def init_data(self, year=2010, household_data_path="./inputs/HMTS.xlsx", houses_path='./inputs/parcel-yearwise/'):
        self.household_data = pd.read_excel(household_data_path)
        self.houses = pd.read_csv(f'{houses_path}/parcel_data_{year}.csv')

    def init_housing_stock(self):
        self.housing_stock = [House(row) for _, row in self.houses.iterrows()]

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
        start_time = time.time()
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
        for household_type, household_dict in households.items():
            for household in household_dict.values():
                household.end_episode()
        # Plot metrics for the first household
        # if self.episode % 1 == 0 and households:
        #     first_household = next(iter(next(iter(households.values())).values()))
        #     save_path = f'./output-analysis/model-drl/metrics_episode_{self.episode}.png'
        #     first_household.plot_metrics(save_path=save_path)
        #     print(f"Saved metrics visualization for episode {self.episode}")
        # Update year stats
        year_stats.update({
            'transactions_completed': transactions,
            'avg_house_price': np.mean([h.price for h in self.housing_stock if h.price > 0]),
            'median_house_price': np.median([h.price for h in self.housing_stock if h.price > 0]),
            'successful_households': sum(
                1
                for household_dict in households.values()
                for h in household_dict.values()
                if h.chosen_house is not None
            )
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
        end_time = time.time()
        print(f"Time taken for Year {self.current_year}: {end_time - start_time:.2f} seconds")
        return year_stats

    def simulate(self):
        # Create directories for outputs
        os.makedirs("./results-convergence/model-drl", exist_ok=True)
        os.makedirs("./results-convergence/model-drl/market_state", exist_ok=True)
        
        # Create households in the structure expected by BiddingGame.run
        households = {'regular': {}}
        for household_id in self.household_data.index:
            row = self.household_data.loc[household_id]
            h = Household(household_id, self, row)
            households['regular'][household_id] = h
            self.household_data.loc[household_id, 'wtp'] = h.wtp
        
        Household.save_eligibility_data(households['regular'])
        # Track market statistics over time
        market_stats = []
        
        while self.current_year < self.total_years:
            year_stats = self.run_year(households)
            market_stats.append(year_stats)
            
            # Save intermediate results
            self.save_simulation_state(self.current_year - 1, households['regular'])
        
        # Save final market statistics
        pd.DataFrame(market_stats).to_csv("./results-convergence/model-drl/market_state/market_stats.csv", index=False)
        
        # Plot market dynamics
        # self.plot_market_dynamics(market_stats)
        
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
                    'Assessment Account Number': house.id,
                    'Assessed Value': house.price,
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
        os.makedirs("./results/model-drl/market_state", exist_ok=True)
        pd.DataFrame(market_data).to_csv(f"./results/model-drl/market_state/year_{year}.csv", index=False)
        
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
        # Save RL agent metrics
        rl_metrics = []
        for h in households.values():
            rl_metrics.append({
                'household_id': h.unique_id,
                'episode': year + 1,
                'total_reward': h.episode_rewards[-1] if h.episode_rewards else 0,
                'total_loss': h.episode_losses[-1] if h.episode_losses else 0,
                'successful_deals': h.successful_deals
            })
        pd.DataFrame(rl_metrics).to_csv(
            f"./results-convergence/model-drl/rl_metrics_year_{year+1}.csv", 
            index=False
        )

    def reset_households(self, households):
        """Reset household states for a new year"""
        for household_type, household_dict in households.items():
            for household in household_dict.values():
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


# Run the simulation
if __name__ == "__main__":
    print("Starting Housing Market Simulation with Deep RL")
    start_time = time.time()
    
    model = HousingMarket(years=15)
    model.simulate()
    
    duration = time.time() - start_time
    print(f"\nSimulation completed in {duration:.2f} seconds")