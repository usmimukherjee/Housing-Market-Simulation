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
from dqn_network import DQN
from replay_memory import ReplayMemory
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
