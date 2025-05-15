import pandas as pd
import numpy as np
import mesa
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from mesa.time import RandomActivation
import csv
import random
import warnings
import time
from collections import defaultdict
from household import Household
import matplotlib.pyplot as plt
ignore_warnings = warnings.filterwarnings("ignore")
np.random.seed(0)

PERCENTAGE_ABOVE_WTP = 1.2
PERCENTAGE_BELOW_WTP = 0.8
ALPHA = 0.001 
GAMMA = 0.9 
EPSILON = 0.1  
NEGOTIATION_PENALTY_FACTOR = 0.05
NUM_HOUSE_SAMPLE = 15

class RLHousehold(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)
        self.unique_id = unique_id
        self.wtp = Household.calculate_wtp(self, household_data.loc[unique_id])
        self.lower_bound = self.wtp * PERCENTAGE_BELOW_WTP
        self.upper_bound = self.wtp * PERCENTAGE_ABOVE_WTP
        self.household_data = household_data
        self.successful_deals = 0
        self.chosen_house = None
        self.house_choices = {}
        self.successful_deals = 0
        self.deals = []
        self.memory = []
        self.model = self.build_model()
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.current_episode_reward = 0
        self.current_episode_loss = 0
    
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(1,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(3, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')
        return model
        
    def update_metrics(self, reward, loss):
        self.current_episode_reward += reward
        self.current_episode_loss += loss
        
    def end_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_losses.append(self.current_episode_loss)
        self.current_episode_reward = 0
        self.current_episode_loss = 0
        
    def plot_metrics(self, save_path=None):
        plt.figure(figsize=(12, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Mean Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_losses)
        plt.title('Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()