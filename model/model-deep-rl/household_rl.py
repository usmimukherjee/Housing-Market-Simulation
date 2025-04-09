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
    
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(1,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(3, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')
        return model