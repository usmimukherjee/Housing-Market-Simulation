import numpy as np
import mesa
from mesa.time import RandomActivation
import random
import warnings
from collections import defaultdict

ignore_warnings = warnings.filterwarnings("ignore")
np.random.seed(0)

PERCENTAGE_ABOVE_WTP = 1.2
PERCENTAGE_BELOW_WTP = 0.8

# Q-Learning Parameters
ALPHA = 0.1 
GAMMA = 0.9 
EPSILON = 0.1 

class Household(mesa.Agent):
    def __init__(self, unique_id, model, household_data):
        super().__init__(model)
        self.unique_id = unique_id
        self.wtp = self.calculate_wtp(household_data.loc[unique_id])
        household_data.loc[unique_id, 'wtp'] = self.wtp
        self.lower_bound, self.upper_bound = self.range_calculation(self.wtp)
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.successful_deals = 0
        self.deals = []

    def range_calculation(self, wtp):
        lower_bound = wtp * PERCENTAGE_BELOW_WTP
        upper_bound = wtp * PERCENTAGE_ABOVE_WTP
        return lower_bound, upper_bound

    def calculate_wtp(self, data):
        y_values = np.ones(10)
        self.wtp = (
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