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

class House:
    def __init__(self, data):
        self.id = data['OBJECTID']
        self.status = "vacant"  # "vacant", "for sale", "occupied"
        self.price = data['PUBVALUE']
        self.original_price = data['PUBVALUE']  # Store original price
        self.years_occupied = 0