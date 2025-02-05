import pandas as pd
import numpy as np
import mesa
import os
from mesa.time import RandomActivation
import csv
import random
import warnings
import time
from collections import defaultdict

from housing_market import HousingMarket

model = HousingMarket(years=1)
model.simulate()