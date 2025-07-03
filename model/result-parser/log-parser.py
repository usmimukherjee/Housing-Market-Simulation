import re
import pandas as pd
import matplotlib.pyplot as plt

with open("../../logs/deep-rl-log.txt", 'r') as file:
    log_data = file.read()

household_pattern = re.compile(r"Household (\d+) chose house \d+ at \$(\d+\.\d+)")
bid_range_pattern = re.compile(r"Starting bids for Household ID: (\d+) with price range \$(\d+\.\d+) - \$(\d+\.\d+)")

households = []
chosen_prices = []
min_bids = []
max_bids = []

for match in household_pattern.finditer(log_data):
    households.append(int(match.group(1)))
    chosen_prices.append(float(match.group(2)))

bid_dict = {}
for match in bid_range_pattern.finditer(log_data):
    bid_dict[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))

for household in households:
    if household in bid_dict:
        min_bids.append(bid_dict[household][0])
        max_bids.append(bid_dict[household][1])
    else:
        min_bids.append(None)
        max_bids.append(None)

df = pd.DataFrame({
    "Household": households,
    "Chosen Price": chosen_prices,
    "Min Bid": min_bids,
    "Max Bid": max_bids
})

# Sort by household ID
df.sort_values(by="Household", inplace=True)
df.to_csv("../../results/model-deep-rl/household_prices.csv", index=False)