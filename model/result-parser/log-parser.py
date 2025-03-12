import re
import pandas as pd
import matplotlib.pyplot as plt

# Sample log data (Replace this with reading from a file)
# read a file
with open("../../logs/logs_rl.txt", 'r') as file:
    log_data = file.read()

# Regular expressions to extract data
household_pattern = re.compile(r"Household (\d+) chose house \d+ at \$(\d+\.\d+)")
bid_range_pattern = re.compile(r"Starting bids for Household ID: (\d+) with price range \$(\d+\.\d+) - \$(\d+\.\d+)")

# Data storage
households = []
chosen_prices = []
min_bids = []
max_bids = []

# Extract household choices (household ID and final price)
for match in household_pattern.finditer(log_data):
    households.append(int(match.group(1)))
    chosen_prices.append(float(match.group(2)))

# Extract bid price ranges
bid_dict = {}
for match in bid_range_pattern.finditer(log_data):
    bid_dict[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))

# Align bid ranges with household choices
for household in households:
    if household in bid_dict:
        min_bids.append(bid_dict[household][0])
        max_bids.append(bid_dict[household][1])
    else:
        min_bids.append(None)
        max_bids.append(None)

# Create a pandas DataFrame
df = pd.DataFrame({
    "Household": households,
    "Chosen Price": chosen_prices,
    "Min Bid": min_bids,
    "Max Bid": max_bids
})

# Sort by household ID
df.sort_values(by="Household", inplace=True)

# Print extracted data
# print(df)

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(df["Household"], df["Chosen Price"], marker='o', linestyle='-', color='b', label="Chosen House Price")
if df["Min Bid"].notna().any():
    plt.plot(df["Household"], df["Min Bid"], marker='x', linestyle='--', color='g', label="Min Bid")
if df["Max Bid"].notna().any():
    plt.plot(df["Household"], df["Max Bid"], marker='x', linestyle='--', color='r', label="Max Bid")

plt.xlabel("Household ID")
plt.ylabel("Price ($)")
plt.title("Household Price Trends")
plt.legend()
plt.grid(True)
plt.show()