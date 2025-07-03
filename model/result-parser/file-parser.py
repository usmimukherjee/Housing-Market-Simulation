import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

df = pd.read_csv("../../results/model-deep-rl/bidding_results.csv")
output_dir = "../../output-analysis/model-deep-rl/"
os.makedirs(output_dir, exist_ok=True)

# Convert `negotiation_prices` from string to list of tuples
df["negotiation_prices"] = df["negotiation_prices"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Drop non-numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

#Basic Statistics as CSV
stats_df = numeric_df.describe()
stats_df.to_csv(os.path.join(output_dir, "basic_statistics.csv"))
print("\n Basic statistics saved to 'basic_statistics.csv'.")

# Distribution of Final Prices
plt.figure(figsize=(8, 5))
sns.histplot(df["final_price"], bins=20, kde=True)
plt.xlabel("Final Price")
plt.ylabel("Frequency")
plt.title("Distribution of Final Prices")
plt.savefig(os.path.join(output_dir, "final_price_distribution.png"))
plt.close()

#Negotiation Steps vs. Final Price
plt.figure(figsize=(8, 5))
sns.barplot(x=df["negotiation_steps"], y=df["final_price"])
plt.xlabel("Negotiation Steps")
plt.ylabel("Final Price")
plt.title("Negotiation Steps vs Final Price")
plt.savefig(os.path.join(output_dir, "negotiation_steps_vs_final_price.png"))
plt.close()

# Initial Bids vs. Final Prices
plt.figure(figsize=(8, 5))
sns.lineplot(x=df["initial_bid"], y=df["final_price"])
plt.xlabel("Initial Bid")
plt.ylabel("Final Price")
plt.title("Initial Bids vs Final Prices")
plt.savefig(os.path.join(output_dir, "initial_bid_vs_final_price.png"))
plt.close()

# Correlation Heatmap (Only Numeric Data)
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# Negotiation Trends for First 5 Households
plt.figure(figsize=(10, 6))
for i, row in df.head(5).iterrows():
    bid_prices = [p[0] for p in row["negotiation_prices"]]  
    counter_prices = [p[1] for p in row["negotiation_prices"]] 
    rounds = list(range(len(bid_prices)))

    plt.plot(rounds, bid_prices, marker='o', label=f'Household {row["household_id"]} (Bid)')
    plt.plot(rounds, counter_prices, marker='x', linestyle='dashed', label=f'Household {row["household_id"]} (Counter-offer)')

plt.xlabel("Negotiation Round")
plt.ylabel("Price")
plt.title("Negotiation Price Trends for First 5 Households")
plt.legend()
plt.savefig(os.path.join(output_dir, "negotiation_trends.png"))
plt.close()

print("\n Data Analysis Complete! Plots saved in 'analysis_plots' folder.")
