from housing_market import HousingMarket
import time
# Run the simulation
if __name__ == "__main__":
    print("Starting Housing Market Simulation with Deep RL")
    start_time = time.time()
    
    model = HousingMarket(years=5)
    model.simulate()
    
    duration = time.time() - start_time
    print(f"\nSimulation completed in {duration:.2f} seconds")