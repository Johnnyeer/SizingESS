import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the industrial and residential load profiles
industrial_load = pd.read_csv('DataCIGREInd15min2.csv')
residential_load = pd.read_csv('CIGREResiload.csv')

# Rename columns
industrial_load.columns = ['Time', 'Load_Industrial']
residential_load.columns = ['Time', 'Load_Residential']

# Merge
merged_load = pd.merge(industrial_load, residential_load, on='Time', how='inner')
merged_load = merged_load[merged_load['Time'] < 24.0]

# Group into 48 time steps (30-minute intervals)
merged_load['Group'] = merged_load.index // 2
merged_load_48 = merged_load.groupby('Group').agg({
    'Time': 'mean',
    'Load_Industrial': 'mean',
    'Load_Residential': 'mean'
}).reset_index(drop=True)

# Combined daily load (90% residential + 10% industrial)
combined_daily_profile = 0.9 * merged_load_48['Load_Residential'] + 0.1 * merged_load_48['Load_Industrial']

# Step 2: Plot
plt.figure(figsize=(14, 7))

plt.plot(merged_load_48['Time'], merged_load_48['Load_Residential'], label='Residential Load (30-min avg)', linewidth=2)
plt.plot(merged_load_48['Time'], merged_load_48['Load_Industrial'], label='Industrial Load (30-min avg)', linewidth=2)
plt.plot(merged_load_48['Time'], combined_daily_profile, label='Merged Load (90% Resi + 10% Ind)', linestyle='--', linewidth=3)

# Step 3: Beautify the Plot
plt.xlabel('Time of Day (Hours)', fontsize=12)
plt.ylabel('Load (pu)', fontsize=12)
plt.title('Comparison of Residential, Industrial, and Merged Load Profiles', fontsize=14)
plt.grid(True)
plt.legend()
plt.xticks(range(0, 25, 2))  # ticks every 2 hours
plt.tight_layout()

# Parameters for day-to-day variation
days_in_year = 366  # end in 1 Jan of next year
variation_mean = 1.0      # Centered at 1.0 (no bias)
variation_stddev = 0.05   # 5% standard deviation in daily load

# Generate random scaling factors for each day
daily_scaling_factors = np.random.normal(loc=variation_mean, scale=variation_stddev, size=days_in_year)

# Build the simulated year
simulated_year_load = np.hstack([
    daily_scaling_factors[day] * combined_daily_profile.values for day in range(days_in_year)
])

# Create corresponding time index
time_index = pd.date_range(start='2025-01-01 00:00', end='2026-01-01 23:30', freq='30min')

print(len(time_index))  # Should print 17568
print(len(simulated_year_load))  # Should print 17568


# Final DataFrame
simulated_load_df = pd.DataFrame({
    'Timestamp': time_index,
    'Load': simulated_year_load
})

# Plot a sample (e.g., first 4 days)
plt.figure(figsize=(14, 6))
plt.plot(simulated_load_df['Timestamp'][:4*48], simulated_load_df['Load'][:4*48])
plt.title('Simulated Composite Load Profile with Daily Fluctuations (First 4 Days)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Load (pu)', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save to CSV if needed
simulated_load_df.to_csv('Simulated_Composite_Load_Year_with_Fluctuations.csv', index=False)

# Find the current maximum load
current_max_load = simulated_load_df['Load'].max()

# Desired maximum load
desired_max_load = 500.0  # 100 kWh per 30 minutes

# Scaling factor
scaling_factor = desired_max_load / current_max_load

# Apply scaling
simulated_load_df['Load'] *= scaling_factor

# Save to CSV if needed
simulated_load_df.to_csv('Simulated_Composite_Load_Year_with_Fluctuations_Scaled.csv', index=False)

# Mean and Standard Deviation Calculations

# Residential
mean_residential = merged_load_48['Load_Residential'].mean()
std_residential = merged_load_48['Load_Residential'].std()

# Industrial
mean_industrial = merged_load_48['Load_Industrial'].mean()
std_industrial = merged_load_48['Load_Industrial'].std()

# Merged
mean_merged = combined_daily_profile.mean()
std_merged = combined_daily_profile.std()

# Print Results
print(f"Residential Load - Mean: {mean_residential:.4f}, Std: {std_residential:.4f}")
print(f"Industrial Load  - Mean: {mean_industrial:.4f}, Std: {std_industrial:.4f}")
print(f"Merged Load      - Mean: {mean_merged:.4f}, Std: {std_merged:.4f}")


# Plot a sample again to confirm
plt.figure(figsize=(14, 6))
plt.plot(simulated_load_df['Timestamp'][:4*48], simulated_load_df['Load'][:4*48])
plt.title('Simulated Load (Scaled to Max 500 kWh per Step)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Load (kWh)', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
