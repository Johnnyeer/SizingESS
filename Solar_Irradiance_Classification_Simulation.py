import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA
import os

# --- FUNCTION TO LOAD MULTI-YEAR DATA ---
def load_data(data_dir, years):
    dfs = []
    for year in years:
        file_path = os.path.join(data_dir, f'NSRDB_Himawari_Singapore_{year}.csv')
        df = pd.read_csv(file_path, skiprows=2, low_memory=False)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# --- MAIN PROGRAM START ---
# Set data folder and years
data_dir = 'C://Users//johnn//Documents//NUS//CEG5001 Computer Engineering Project (Minor) I//NSRDB Himawari 2016-2020 Data'
years = [2016, 2017, 2018, 2019, 2020]

# Load the data
df = load_data(data_dir, years)

# --- CREATE DATETIME AND CONVERT TO SINGAPORE TIME ---
df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df['Datetime'] = df['Datetime'] + pd.Timedelta(hours=8)  # GMT -> SGT

# Set Datetime as index
df = df.set_index('Datetime')
df = df.resample('30T').mean()

# Create Date and Time columns
df['Date'] = df.index.date
df['HourMinute'] = df.index.strftime('%H:%M')

# --- CREATE DAILY GHI PROFILE (PIVOT) ---
pivot = df.pivot_table(index='Date', columns='HourMinute', values='GHI')
pivot = pivot.dropna()
profile_columns = [col for col in pivot.columns if ':' in col]
print(f"✅ Daily GHI profiles shape: {pivot.shape}")

# --- AGGREGATE DAILY EXTRA FEATURES ---
daily_extra_features = df.groupby(df.index.date).agg({
    'Clearsky DHI': 'mean',
    'Clearsky DNI': 'mean',
    'Clearsky GHI': 'mean',
    'Cloud Type': 'mean',
    'Ozone': 'mean',
    'Solar Zenith Angle': 'mean',
    'Precipitable Water': 'mean',
    'Temperature': 'mean',
    'Dew Point': 'mean',
    'DHI': 'mean',
    'DNI': 'mean',
    'Relative Humidity': 'mean',
    'Surface Albedo': 'mean',
    'Pressure': 'mean',
    'Wind Direction': 'mean',
    'Wind Speed': 'mean'
})

pivot_full = pivot.merge(daily_extra_features, left_index=True, right_index=True)
print(f"✅ Full feature set shape (GHI profile + meteorology): {pivot_full.shape}")

# --- SCALE ALL FEATURES ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_full)

# --- FIND BEST NUMBER OF CLUSTERS ---
inertia = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

knee_locator = KneeLocator(range(2, 11), inertia, curve="convex", direction="decreasing")
best_k = knee_locator.elbow
print(f"\n✅ Optimal number of clusters: {best_k}")

# --- FINAL CLUSTERING ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
pivot_full['cluster'] = kmeans.fit_predict(X_scaled)

# --- PCA FOR VISUALIZATION ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pivot_full['pca_1'] = X_pca[:, 0]
pivot_full['pca_2'] = X_pca[:, 1]

# --- CALCULATE PROBABILITIES FOR SEASONALITY CONTROL ---
pivot_full['Month'] = pd.to_datetime(pivot_full.index).month
month_cluster_counts = pivot_full.groupby(['Month', 'cluster']).size().unstack(fill_value=0)
month_cluster_probs = month_cluster_counts.div(month_cluster_counts.sum(axis=1), axis=0)
print("\n✅ Monthly Cluster Probabilities:")
print(month_cluster_probs)

# --- CALCULATE MEAN AND STD PER CLUSTER ---
cluster_means = pivot_full.groupby('cluster')[profile_columns].mean()
cluster_stds = pivot_full.groupby('cluster')[profile_columns].std()
print("\n✅ Per-cluster standard deviation calculated.")

# 2. Save cluster mean profiles (raw GHI)
cluster_means.to_csv('raw_cluster_mean_profiles.csv')
print("✅ Saved raw cluster mean profiles to 'raw_cluster_mean_profiles.csv'.")

# 3. Save cluster std profiles (raw GHI std)
cluster_stds.to_csv('raw_cluster_std_profiles.csv')
print("✅ Saved raw cluster std profiles to 'raw_cluster_std_profiles.csv'.")

# --- LABEL CLUSTERS (Clear Sky, Partly Cloudy, Cloudy, Overcast) ---
pivot_full['Total_GHI'] = pivot_full[profile_columns].sum(axis=1)
cluster_total_ghi = pivot_full.groupby('cluster')['Total_GHI'].mean()
sorted_clusters = cluster_total_ghi.sort_values(ascending=False)

fixed_labels = ['Clear Sky', 'Partly Cloudy', 'Cloudy', 'Overcast']
cluster_labels = {}
for idx, cluster_id in enumerate(sorted_clusters.index):
    if idx < len(fixed_labels):
        label = fixed_labels[idx]
    else:
        label = f"Other_{idx}"
    cluster_labels[cluster_id] = label
print("\n✅ Final Cluster Labels Assigned:")
for cid, lbl in cluster_labels.items():
    print(f"Cluster {cid}: {lbl}")

# --- Define Cluster Color Map ---
cluster_color_map = {
    'Clear Sky': 'blue',
    'Partly Cloudy': 'green',
    'Cloudy': 'orange',
    'Overcast': 'red'
}

# --- CREATE REALISTIC WEATHER MEMORY TRANSITION MATRIX ---
cluster_label_to_id = {v: k for k, v in cluster_labels.items()}
transition_matrix = {
    cluster_label_to_id['Clear Sky']: {
        cluster_label_to_id['Clear Sky']: 0.7,
        cluster_label_to_id['Partly Cloudy']: 0.2,
        cluster_label_to_id['Cloudy']: 0.1,
        cluster_label_to_id['Overcast']: 0.0
    },
    cluster_label_to_id['Partly Cloudy']: {
        cluster_label_to_id['Clear Sky']: 0.3,
        cluster_label_to_id['Partly Cloudy']: 0.4,
        cluster_label_to_id['Cloudy']: 0.2,
        cluster_label_to_id['Overcast']: 0.1
    },
    cluster_label_to_id['Cloudy']: {
        cluster_label_to_id['Clear Sky']: 0.1,
        cluster_label_to_id['Partly Cloudy']: 0.3,
        cluster_label_to_id['Cloudy']: 0.4,
        cluster_label_to_id['Overcast']: 0.2
    },
    cluster_label_to_id['Overcast']: {
        cluster_label_to_id['Clear Sky']: 0.0,
        cluster_label_to_id['Partly Cloudy']: 0.1,
        cluster_label_to_id['Cloudy']: 0.3,
        cluster_label_to_id['Overcast']: 0.6
    }
}

# --- PREPARE DATA FOR SAMPLING ---
clustered_days = {cluster_id: pivot_full[pivot_full['cluster'] == cluster_id] for cluster_id in pivot_full['cluster'].unique()}

# --- SIMULATE 2025 SOLAR YEAR ---
simulated_profiles = []
simulated_clusters = []
simulated_dates = pd.date_range(start='2025-01-01', end='2026-01-01')
np.random.seed(42)

first_month_probs = month_cluster_probs.loc[1]
current_cluster = np.random.choice(first_month_probs.index, p=first_month_probs.values)

for date in simulated_dates:
    month = date.month
    transition_probs = transition_matrix[current_cluster]
    month_probs = month_cluster_probs.loc[month]
    combined_probs = np.array([transition_probs[c] * month_probs.get(c, 0) for c in range(best_k)])
    combined_probs = combined_probs / combined_probs.sum()
    current_cluster = np.random.choice(range(best_k), p=combined_probs)

    selected_profile = clustered_days[current_cluster].sample(1)
    ghi_profile = selected_profile[profile_columns].values.flatten()
    std_profile = cluster_stds.loc[current_cluster, profile_columns].values
    noise = np.random.normal(0, std_profile)
    noisy_profile = ghi_profile + noise
    noisy_profile = np.clip(noisy_profile, 0, None)

    simulated_profiles.append(noisy_profile)
    simulated_clusters.append(current_cluster)

simulated_profiles = np.array(simulated_profiles)
simulated_df = pd.DataFrame(simulated_profiles, index=simulated_dates, columns=profile_columns)
print(f"\n✅ Simulated 2025 solar irradiance (full feature clustering + memory + variability): {simulated_df.shape}")

# --- PREPARE SIMULATED DAILY GHI ---
simulated_df['cluster'] = simulated_clusters
simulated_df['Cluster_Label'] = simulated_df['cluster'].map(cluster_labels)
simulated_df['Daily_GHI_Energy'] = simulated_df[profile_columns].sum(axis=1) * 0.5 / 1000

# --- SIMULATED YEAR SUMMARY ---
print("\n📈 Simulated 2025 Cluster Breakdown:")
cluster_counts = simulated_df['Cluster_Label'].value_counts()
for label in ['Clear Sky', 'Partly Cloudy', 'Cloudy', 'Overcast']:
    count = cluster_counts.get(label, 0)
    print(f"{label}: {count} days")
print(f"\n✅ Total simulated days: {simulated_df.shape[0]} days")

# --- EXPORT RAW (UNNORMALIZED) SIMULATED 2025 PROFILES ---

# 1. Save simulated 2025 raw GHI profiles
simulated_raw_df = simulated_df.copy()  # Before normalization
simulated_raw_df['Cluster_Label'] = simulated_df['Cluster_Label']
simulated_raw_df['Cluster_ID'] = simulated_df['cluster']

simulated_raw_df.to_csv('simulated_2025_raw_profiles.csv')
print("✅ Saved simulated 2025 raw GHI profiles to 'simulated_2025_raw_profiles.csv'.")

# --- PLOT CLUSTER MEAN PROFILES ± 1 STD DEV ---

plt.figure(figsize=(14, 8))

for cluster_id in cluster_means.index:
    mean_profile = cluster_means.loc[cluster_id]
    std_profile = cluster_stds.loc[cluster_id]

    label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
    color = cluster_color_map.get(label, 'black')  # Default black if unknown

    plt.plot(profile_columns, mean_profile, label=f'{label}', color=color)
    plt.fill_between(profile_columns,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     color=color, alpha=0.3)

plt.xticks(rotation=90)
plt.xlabel('Time of Day')
plt.ylabel('Clearsky GHI (W/m²)')
plt.title('Cluster Load Profiles (Mean ± 1 Std Dev)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Plot Mean Load Profile ± 1 Std for Each Cluster Individually (Color Coded) ---

for cluster_id in sorted(cluster_labels.keys()):
    label = cluster_labels[cluster_id]

    mean_profile = cluster_means.loc[cluster_id]
    std_profile = cluster_stds.loc[cluster_id]

    color = cluster_color_map.get(label, 'black')  # fallback color if label missing

    plt.figure(figsize=(12, 5))

    plt.plot(profile_columns, mean_profile, label=f'Mean GHI Profile', color=color, linewidth=2)
    plt.fill_between(profile_columns,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     alpha=0.3,
                     color=color,
                     label='±1 Std Dev')

    plt.xticks(rotation=90)
    plt.xlabel('Time of Day')
    plt.ylabel('GHI (W/m²)')
    plt.title(f'{label} - Average Solar Irradiance Profile')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# --- PLOT SIMULATED YEAR DAILY GHI ENERGY, COLORED BY CLUSTER ---

plt.figure(figsize=(16, 6))

for label in fixed_labels:
    mask = simulated_df['Cluster_Label'] == label
    color = cluster_color_map[label]
    plt.scatter(simulated_df.index[mask], simulated_df['Daily_GHI_Energy'][mask], label=label, s=20, color=color)

plt.xlabel('Date')
plt.ylabel('Daily GHI Energy (kWh/m²)')
plt.title('Simulated 2025 Solar Irradiance Year by Cluster Type')
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Plot Solar Irradiance vs Real Time Axis for First 7 Simulated Days ---

# Select first 7 days
first_7_days = simulated_df.iloc[:7]

# Reshape the GHI values into a continuous timeseries
timeseries_ghi = first_7_days[profile_columns].values.flatten()

# Build a real datetime index
start_datetime = pd.Timestamp('2025-01-01 00:00')
time_steps = pd.date_range(start=start_datetime, periods=len(timeseries_ghi), freq='30T')

# Plot
plt.figure(figsize=(16, 6))
plt.plot(time_steps, timeseries_ghi, marker='o', markersize=3, linewidth=1.5)

plt.xlabel('Time')
plt.ylabel('Simulated GHI (W/m²)')
plt.title('Simulated Solar Irradiance (W/m²) for First 7 Days of 2025')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()



'''
# --- READ Historical 2020 Data (Corrected) ---
historical_file_path = 'C://Users//johnn//Documents//NUS//CEG5001 Computer Engineering Project (Minor) I//NSRDB Himawari 2016-2020 Data//NSRDB_Himawari_Singapore_2020.csv'

df_hist = pd.read_csv(historical_file_path, skiprows=2, low_memory=False)

df_hist['Datetime'] = pd.to_datetime(df_hist[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df_hist['Datetime'] = df_hist['Datetime'] + pd.Timedelta(hours=8)  # GMT → SGT
df_hist = df_hist.set_index('Datetime')

# Time step detection
time_step_minutes = df_hist.index.to_series().diff().dropna().dt.total_seconds().mode()[0] / 60
time_step_hours = time_step_minutes / 60

print(f"✅ Detected historical file time step: {time_step_minutes} minutes ({time_step_hours:.3f} hours)")

historical_daily_energy = (df_hist['GHI'] * time_step_hours).resample('D').sum() / 1000  # W/m² × hr → Wh/m² ÷ 1000 → kWh/m²

print(f"✅ Historical 2020 daily GHI energy shape: {historical_daily_energy.shape}")

# --- PLOT HISTORICAL vs SIMULATED YEAR COMPARISON ---

plt.figure(figsize=(16, 6))

# Plot historical 2020 data (reference line)
# plt.plot(historical_daily_energy.index, historical_daily_energy.values, label='Historical 2020', color='grey', alpha=0.7, linewidth=1.5)

# Plot simulated 2025 points color-coded
for label in fixed_labels:
    mask = simulated_df['Cluster_Label'] == label
    color = cluster_color_map[label]
    plt.scatter(simulated_df.index[mask], simulated_df['Daily_GHI_Energy'][mask], label=f'Simulated - {label}', s=20, color=color)

plt.xlabel('Date')
plt.ylabel('Daily Solar Energy (kWh/m²)')
plt.title('Comparison: Historical 2020 vs Simulated 2025 Solar Energy')
plt.legend()
plt.grid(True)
plt.tight_layout()
'''

# --- Normalize and Plot Mean Load Profile ± 1 Std for Each Cluster Individually (Global Scale) ---

# Step 1: Find Global Maximum across all cluster means
global_max = cluster_means[profile_columns].max().max()

print(f"✅ Global maximum GHI across all clusters = {global_max:.2f} W/m²")

# --- EXPORT NORMALIZED MEAN AND STD PROFILES OF EACH CLUSTER ---

# 1. Prepare mean and std profiles normalized by the global max
normalized_cluster_means = cluster_means / global_max
normalized_cluster_stds = cluster_stds / global_max

# 2. Save mean profiles
normalized_cluster_means.to_csv('normalized_cluster_mean_profiles.csv')
print("✅ Saved normalized cluster mean profiles to 'normalized_cluster_mean_profiles.csv'.")

# 3. Save std profiles
normalized_cluster_stds.to_csv('normalized_cluster_std_profiles.csv')
print("✅ Saved normalized cluster std profiles to 'normalized_cluster_std_profiles.csv'.")

# Step 2: Plot normalized profiles
for cluster_id in sorted(cluster_labels.keys()):
    label = cluster_labels[cluster_id]

    mean_profile = cluster_means.loc[cluster_id] / global_max
    std_profile = cluster_stds.loc[cluster_id] / global_max

    color = cluster_color_map.get(label, 'black')  # fallback color if label missing

    plt.figure(figsize=(12, 5))

    plt.plot(profile_columns, mean_profile, label='Normalized Mean Profile', color=color, linewidth=2)
    plt.fill_between(profile_columns,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     alpha=0.2,
                     color=color,
                     label='±1 Std Dev')

    plt.xticks(rotation=90)
    plt.xlabel('Time of Day')
    plt.ylabel('Normalized GHI (fraction of global max)')
    plt.title(f'{label} - Normalized Solar Irradiance Profile')
    plt.ylim(0, 1.2)  # keep same y-scale for all
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Step 2: Create combined plot
plt.figure(figsize=(14, 6))

for cluster_id in sorted(cluster_labels.keys()):
    label = cluster_labels[cluster_id]

    mean_profile = cluster_means.loc[cluster_id] / global_max
    std_profile = cluster_stds.loc[cluster_id] / global_max

    color = cluster_color_map.get(label, 'black')  # fallback color if label missing

    plt.plot(profile_columns, mean_profile, label=f'{label}', color=color, linewidth=2)
    plt.fill_between(profile_columns,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     color=color,
                     alpha=0.2)

plt.xticks(rotation=90)
plt.xlabel('Time of Day')
plt.ylabel('Normalized GHI (fraction of global max)')
plt.title('Normalized Solar Irradiance Profiles by Cluster (Overlay)')
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Normalize Simulated Year Profiles (Consistent with Cluster Profiles) ---

# Step 1: Already computed global_max from cluster_means
# (If not, recompute:)
# global_max = cluster_means[profile_columns].max().max()

# Step 2: Normalize each profile by the same global max
simulated_normalized_df = simulated_df.copy()
simulated_normalized_df[profile_columns] = simulated_normalized_df[profile_columns] / global_max

print("\n✅ Simulated 2025 profiles normalized using cluster global max.")

# --- Calculate Normalized Daily GHI Energy for Simulated Year ---

# Remember: Daily energy is the sum of all 48 steps × (0.5 hr) × (W/m²) ÷ 1000 = kWh/m²
# Now since GHI is normalized (unitless), we adjust:

simulated_normalized_df['Normalized_Daily_GHI_Energy'] = simulated_normalized_df[profile_columns].sum(axis=1) * 0.5

# (0.5 because each step is 30 minutes = 0.5 hour)

# --- EXPORT SIMULATED 2025 DATA TO CSV ---

# 1. Export full 30-minute resolution normalized GHI profiles
simulated_normalized_df_output = simulated_normalized_df.copy()
simulated_normalized_df_output['Cluster_Label'] = simulated_df['Cluster_Label']  # Copy cluster labels
simulated_normalized_df_output['Cluster_ID'] = simulated_df['cluster']            # Copy cluster IDs

simulated_normalized_df_output.to_csv('simulated_2025_normalized_profiles.csv')
print("✅ Saved simulated 2025 normalized GHI profiles to 'simulated_2025_normalized_profiles.csv'.")

# 2. Export daily normalized energy and cluster label
daily_summary_df = simulated_normalized_df[['Normalized_Daily_GHI_Energy']].copy()
daily_summary_df['Cluster_Label'] = simulated_df['Cluster_Label']
daily_summary_df['Cluster_ID'] = simulated_df['cluster']

daily_summary_df.to_csv('simulated_2025_daily_summary.csv')
print("✅ Saved simulated 2025 daily GHI energy summary to 'simulated_2025_daily_summary.csv'.")

# --- Plot Normalized Daily GHI Energy by Cluster ---

plt.figure(figsize=(16, 6))

# Plot each cluster separately
for label in fixed_labels:
    mask = simulated_normalized_df['Cluster_Label'] == label
    plt.scatter(simulated_normalized_df.index[mask],
                simulated_normalized_df['Normalized_Daily_GHI_Energy'][mask],
                label=label,
                color=cluster_color_map[label],
                s=20)

plt.xlabel('Date')
plt.ylabel('Normalized Daily Solar Energy (relative)')
plt.title('Normalized Simulated 2025 Daily Solar Energy by Cluster Type')
plt.legend()
plt.grid(True)
plt.tight_layout()


# --- SHOW ALL PLOTS AT THE END ---
plt.show()




