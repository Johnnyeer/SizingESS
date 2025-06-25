import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from typing import List, Tuple
import matplotlib.dates as mdates
from datetime import timedelta, datetime
from itertools import product



class ESSPortfolio:
    def __init__(self, capacities: List[float],
                 efficiencies: List[Tuple[float, float, float]],
                 capex_list: List[float],
                 cycle_lives: List[float]):
        self.capacities = capacities
        self.efficiencies = efficiencies
        self.capex_list = capex_list
        self.num_units = len(capacities)
        self.cycle_lives = cycle_lives

        if not (len(capacities) == len(efficiencies) == len(capex_list)):
            raise ValueError("Capacities, efficiencies, and capex_list must have the same length")


def optimize_portfolio(portfolio: ESSPortfolio,
                       demand_forecast,
                       solar_power_forecast,
                       t_horizon,
                       n_days):

    q_current = [0 for Q in portfolio.capacities]
    total_profit = 0.0
    q_history = [q_current.copy()]

    t_total = n_days * t_horizon

    s_output = np.zeros(t_total)
    d_output = np.zeros(t_total)
    u_output = np.zeros((t_total, portfolio.num_units))
    u_c_output = np.zeros((t_total, portfolio.num_units))
    u_d_output = np.zeros((t_total, portfolio.num_units))
    a_output = np.zeros(t_total)

    p_d_kWh = 0.5  # $/kWh
    p_s_kWh = 1.0  # $/kWh
    p_d = p_s_kWh / 10  # normalised
    p_s = p_d_kWh / 10  # normalised
    p_d_MWh = p_d_kWh * 1000  # $/MWh
    p_s_MWh = p_s_kWh * 1000  # $/MWh

    capex = sum(portfolio.capex_list)
    capex_per_t_total = capex * t_total / (365 * 10 * 48)

    # 1. Define Variables once
    s = cp.Variable(t_horizon)
    d = cp.Variable(t_horizon)
    a = cp.Variable(t_horizon)
    q = [cp.Variable(t_horizon + 1) for _ in range(portfolio.num_units)]
    u_c = [cp.Variable(t_horizon, nonneg=True) for _ in range(portfolio.num_units)]
    u_d = [cp.Variable(t_horizon, nonneg=True) for _ in range(portfolio.num_units)]

    # 2. Define Parameters once
    r_hat_param = cp.Parameter(t_horizon)
    a_hat_param = cp.Parameter(t_horizon)
    q0_param = [cp.Parameter() for _ in range(portfolio.num_units)]

    # 3. Build constraints once
    constraints = [s >= 0, d >= 0, d == r_hat_param, a <= a_hat_param]
    for i in range(portfolio.num_units):
        eta_c, eta_d, eta_l = portfolio.efficiencies[i]
        unit_cap = portfolio.capacities[i]
        unit_max_rate = unit_cap / 2
        constraints += [
            q[i][0] == q0_param[i],
            q[i][1:] == eta_l * q[i][:-1] + eta_c * u_c[i] - u_d[i] / eta_d,
            q[i] >= 0,
            q[i] <= unit_cap,
            u_c[i] <= unit_max_rate,
            u_d[i] <= unit_max_rate,
            u_c[i] + u_d[i] <= unit_max_rate,
        ]

    grid_balance = (
            d
            + cp.sum([u_c[i] / portfolio.efficiencies[i][0] for i in range(portfolio.num_units)], axis=0)
            - cp.sum([portfolio.efficiencies[i][1] * u_d[i] for i in range(portfolio.num_units)], axis=0)
            - s
            - a
    )
    constraints += [grid_balance == 0]

    # 4. Define objective once
    penalty = cp.sum([cp.sum(u_c[i] + u_d[i]) for i in range(portfolio.num_units)])
    objective = cp.Maximize(cp.sum(p_d * d) - cp.sum(p_s * s) - 0.01 * penalty)

    # 5. Build problem once
    prob = cp.Problem(objective, constraints)

    for t in range(t_total):
        # Update Parameters
        r_hat_param.value = demand_forecast[t:t + t_horizon]
        a_hat_param.value = solar_power_forecast[t:t + t_horizon]
        for i in range(portfolio.num_units):
            q0_param[i].value = q_current[i]

        # Solve
        prob.solve(solver=cp.ECOS, verbose=False, warm_start=True, abstol=1e-5, reltol=1e-5)

        # Extract first step
        s_output[t] = s.value[0]
        d_output[t] = d.value[0]
        a_output[t] = a.value[0]
        for i in range(portfolio.num_units):
            u_c_output[t, i] = u_c[i].value[0]
            u_d_output[t, i] = u_d[i].value[0]
            u_output[t, i] = u_c[i].value[0] - u_d[i].value[0]
            q_current[i] = q[i].value[1]

        q_history.append(q_current.copy())

        total_profit += p_d_MWh * d_output[t] - p_s_MWh * s_output[t]

    return total_profit - capex_per_t_total, s_output, d_output, q_history, u_output, u_c_output, u_d_output, capex, a_output

def main():
    print("Loading data...")
    demand_df = pd.read_csv("Simulated_Composite_Load_Year_with_Fluctuations_Scaled.csv")
    solar_df = pd.read_csv("simulated_2025_raw_profiles.csv")
    print(solar_df.shape)

    print("Preparing data...")
    # Demand profile
    demand_series_raw = demand_df['Load'].values
    demand_series = demand_series_raw / 1000

    # Solar profile
    # Half-hour timestamps expected
    expected_times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    # Select the valid 48 time columns
    time_cols = [col for col in solar_df.columns if col in expected_times]
    ghi_series = np.array(solar_df[time_cols].values.flatten())
    # Solar Energy Generation Calculation
    eta_pv = 0.2
    area_pv = 10000  #m2
    solar_power_series_raw = (ghi_series / 1000) * eta_pv * area_pv * 0.5 # (W/m²) × (m²) × (h) → kWh
    solar_power_series = solar_power_series_raw / 1000  # MWh

    print("Setting up optimization parameters...")
    T_horizon = 48
    n_days = 365

    ESS_Type1_size = 1
    ESS_Type2_size = 1
    ESS_Type3_size = 1

    ESS_TYPES = [{"capacity": ESS_Type1_size, "efficiency": (0.95, 0.95, 0.995), "capex": ESS_Type1_size * 1200000, "cycle_life": 5000},
                 {"capacity": ESS_Type2_size, "efficiency": (0.9, 0.9, 0.99), "capex": ESS_Type2_size * 800000, "cycle_life": 5000},
                 {"capacity": ESS_Type3_size, "efficiency": (0.8, 0.8, 0.98), "capex": ESS_Type3_size * 400000, "cycle_life": 5000}]

    max_num_units = 8
    portfolio_candidates = []
    portfolio_metadata = []
    # Step 1: Generate all possible combinations using numpy
    combinations = np.array(list(product(range(max_num_units + 1), repeat=3)))

    # Step 2: Filter combinations where total units ≤ max_num_units
    valid_mask = np.sum(combinations, axis=1) <= max_num_units
    valid_combinations = combinations[valid_mask]

    for n1, n2, n3 in valid_combinations:
        capacities = (
                [ESS_TYPES[0]["capacity"]] * n1 +
                [ESS_TYPES[1]["capacity"]] * n2 +
                [ESS_TYPES[2]["capacity"]] * n3
        )
        efficiencies = (
                [ESS_TYPES[0]["efficiency"]] * n1 +
                [ESS_TYPES[1]["efficiency"]] * n2 +
                [ESS_TYPES[2]["efficiency"]] * n3
        )
        capex_list = (
                [ESS_TYPES[0]["capex"]] * n1 +
                [ESS_TYPES[1]["capex"]] * n2 +
                [ESS_TYPES[2]["capex"]] * n3
        )
        cycle_lives = (
                [ESS_TYPES[0]["cycle_life"]] * n1 +
                [ESS_TYPES[1]["cycle_life"]] * n2 +
                [ESS_TYPES[2]["cycle_life"]] * n3
        )

        portfolio_candidates.append(
            ESSPortfolio(capacities, efficiencies, capex_list, cycle_lives)
        )
        portfolio_metadata.append((n1, n2, n3))

    print("Starting parallel portfolio optimization...")
    start_time = time.time()

    optimize_func = partial( optimize_portfolio,
                             demand_forecast=demand_series,
                             solar_power_forecast=solar_power_series,
                             t_horizon=T_horizon,
                             n_days=n_days)

    total_portfolios = len(portfolio_candidates)
    print(f"Total portfolios to optimize: {total_portfolios}")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(optimize_func, portfolio_candidates))

    raw_results = [r for r in results if r is not None]

    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")

    profit = np.array([r[0] for r in raw_results])
    Capex = np.array([r[7] for r in raw_results])

    optimal_idx = np.argmax(profit)
    optimal_portfolio = portfolio_candidates[optimal_idx]
    optimal_result = raw_results[optimal_idx]  # Store the optimal result for visualization
    total_capacity = sum(optimal_portfolio.capacities)

    # Get (n1, n2, n3) for the optimal portfolio
    optimal_metadata = portfolio_metadata[optimal_idx]

    # Rebuild unit type list
    optimal_unit_types = (
            [1] * optimal_metadata[0] +  # Type 1
            [2] * optimal_metadata[1] +  # Type 2
            [3] * optimal_metadata[2]    # Type 3
    )

    print(f"\n✅ Optimal portfolio configuration:")
    for i, (cap, eff) in enumerate(zip(optimal_portfolio.capacities, optimal_portfolio.efficiencies)):
        print(f"ESS Unit {i + 1}:")
        print(f"  Type: ESS Type {optimal_unit_types[i]}")
        print(f"  Capacity: {cap:.2f} MWh")
        print(f"  Efficiencies: charging={eff[0]}, discharging={eff[1]}, self-discharge={eff[2]}")
    print(f"💰 Operating Profit: {profit[optimal_idx]:.2f}")

    # Save results
    results_df = pd.DataFrame({
        "Total_Capacity_kWh": [sum(p.capacities) for p in portfolio_candidates],
        "Num_ESS_Units": [len(p.capacities) for p in portfolio_candidates],
        "Num_ESS_1": [meta[0] for meta in portfolio_metadata],  # Type 1 units
        "Num_ESS_2": [meta[1] for meta in portfolio_metadata],  # Type 2 units
        "Num_ESS_3": [meta[2] for meta in portfolio_metadata],  # Type 3 units
        "Operating_Profit": profit,
        "CAPEX": Capex,
    })
    results_df.to_csv("ess_analysis_usingProfiles_final.csv", index=False)

    # Save results
    results_df = pd.DataFrame({
        "Total_Capacity_kWh": [sum(p.capacities) for p in portfolio_candidates],
        "Num_ESS_Units": [len(p.capacities) for p in portfolio_candidates],
        "Num_ESS_1": [meta[0] for meta in portfolio_metadata],  # Type 1 units
        "Num_ESS_2": [meta[1] for meta in portfolio_metadata],  # Type 2 units
        "Num_ESS_3": [meta[2] for meta in portfolio_metadata],  # Type 3 units
        "Operating_Profit": profit,
        "CAPEX": Capex,
    })
    results_df.to_csv("ess_analysis_usingProfiles01.csv", index=False)

    # -----------------------------
    # 8. Visualization
    # -----------------------------
    print("Generating plots...")
    N_days = 3
    time_axis = [datetime(2025, 1, 1) + timedelta(minutes=30 * i) for i in range(T_horizon * N_days)]

    # Extract data from optimal result
    s_output = optimal_result[1]
    d_output = optimal_result[2]
    q_history = np.array(optimal_result[3])
    q_history_total = np.sum(q_history, axis=1)
    u_output = optimal_result[4]
    u_c_output = optimal_result[5]
    u_d_output = optimal_result[6]
    u_d_total = np.sum(u_d_output, axis=1)  # sum across units for each timestep
    a_output = optimal_result[8]

    # Plot 1: Demand vs Delivered Energy
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, demand_series[:T_horizon * N_days], label='Demand (MW)', linestyle='--', color='black')
    plt.plot(time_axis, d_output[:T_horizon * N_days], label='Delivered Energy (MW)', color='tab:green')
    plt.ylabel('Energy (MWh)')
    plt.title('Demand vs Delivered Energy')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.tight_layout()

    # Plot 2: Charge (+) and Discharge (-) per ESS Unit
    plt.figure(figsize=(12, 6))
    for i in range(u_output.shape[1]):
        plt.plot(time_axis, u_d_output[:T_horizon * N_days, i], label=f'ESS Unit {i + 1} Discharge Rate')
        plt.plot(time_axis, u_c_output[:T_horizon * N_days, i], label=f'ESS Unit {i + 1} Charge Rate')
        plt.plot(time_axis, q_history[:T_horizon * N_days, i], label=f'ESS Unit {i + 1} Charge Level')
    plt.ylabel('Energy (MWh)')
    plt.title('Charge (+) and Discharge (-) Over Time per ESS Unit')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.tight_layout()

    # Plot 3: Charge (+) and Discharge (-) per ESS Unit
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, q_history_total[:T_horizon * N_days], label='Charge Level')
    for i in range(u_output.shape[1]):
        plt.plot(time_axis, q_history[:T_horizon * N_days, i], label=f'ESS Unit {i + 1} Charge Level')
        plt.plot(time_axis, u_c_output[:T_horizon * N_days, i] - u_d_output[:T_horizon * N_days, i], label=f'ESS Unit {i + 1} Charge Rate')
    plt.ylabel('Energy (MWh)')
    plt.title('Charge Levels Over Time')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.tight_layout()

    # Plot 4: Grid Balance Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, solar_power_series[:T_horizon * N_days], label='Solar Generation (MWh)', color='orange')
    plt.plot(time_axis, demand_series[:T_horizon * N_days], label='Demand (MW)', linestyle='--', color='black')
    plt.plot(time_axis, d_output[:T_horizon * N_days], label='Delivered Energy (MW)', color='tab:green')
    plt.plot(time_axis, u_output[:T_horizon * N_days], label='Energy Charge Rate (MWh)', color='tab:purple')
    plt.plot(time_axis, a_output[:T_horizon * N_days], label='Solar Energy Injected(MWh)', linestyle='--', color='yellow')
    plt.plot(time_axis, s_output[:T_horizon * N_days], label='Grid Energy Delivered (MWh)', linestyle='--', color='red')
    plt.ylabel('Energy (MWh)')
    plt.title('Grid Balance Over Time')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()