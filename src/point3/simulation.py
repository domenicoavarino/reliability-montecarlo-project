import numpy as np
import matplotlib.pyplot as plt

from src.point3.system import simulate_system_once, system_state_on_grid

"""
MONTE CARLO SIMULATION MODULE - POINT 3
---------------------------------------
This script orchestrates the advanced Monte Carlo simulation for the fully 
repairable system under realistic constraints. It models limited maintenance 
resources (FIFO repair queues) and Common Cause Failures (external shocks).

It computes and plots the following reliability and maintainability metrics:
  i.   R(t)       — Time-dependent Reliability
  ii.  MTTF       — Mean Time To First Failure (with right-censoring)
  iii. A(t)       — Instantaneous Availability
  iv.  Ā          — Average Availability over [0, T_M]
  v.   E[N_rep]   — Expected number of completed repairs in [0, T_M]
  vi.  Statistical uncertainties (standard deviations) for all estimates.
"""
# ── System Parameters ──────────────────────────────────────────────────────
LAMBDA_1     = 1e-3    # failure rate A, B, C [h⁻¹]
LAMBDA_2     = 2e-3    # failure rate D, E    [h⁻¹]
MU_1         = 1e-2    # repair rate  A, B, C [h⁻¹]  → MTTR_ABC = 100 h
MU_2         = 2e-2    # repair rate  D, E    [h⁻¹]  → MTTR_DE  =  50 h
LAMBDA_C     = 2e-3    # shock arrival rate   [h⁻¹]
P_SHOCK      = 0.1     # Probability of component failure per shock

MISSION_TIME  = 1000
N_SIMULATIONS = 100_000
N_GRID        = 200


# ── Full History Simulation ───────────────────────────────────────────────────

def run_simulation(n_simulations, lambda_1, lambda_2, mu_1, mu_2,
                   lambda_c, p_shock, mission_time, time_grid):
    """
    Executes N independent event-driven Monte Carlo simulations for the Point 3 
    system (incorporating FIFO repair queues and external shocks).

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo histories.
    lambda_1, lambda_2, mu_1, mu_2, lambda_c, p_shock, mission_time : float
        System and physical transition parameters.
    time_grid : np.ndarray
        Array of discrete times to project and sample the system state.

    Returns
    -------
    tuple containing:
        failure_times : np.ndarray (N,) — Absolute time of FIRST system failure.
        time_ups      : np.ndarray (N,) — Cumulative UP time per run.
        n_repairs_arr : np.ndarray (N,) — Total completed repairs per run.
        states_matrix : np.ndarray (N, len(time_grid)) — Binary system state 
                        (1=UP, 0=DOWN) sampled at each point in time_grid.
    """
    n_grid        = len(time_grid)
    # Pre-allocate memory for computational efficiency
    failure_times = np.empty(n_simulations)
    time_ups      = np.empty(n_simulations)
    n_repairs_arr = np.empty(n_simulations, dtype=int)
    states_matrix = np.zeros((n_simulations, n_grid), dtype=np.int8)

    for i in range(n_simulations):
        # Run the advanced Discrete Event Simulation for a single history
        res = simulate_system_once(
            lambda_1, lambda_2, mu_1, mu_2,
            lambda_c, p_shock, mission_time
        )
        failure_times[i] = res["t_first_failure"]
        time_ups[i]      = res["time_up"]
        n_repairs_arr[i] = res["n_repairs"]
        # Reconstruct the instantaneous state array from the sparse transitions list
        states_matrix[i] = system_state_on_grid(res["state_at"], time_grid)

    return failure_times, time_ups, n_repairs_arr, states_matrix


# ── Statistical Estimation Functions ─────────────────────────────────────────────────────────

def estimate_reliability(failure_times, time_grid):
    """
    Estimates R(t) = P(T_first_failure > t) and its standard deviation.
    """
    n     = len(failure_times)
    R     = np.array([np.mean(failure_times > t) for t in time_grid])
    R_std = np.sqrt(R * (1 - R) / n)
    return R, R_std


def estimate_availability_pointwise(states_matrix):
    """
    Estimates Instantaneous Availability A(t).
    Computed as the ensemble average (across all runs) of the binary system 
    state (1=UP, 0=DOWN) at each specific grid point.
    """
    n       = states_matrix.shape[0]
    A_t     = np.mean(states_matrix, axis=0)
    A_t_std = np.sqrt(A_t * (1 - A_t) / n)
    return A_t, A_t_std


def estimate_average_availability(time_ups, mission_time, n_simulations):
    """
    Estimates the Average Availability over the mission time.
    Ā = E[Total Time UP] / Mission Time.
    """
    A_avg     = np.mean(time_ups) / mission_time
    A_avg_std = (np.std(time_ups, ddof=1) / np.sqrt(n_simulations)) / mission_time
    return A_avg, A_avg_std


def estimate_mttf(failure_times, mission_time):
    """
    Estimates the Mean Time To First Failure (MTTF).

    RIGHT-CENSORING LOGIC:
    Since the simulation is strictly truncated at the mission time T_M, runs 
    where the system never failed record a failure time of infinity. Excluding 
    these "survival" runs would create a severe selection bias. We apply 
    "right-censoring" by capping these infinite values at T_M. This provides a 
    mathematically rigorous and conservative lower-bound estimate of the MTTF.
    """
    n           = len(failure_times)
    # Apply Right-Censoring: replace 'inf' with 'mission_time'
    ft_censored = np.where(np.isinf(failure_times), mission_time, failure_times)
    mttf        = np.mean(ft_censored)
    mttf_std    = np.std(ft_censored, ddof=1) / np.sqrt(n)
    return mttf, mttf_std


def estimate_n_repairs(n_repairs_arr):
    """
    Estimates the expected number of completed repairs per run (E[N_rep]).
    """
    n        = len(n_repairs_arr)
    mean_rep = np.mean(n_repairs_arr)
    std_rep  = np.std(n_repairs_arr, ddof=1) / np.sqrt(n)
    return mean_rep, std_rep


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Fix the seed for reproducibility, consistent across all project points
    np.random.seed(42)

    time_grid = np.linspace(0, MISSION_TIME, N_GRID)

    print(f"Executing {N_SIMULATIONS:,} Monte Carlo simulations (Point 3)...")
    failure_times, time_ups, n_repairs_arr, states_matrix = run_simulation(
        N_SIMULATIONS, LAMBDA_1, LAMBDA_2, MU_1, MU_2,
        LAMBDA_C, P_SHOCK, MISSION_TIME, time_grid
    )
    print("Simulations completed.")

    # ── i. R(t) ──────────────────────────────────────────────────────────────
    R, R_std = estimate_reliability(failure_times, time_grid)

    # ── ii. MTTF ─────────────────────────────────────────────────────────────
    mttf, mttf_std = estimate_mttf(failure_times, MISSION_TIME)

    # ── iii. A(t) ─────────────────────────────────────────────────────────────
    A_t, A_t_std = estimate_availability_pointwise(states_matrix)

    # ── iv. Ā ─────────────────────────────────────────────────────────────────
    A_avg, A_avg_std = estimate_average_availability(time_ups, MISSION_TIME, N_SIMULATIONS)

    # ── v. E[N_rep] ───────────────────────────────────────────────────────────
    mean_rep, std_rep = estimate_n_repairs(n_repairs_arr)

    # ── Figure 1: R(t)  ────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R, color="tab:blue", label="R(t) Monte Carlo")
    plt.fill_between(
        time_grid, R - R_std, R + R_std,
        alpha=0.3, color="tab:blue", label="Uncertainty ±1σ"
    )
    plt.xlabel("Time [h]")
    plt.ylabel("R(t)")
    plt.title("Time-Dependent Reliability — Point 3")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # ── Figure 2: R_std(t) ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R_std, color="tab:blue")
    plt.xlabel("Time [h]")
    plt.ylabel("σ[R(t)]")
    plt.title("Reliability Standard Deviation — Point 3")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── Figure 3: A(t) con banda di incertezza ────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, A_t, color="tab:orange", label="A(t) Monte Carlo")
    plt.fill_between(
        time_grid, A_t - A_t_std, A_t + A_t_std,
        alpha=0.3, color="tab:orange", label="Uncertainty ±1σ"
    )
    plt.axhline(
        A_avg, color="tab:red", linestyle=":",
        label=f"Ā (avg) = {A_avg:.4f} ± {A_avg_std:.4e}"
    )
    plt.xlabel("Time [h]")
    plt.ylabel("A(t)")
    plt.title("Instantaneous Availability — Point 3")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # ── Figure 4: A_std(t) ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, A_t_std, color="tab:orange")
    plt.xlabel("Time [h]")
    plt.ylabel("σ[A(t)]")
    plt.title("Availability Standard Deviation — Point 3")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── Print risultati ───────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("          RESULTS — POINT 3")
    print("=" * 52)
    print(f"  Simulations        : {N_SIMULATIONS:,}")
    print(f"  Mission time       : {MISSION_TIME} h")
    print(f"  Shock rate λ_c     : {LAMBDA_C} h⁻¹")
    print(f"  Shock probability  : {P_SHOCK}")
    print("-" * 52)
    print(f"  R(T_M)             = {R[-1]:.4f}  ±  {R_std[-1]:.4e}")
    print(f"  MTTF               = {mttf:.2f}  ±  {mttf_std:.2f} h")
    print(f"  A(T_M)             = {A_t[-1]:.4f}  ±  {A_t_std[-1]:.4e}")
    print(f"  Ā  [0, T_M]        = {A_avg:.4f}  ±  {A_avg_std:.4e}")
    print(f"  E[N_repairs]       = {mean_rep:.2f}  ±  {std_rep:.4f}")
    print("=" * 52)

    # =========================================================================
    # PHYSICAL INTERPRETATION OF THE RESULTS (POINT 3 vs POINT 2)
    # =========================================================================
    # 1. Impact of Limited Repair Resources (Queuing/Bottlenecks):
    # In Point 2, each component had an independent repair team. In Point 3, 
    # we introduced a bottleneck: only 1 team per subsystem. When multiple 
    # failures occur, components are forced into a FIFO waiting queue. This 
    # significantly increases the effective downtime (MTTR) of the components, 
    # drastically reducing the overall Instantaneous and Average Availability A(t).
    #
    # 2. Impact of Common Cause Failures (External Shocks):
    # The external shocks introduce dependent failures, defeating the primary 
    # purpose of redundancy (Subsystems 2oo3 and Parallel). A single shock can 
    # simultaneously disable multiple operational components. 
    # As a direct consequence, the Reliability R(t) curve and the MTTF exhibit 
    # a sharp degradation compared to Point 2.
    # 
    # CONCLUSION: 
    # Due to maintenance bottlenecks and Common Cause Failures, the system in 
    # Point 3 represents a much more realistic, conservative, and severely 
    # degraded operational scenario compared to the idealized model in Point 2.
    # =========================================================================