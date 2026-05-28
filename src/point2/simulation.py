import numpy as np
import matplotlib.pyplot as plt
from src.point2.system import simulate_system_once, _system_is_up
from src.common.sampling import sample_exponential
"""
MONTE CARLO SIMULATION MODULE - POINT 2
---------------------------------------
This script orchestrates the Monte Carlo simulation for the repairable system.
It utilizes a Discrete Event Simulation (DES) approach to model both failure 
and repair processes. 

It computes and plots the following reliability and maintainability metrics:
  i.   R(t)       — Time-dependent Reliability
  ii.  MTTF       — Mean Time To First Failure (with right-censoring)
  iii. A(t)       — Instantaneous Availability
  iv.  Ā          — Average Availability over [0, T_M]
  v.   E[N_rep]   — Expected number of completed repairs in [0, T_M]
  vi.  Statistical uncertainties (standard deviations) for all estimates.
"""

# ── System Parameters ─────────────────────────────────────────────────────────
LAMBDA_1 = 1e-3    # failure rate A, B, C [h⁻¹]
LAMBDA_2 = 2e-3    # failure rate D, E    [h⁻¹]
MU_1     = 1e-2    # repair rate  A, B, C [h⁻¹]
MU_2     = 2e-2    # repair rate  D, E    [h⁻¹]

MISSION_TIME  = 1000
N_SIMULATIONS = 100000
N_GRID        = 200


# ── Full History Simulation ───────────────────────────────────────────

def run_simulation_with_states(n_simulations, lambda_1, lambda_2,
                                mu_1, mu_2, mission_time, time_grid):
    """
    Executes N independent event-driven Monte Carlo simulations. 
    Unlike static simulations, this records the continuous operational history 
    of the system over the time grid to compute instantaneous availability.

    Parameters
    ----------
    n_simulations : int
    lambda_1, lambda_2 : float  — Failure rates [h^-1]
    mu_1, mu_2         : float  — Repair rates [h^-1]
    mission_time       : float  — Time horizon [h]
    time_grid          : np.ndarray — Array of discrete times to sample the state

    Returns
    -------
    tuple containing:
        failure_times : np.ndarray (N,) — Absolute time of FIRST system failure.
        time_ups      : np.ndarray (N,) — Cumulative UP time per run.
        n_repairs_arr : np.ndarray (N,) — Total completed repairs per run.
        states_matrix : np.ndarray (N, len(time_grid)) — Binary system state 
                        (1=UP, 0=DOWN) sampled at each point in time_grid.
    """
    n_grid = len(time_grid)
    # Pre-allocate memory for performance
    states_matrix = np.zeros((n_simulations, n_grid), dtype=np.int8)
    failure_times = np.empty(n_simulations)
    time_ups      = np.empty(n_simulations)
    n_repairs_arr = np.empty(n_simulations, dtype=int)

    lmbda_map = {"A": lambda_1, "B": lambda_1, "C": lambda_1,
                 "D": lambda_2, "E": lambda_2}
    mu_map    = {"A": mu_1,     "B": mu_1,     "C": mu_1,
                 "D": mu_2,     "E": mu_2}

    for i in range(n_simulations):

        # Initial state: all components are working perfectly
        states = {c: "UP" for c in ["A", "B", "C", "D", "E"]}
        # Schedule the first spontaneous failure for each component
        t_next = {c: sample_exponential(lmbda_map[c]) for c in states}

        t               = 0.0
        t_first_failure = float("inf")
        time_up         = 0.0
        n_rep           = 0
        sys_up          = True   # System starts UP
        grid_idx        = 0
        # Event-driven loop
        while t < mission_time:
            # Jump to the next chronological event
            next_comp = min(t_next, key=t_next.get)
            t_event   = t_next[next_comp]
            t_clipped = min(t_event, mission_time)

            # Sample and record the system state onto the time_grid array
            # for all grid points passed during this time jump
            while grid_idx < n_grid and time_grid[grid_idx] <= t_clipped:
                states_matrix[i, grid_idx] = int(sys_up)
                grid_idx += 1

            # Accumulate Total UP time
            if sys_up:
                time_up += t_clipped - t

            t = t_clipped
            if t >= mission_time:
                break

            # Process the specific component event
            comp = next_comp
            if states[comp] == "UP":
                # Event: Failure -> Schedule the immediate repair process
                states[comp] = "DOWN"
                t_next[comp] = t + sample_exponential(mu_map[comp])
            else:
                # Event: Repair complete -> Schedule the next spontaneous failure
                states[comp] = "UP"
                n_rep += 1
                t_next[comp] = t + sample_exponential(lmbda_map[comp])

            # Update the global system state
            was_up = sys_up
            sys_up = _system_is_up(states)
            # Catch the very first global failure (UP -> DOWN transition)
            if was_up and not sys_up and t_first_failure == float("inf"):
                t_first_failure = t

        # Fill any remaining grid points beyond the last event
        while grid_idx < n_grid:
            states_matrix[i, grid_idx] = int(sys_up)
            grid_idx += 1
        # Store the metrics for this specific Monte Carlo history
        failure_times[i] = t_first_failure
        time_ups[i]      = time_up
        n_repairs_arr[i] = n_rep

    return failure_times, time_ups, n_repairs_arr, states_matrix


# ── Statistical Estimation Functions ─────────────────────────────────────────────────────────

def estimate_reliability(failure_times, time_grid):
    """
    Estimates R(t) = P(T_first_failure > t) and its standard deviation.
    """
    n = len(failure_times)
    R     = np.array([np.mean(failure_times > t) for t in time_grid])
    R_std = np.sqrt(R * (1 - R) / n)
    return R, R_std


def estimate_availability_pointwise(states_matrix):
    """
    Estimates Instantaneous Availability A(t).
    
    A(t) is the probability that the system is functioning at time t, regardless 
    of past failures. It is computed as the ensemble average (across all runs) 
    of the binary system state (1=UP, 0=DOWN) at each specific grid point.
    """
    n = states_matrix.shape[0]
    # Calculate the mean across the columns (axis=0) of the states matrix
    A_t     = np.mean(states_matrix, axis=0)
    # Standard deviation of the binomial estimator
    A_t_std = np.sqrt(A_t * (1 - A_t) / n)
    return A_t, A_t_std


def estimate_average_availability(time_ups, mission_time, n_simulations):
    """
    Ā = E[time_up] / mission_time  — average availability su [0, T_M].

    Returns
    -------
    A_avg     : float
    A_avg_std : float  — std della stima MC
    """
    A_avg     = np.mean(time_ups) / mission_time
    A_avg_std = (np.std(time_ups, ddof=1) / np.sqrt(n_simulations)) / mission_time
    return A_avg, A_avg_std


def estimate_mttf(failure_times, mission_time):
    """
    Estimates the Mean Time To First Failure (MTTF).

    RIGHT-CENSORING LOGIC:
    Since the simulation strictly stops at T_M, runs where the system never 
    failed have a recorded failure time of infinity. We apply "right-censoring" 
    by capping these infinite values at T_M. This provides a mathematically 
    rigorous, conservative lower-bound estimate of the true MTTF.
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


# ── Soluzione analitica — sanity check ───────────────────────────────────────

def analytical_reliability_norepair(time_grid, lambda_1, lambda_2):
    """
    Sanity Check: Exact analytical R(t) of the system WITHOUT repairs (from Point 1).
    A correctly functioning repairable system MUST exhibit a reliability curve 
    strictly greater than or equal to this non-repairable lower bound.
    """
    R1    = np.exp(-lambda_1 * time_grid)
    R_ABC = 3 * R1**2 * (1 - R1) + R1**3
    R2    = np.exp(-lambda_2 * time_grid)
    R_DE  = 1 - (1 - R2)**2
    return R_ABC * R_DE


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)   # Fix the seed for reproducibility
    time_grid = np.linspace(0, MISSION_TIME, N_GRID)

    print(f"Esecuzione {N_SIMULATIONS:,} simulazioni Monte Carlo...")
    failure_times, time_ups, n_repairs_arr, states_matrix = run_simulation_with_states(
        N_SIMULATIONS, LAMBDA_1, LAMBDA_2, MU_1, MU_2, MISSION_TIME, time_grid
    )
    print("Simulazioni completate.")

    # ── Calculate Estimates ────────────────────────────────────────────────────────

    # i. R(t) e std
    R, R_std = estimate_reliability(failure_times, time_grid)
    R_norepair = analytical_reliability_norepair(time_grid, LAMBDA_1, LAMBDA_2)

    # ii. MTTF
    mttf, mttf_std = estimate_mttf(failure_times, MISSION_TIME)

    # iii. A(t) e std
    A_t, A_t_std = estimate_availability_pointwise(states_matrix)

    # iv. Ā
    A_avg, A_avg_std = estimate_average_availability(time_ups, MISSION_TIME, N_SIMULATIONS)

    # v. E[N_rep]
    mean_rep, std_rep = estimate_n_repairs(n_repairs_arr)

    # ── Figure 1: R(t) ───────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R, color="tab:blue", label="R(t) Monte Carlo (with repair)")
    plt.fill_between(time_grid, R - R_std, R + R_std,
                     alpha=0.3, color="tab:blue", label="Uncertainty ±1σ")
    plt.plot(time_grid, R_norepair, "--", color="gray",
             label="R(t) analytical — no repair (sanity check)")
    plt.xlabel("Time [h]")
    plt.ylabel("R(t)")
    plt.title("Time-Dependent Reliability — Point 2")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # ── Figure 2: R_std(t) ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R_std, color="tab:blue")
    plt.xlabel("Time [h]")
    plt.ylabel("σ[R(t)]")
    plt.title("Reliability Standard Deviation — Point 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── Figure 3: A(t) ───────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, A_t, color="tab:orange", label="A(t) Monte Carlo")
    plt.fill_between(time_grid, A_t - A_t_std, A_t + A_t_std,
                     alpha=0.3, color="tab:orange", label="Uncertainty ±1σ")
    plt.axhline(A_avg, color="tab:red", linestyle=":",
                label=f"Ā (avg) = {A_avg:.4f} ± {A_avg_std:.4e}")
    plt.xlabel("Time [h]")
    plt.ylabel("A(t)")
    plt.title("Instantaneous Availability — Point 2")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # ── Figure 4: A_std(t) ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, A_t_std, color="tab:orange")
    plt.xlabel("Time [h]")
    plt.ylabel("σ[A(t)]")
    plt.title("Availability Standard Deviation — Point 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── Print risultati ───────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("          RESULTS — POINT 2")
    print("=" * 52)
    print(f"  Simulations        : {N_SIMULATIONS:,}")
    print(f"  Mission time       : {MISSION_TIME} h")
    print("-" * 52)
    print(f"  R(T_M)             = {R[-1]:.4f}  ±  {R_std[-1]:.4e}")
    print(f"  MTTF               = {mttf:.2f}  ±  {mttf_std:.2f} h")
    print(f"  A(T_M)             = {A_t[-1]:.4f}  ±  {A_t_std[-1]:.4e}")
    print(f"  Ā  [0, T_M]        = {A_avg:.4f}  ±  {A_avg_std:.4e}")
    print(f"  E[N_repairs]       = {mean_rep:.2f}  ±  {std_rep:.4f}")
    print("=" * 52)

    # =========================================================================
    # PHYSICAL INTERPRETATION OF THE RESULTS
    # =========================================================================
    # 1. R(t) vs R_norepair(t): 
    # The introduction of maintenance improves system survival. As expected, 
    # the blue Monte Carlo curve R(t) strictly dominates the grey dashed 
    # analytical lower bound curve (system without repairs).
    #
    # 2. A(t) vs R(t): 
    # Reliability R(t) requires continuous survival (no failures allowed). 
    # Availability A(t) allows the system to fail and be repaired. Therefore, 
    # the probability of the system being operational (A(t)) is significantly 
    # higher than the probability of it never having failed (R(t)). A(t) curve 
    # stabilizes around its steady-state asymptotic value, while R(t) continues 
    # to decay monotonically towards 0.
    # 3. MTTF and Right-Censoring:
    # Since the simulation is strictly truncated at the mission time T_M = 1000 h, 
    # some system histories will never experience a global failure. 
    # Excluding these "survival" runs from the MTTF calculation would create a 
    # severe selection bias, mathematically underestimating the true reliability 
    # (averaging only the "unlucky" runs). By applying "right-censoring" 
    # (capping the survival times at T_M), we correctly account for these 
    # successful runs, providing a mathematically rigorous and conservative 
    # lower-bound estimate of the true Mean Time To First Failure.
    # =========================================================================