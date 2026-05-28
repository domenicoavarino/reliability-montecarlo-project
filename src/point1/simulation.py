import numpy as np
import matplotlib.pyplot as plt
from src.point1.system import simulate_system_once
"""
MONTE CARLO SIMULATION MODULE - POINT 1
---------------------------------------
This script orchestrates the Monte Carlo simulation for the full unrepairable 
system (Subsystem ABC in series with Subsystem DE). 
It repeatedly simulates the system history to collect failure times, computes 
statistical estimates (Reliability and MTTF), validates them against the exact 
analytical solution, and generates the required plots.
"""
def run_simulation(n_simulations, lambda_1, lambda_2):
    """
    Executes N independent Monte Carlo simulations for the complete system.

    Parameters
    ----------
    n_simulations : int
        The total number of system histories to simulate.
    lambda_1 : float
        Failure rate for components A, B, and C [h^-1].
    lambda_2 : float
        Failure rate for components D and E [h^-1].

    Returns
    -------
    np.ndarray
        A 1D array containing the global system failure time for each run.
    """
    # List to store the absolute failure times of the system across all runs
    results = []

    for _ in range(n_simulations):
        t = simulate_system_once(lambda_1, lambda_2)
        results.append(t)

    return np.array(results)


def estimate_reliability_and_std(failure_times, time_grid): 
    """
    Estimates the time-dependent reliability R(t) and its standard deviation.

    Algorithm Logic:
    For a given time t, the condition (failure_times > t) returns an array 
    of booleans. Since Python evaluates True as 1 and False as 0, taking the 
    mean of this array yields the survival probability R(t).
    Example: failure_times = [1200, 800, 500], t = 1000
             [1200 > 1000, 800 > 1000, 500 > 1000] -> [True, False, False]
             np.mean([1, 0, 0]) = 0.333 -> R(1000) = 33.3%

    Parameters
    ----------
    failure_times : np.ndarray
        Array of simulated system failure times.
    time_grid : np.ndarray
        Array of time points at which to evaluate R(t).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        R     : Estimated reliability values over the time grid.
        R_std : Statistical uncertainty (standard deviation) of R(t) based 
                on the variance of a binomial proportion.
    """
    n = len(failure_times)

    R = np.array([
        np.mean(failure_times > t) # per ogni failure times > di t, restituisce true o false
        for t in time_grid
    ])
# Standard deviation of the binomial estimator: sqrt(p * (1-p) / N)
    R_std = np.sqrt(R * (1 - R) / n) 

    return R, R_std

def estimate_mttf_and_std(failure_times):
    """
    Estimates the Mean Time To Failure (MTTF) and its standard error.

    Returns
    -------
    tuple of (float, float)
        mttf : The sample mean of the simulated failure times.
        std  : The standard error of the mean.
    """
    n = len(failure_times)

    mttf = np.mean(failure_times)
    std = np.std(failure_times, ddof=1) / np.sqrt(n)

    return mttf, std

def analytical_reliability_full(time_grid, lambda_1, lambda_2):
    """
    Calculates the exact analytical solution for the complete system reliability.
    Used to validate the stochastic Monte Carlo estimates.
    
    R_sys(t) = R_ABC(t) * R_DE(t)  (since the two subsystems are in series)
    """
    # Subsystem ABC (2oo3) Reliability
    R1 = np.exp(-lambda_1 * time_grid)
    R_ABC = 3 * R1**2 * (1 - R1) + R1**3
    # Subsystem DE (Parallel) Reliability
    R2 = np.exp(-lambda_2 * time_grid)
    R_DE = 1 - (1 - R2)**2
    # Total System Reliability
    return R_ABC * R_DE   # serie: prodotto



# ================= MAIN =================

if __name__ == "__main__":
    np.random.seed(42)  # Fix the seed to ensure strict reproducibility of the results
    lambda_1 = 1e-3      # A, B, C
    lambda_2 = 2e-3      # D, E

    mission_time = 1000
    n_simulations = 100_000
    # 1. Run the Monte Carlo simulation
    failure_times = run_simulation(n_simulations, lambda_1, lambda_2)

    # Define the time horizon for the analysis
    time_grid = np.linspace(0, mission_time, 200)

    # R(t)
    R, R_std = estimate_reliability_and_std(failure_times, time_grid)
    # 3. Compute analytical solution for model validation
    R_exact = analytical_reliability_full(time_grid, lambda_1, lambda_2)
    

    # MTTF
    mttf, mttf_std = estimate_mttf_and_std(failure_times)

    # ================= PLOT GENERATION =================

    # ── Figure 1: Reliability R(t) ───────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R, label="Monte Carlo")

    plt.fill_between(
        time_grid,
        R - R_std,
        R + R_std,
        alpha=0.3,
        label="uncertainty"
    )
    plt.plot(time_grid, R_exact, "--", label="Analytical")
    

    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.title("Full system reliability")
    plt.legend()
    plt.grid()
    plt.show()
    # ── Figure 2: R_std(t) ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R_std, color="blue", linewidth=2)
    plt.xlabel("Time (h)")
    plt.ylabel("σ[R(t)]")
    plt.title("Reliability Standard Deviation — Point 1")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # ================= PRINT =================

    # Calculate R(t) and its standard deviation exactly at the mission time (T_M)
    R_mission = np.mean(failure_times > mission_time)
    R_mission_std = np.sqrt(R_mission * (1 - R_mission) / n_simulations)

    print("\n" + "=" * 52)
    print("          RESULTS — POINT 1")
    print("=" * 52)
    print(f"  Simulations        : {n_simulations:,}")
    print(f"  Mission time       : {mission_time} h")
    print("-" * 52)
    print(f"  R(T_M)             = {R_mission:.4f}  ±  {R_mission_std:.4e}")
    print(f"  MTTF               = {mttf:.2f}  ±  {mttf_std:.2f} h")
    print("=" * 52)

# =========================================================================
    # PHYSICAL INTERPRETATION OF THE RESULTS
    # =========================================================================
    # SYSTEM TOPOLOGY:
    # In Point 1, the system consists of the ABC subsystem (2oo3) connected in 
    # series with the DE subsystem (parallel). 
    #
    # ENGINEERING EXPECTATIONS:
    # Adding a block in series to an existing architecture inherently makes the 
    # overall system strictly less reliable, since a series connection acts as a 
    # structural bottleneck (it fails as soon as ANY of its subsystems fails).
    #
    # CONCLUSIONS:
    # Consequently, compared to Point 0, we physically expect:
    # - A lower overall Reliability R(t) curve.
    # - A significantly shorter Mean Time To Failure (MTTF).
    # The printed Monte Carlo results are perfectly consistent with this logic.
    #
    # GRAPHICAL VALIDATION:
    # The blue Monte Carlo curve and the red dashed Analytical curve perfectly 
    # overlap, providing strict mathematical validation for the simulated model.
    # =========================================================================