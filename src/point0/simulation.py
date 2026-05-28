import numpy as np
import matplotlib.pyplot as plt
from src.point0.system import simulate_2oo3_once
"""
MONTE CARLO SIMULATION MODULE - POINT 0
---------------------------------------
This script orchestrates the Monte Carlo simulation for the 2-out-of-3 (2oo3)
subsystem without repair. It repeatedly simulates the system history to collect 
failure times, computes statistical estimates (Reliability and MTTF) along with 
their respective uncertainties, and generates the required visual plots.
"""
def run_simulation(n_simulations, lmbda):
    """
    Executes N independent Monte Carlo simulations for the 2oo3 subsystem.

    Parameters
    ----------
    n_simulations : int
        The total number of system histories to simulate.
    lmbda : float
        The constant failure rate (λ) of the components [h^-1].

    Returns
    -------
    np.ndarray
        A 1D array containing the system failure time for each simulation run.
    """
    results = [] # List to store the absolute failure times of the system across all runs
    # Loop to generate N independent system failure times
    for _ in range(n_simulations):
        t = simulate_2oo3_once(lmbda)
        results.append(t)

    return np.array(results)


def estimate_reliability_and_std(failure_times, time_grid): 
    """
    Estimates the time-dependent reliability R(t) and its standard deviation.

    R(t) is estimated as the fraction of simulations where the system survives 
    beyond time t. Since the survival of the system at time t is a Bernoulli 
    trial (1 = survived, 0 = failed), the uncertainty is computed using the 
    standard deviation of a binomial proportion.

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
        R_std : Statistical uncertainty (standard deviation) of R(t).
    """
    n = len(failure_times)
    # Calculate R(t) for each time step: 
    # (failure_times > t) returns a boolean array. np.mean() treats True as 1 
    # and False as 0, effectively calculating the survival probability.
    R = np.array([
        np.mean(failure_times > t) # per ogni failure times > di t, restituisce true o false
        for t in time_grid
    ])
# Standard deviation of the binomial estimator: sqrt(p * (1-p) / N)
    R_std = np.sqrt(R * (1 - R) / n) 

    return R, R_std
#Exemple: [1200, 800, 500] > 1000 → [True, False, False]  
# np.mean(...)

# True = 1
# False = 0

#so:

#(1 + 0 + 0) / 3 = 0.33
def estimate_mttf_and_std(failure_times):
    """
    Estimates the Mean Time To Failure (MTTF) and its standard error.

    Returns
    -------
    tuple of (float, float)
        mttf : The sample mean of the simulated failure times.
        std  : The standard error of the mean (sample standard deviation / sqrt(N)).
    """
    n = len(failure_times)

    mttf = np.mean(failure_times)
    # Use ddof=1 to calculate the unbiased sample standard deviation
    std = np.std(failure_times, ddof=1) / np.sqrt(n)

    return mttf, std


def analytical_reliability_2oo3(time_grid, lmbda):
    """
    Calculates the exact analytical solution for the 2oo3 subsystem reliability.
    Used to validate the Monte Carlo estimates.
    
    R_sys(t) = 3 * R_comp(t)^2 * (1 - R_comp(t)) + R_comp(t)^3
    """
    # Reliability of a single exponential component
    R_comp = np.exp(-lmbda * time_grid)
    # Reliability of the 2-out-of-3 architecture
    R_sys = (
        3 * R_comp**2 * (1 - R_comp)
        + R_comp**3
    )

    return R_sys


# ================= MAIN =================

if __name__ == "__main__":
    np.random.seed(42)  # Fix the seed to ensure strict reproducibility of the results
    # System parameters
    lmbda = 1e-3
    mission_time = 1000
    n_simulations = 100000

    # Monte Carlo
    failure_times = run_simulation(n_simulations, lmbda)

    # Define the time horizon for the analysis
    time_grid = np.linspace(0, mission_time, 200)

    # 2. Compute statistical estimates (Monte Carlo)
    R, R_std = estimate_reliability_and_std(failure_times, time_grid)

    # 3. Compute analytical solution for model validation
    R_exact = analytical_reliability_2oo3(time_grid, lmbda)

    # MTTF
    mttf, mttf_std = estimate_mttf_and_std(failure_times)

    # ================= PLOT =================
# ── Figure 1: Reliability R(t) ───────────────────────────────────────────
    # This plot overlays the Monte Carlo estimate with the analytical solution
    # to visually validate the correctness of the stochastic model.
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R, label="Monte Carlo")
    # Plot the +/- 1 standard deviation uncertainty band
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
    plt.title("2oo3 reliability")
    plt.legend()
    plt.grid()
    plt.show()
    # ── Figure 2: R_std(t) ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, R_std, color="blue", linewidth=2)
    plt.xlabel("Time (h)")
    plt.ylabel("σ[R(t)]")
    plt.title("Reliability Standard Deviation — Point 0")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()
 # ================= PRINT =================

    # Calculate R(t) and its standard deviation exactly at the mission time (T_M)
    R_mission = np.mean(failure_times > mission_time)
    R_mission_std = np.sqrt(R_mission * (1 - R_mission) / n_simulations)

    print("\n" + "=" * 52)
    print("          RESULTS — POINT 0") 
    print("=" * 52)
    print(f"  Simulations        : {n_simulations:,}")
    print(f"  Mission time       : {mission_time} h")
    print("-" * 52)
    print(f"  R(T_M)             = {R_mission:.4f}  ±  {R_mission_std:.4e}")
    print(f"  MTTF               = {mttf:.2f}  ±  {mttf_std:.2f} h")
    print("=" * 52)