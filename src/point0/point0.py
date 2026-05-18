import numpy as np
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # function definitions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def simulate_2oo3_failure_times(n_simulations, failure_rate, rng=None):
    """
        input: - n_simulations: number of simulations to run
               - failure_rate: failure rate of each component (assumed to be the same for all components)
               - rng: optional random number generator (if None, a default generator will be used)
        output: a vector of length n_simulations with the simulated failure times of the subsystem
    """
    if rng is None:
        rng = np.random.default_rng()

    component_failure_times = rng.exponential( 
        scale = 1/failure_rate,     # generates exponentially distributed 1/l*exp{t/l} random numbers, with l = 1/failure_rate
        size = (n_simulations, 3)   # creates a matrix with n_simulations rows and 3 columns, each  row is one simulated system t_a | t_b | t_c
    )

    sorted_failure_times = np.sort(component_failure_times, axis=1) # sorts each row of the component_failure_times matrix in ascending order

    subsystem_failure_times = sorted_failure_times[:, 1] # selects the second column of the sorted_failure_times matrix, which corresponds to the time of the   second failure in each simulation (i.e., the time when the subsystem ABC fails)

    return subsystem_failure_times


def estimate_reliability_and_std(failure_times, time_grid):
    """
    estimate the time-dependent reliability R(t)
    and the standard deviation of the Monte Carlo estimator
    
    input: - failure_times: vector of simulated failure times of the subsystem
           - time_grid: vector of time points at which to estimate the reliability 
    output: - reliability: vector of estimated reliability R(t) at each time point in time_grid
            - reliability_std: vector of estimated standard deviation of the Monte Carlo estimator at each time point in time_grid
    """

    n_simulations = len(failure_times)

    reliability = np.array([ # estimates reliability as the mean of the indicator function (failure_times > t) that the failure time is greater than t, for each time point in time_grid
        np.mean(failure_times > t)
        for t in time_grid
    ])

    reliability_std = np.sqrt(  # since R(t) is a mean of Bernoulli random variables, its standard deviation can be estimated as sqrt(R(t)*(1-R(t))/n_simulations)
        reliability * (1 - reliability) / n_simulations
    )

    return reliability, reliability_std


def analytical_reliability_2oo3(time_grid, failure_rate):
    """
    analytical reliability of a 2oo3 system with identical exponential components
    """
    component_reliability = np.exp(-failure_rate * time_grid) # computes component reliability R(t) = exp(-lambda*t) for each time point in time_grid

    reliability = ( # analytical reliability of a 2oo3 system with identical exponential components given by R_sys(t) = 3*R(t)^2*(1-R(t)) + R(t)^3, where R(t) is the component reliability
        3 * component_reliability**2 * (1 - component_reliability)
        + component_reliability**3
    )

    return reliability

def estimate_mttf_and_std(failure_times):
    """
    estimate the MTTF and the standard deviation of the MTTF estimator.
    """

    n_simulations = len(failure_times)

    mttf = np.mean(failure_times)

    mttf_std = np.std(failure_times, ddof=1) / np.sqrt(n_simulations)

    return mttf, mttf_std




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # main code # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# given data
failure_rate = 1e-3      # h^-1
mission_time = 1000      # h
n_simulations = 1_000  

rng = np.random.default_rng(seed=123) # seed set to 123 for reproducibility

# Monte Carlo simulation
failure_times = simulate_2oo3_failure_times(
    n_simulations=n_simulations,
    failure_rate=failure_rate,
    rng=rng
)

# time grid of 200 points from 0 to mission_time
time_grid = np.linspace(0, mission_time, 200)

# time-dependent reliability and its standard deviation
R_hat, R_hat_std = estimate_reliability_and_std(failure_times, time_grid)

# analytical reference for comparison with the Monte Carlo estimate
R_exact = analytical_reliability_2oo3(time_grid, failure_rate)

# MTTF and uncertainty of the MTTF estimate
mttf_hat, mttf_hat_std = estimate_mttf_and_std(failure_times)


plt.plot( # time-dependent reliability estimate
    time_grid,
    R_hat,
    label=fr"Monte Carlo estimate $\hat R(t)$ with $N={n_simulations}$"
)

plt.fill_between( # uncertainty band
    time_grid,
    R_hat - R_hat_std,
    R_hat + R_hat_std,
    alpha=0.25,
    label=r"$\hat R(t) \pm \sigma_{\hat R(t)}$"
)

plt.plot( # analytical reliability for comparison
    time_grid,
    R_exact,
    "--",
    label=r"analytical reliability $R(t)$"
)

plt.xlabel(r"time $t$ [$\mathrm{h}$]")
plt.ylabel(r"reliability $R(t)$")
plt.title(r"reliability of the 2-out-of-3 subsystem")
plt.grid(True)
plt.legend()
plt.tight_layout()



# plot: standard deviation of the reliability estimator
plt.figure(figsize=(7, 4))

plt.plot(
    time_grid,
    R_hat_std,
    label=r"standard deviation $\sigma_{\hat R(t)}$"
)

plt.xlabel(r"$t$ [$\mathrm{h}$]")
plt.ylabel(r"$\sigma_{\hat R(t)}$")
plt.title(r"uncertainty of the reliability estimate")
plt.grid(True)
plt.legend()
plt.tight_layout()




# print: time-independent quantities
# print: time-independent quantities
R_mission = np.mean(failure_times > mission_time)
R_mission_std = np.sqrt(R_mission * (1 - R_mission) / n_simulations)

print()
print("=" * 78)
print("TIME-INDEPENDENT MONTE CARLO RESULTS")
print("=" * 78)
print(f"{'Quantity':<42} {'Estimate':>16} {'Std. dev.':>16}")
print("-" * 78)

print(
    f"{f'R({mission_time} h)':<42} "
    f"{R_mission:>16.6f} "
    f"{R_mission_std:>16.6e}"
)

print(
    f"{'MTTF [h]':<42} "
    f"{mttf_hat:>16.3f} "
    f"{mttf_hat_std:>16.6f}"
)

print("=" * 78)
print()

plt.show()