import numpy as np
import matplotlib.pyplot as plt

from src.point3.system import simulate_system_once, system_state_on_grid

# =============================================================================
# POINT 3 — Monte Carlo con coda di repair e shock esterno
#
# Stime richieste (stesse del point 2):
#   i.   R(t)     — time-dependent reliability
#   ii.  MTTF     — Mean Time To First Failure
#   iii. A(t)     — instantaneous availability
#   iv.  Ā        — average availability su [0, T_M]
#   v.   E[N_rep] — numero atteso di riparazioni completate in [0, T_M]
#   vi.  std di tutte le stime sopra
#
# Plot prodotti (richiesti dal template di consegna):
#   Figure 1 — R(t) con banda di incertezza ±1σ
#   Figure 2 — R_std(t)
#   Figure 3 — A(t) con banda di incertezza ±1σ
#   Figure 4 — A_std(t)
# =============================================================================


# ── Parametri ─────────────────────────────────────────────────────────────────

LAMBDA_1     = 1e-3    # failure rate A, B, C [h⁻¹]
LAMBDA_2     = 2e-3    # failure rate D, E    [h⁻¹]
MU_1         = 1e-2    # repair rate  A, B, C [h⁻¹]  → MTTR_ABC = 100 h
MU_2         = 2e-2    # repair rate  D, E    [h⁻¹]  → MTTR_DE  =  50 h
LAMBDA_C     = 2e-3    # shock arrival rate   [h⁻¹]
P_SHOCK      = 0.1     # prob. guasto per componente UP a ogni shock

MISSION_TIME  = 1000
N_SIMULATIONS = 10_000
N_GRID        = 200
SEED          = 42


# ── Simulazione Monte Carlo ───────────────────────────────────────────────────

def run_simulation(n_simulations, lambda_1, lambda_2, mu_1, mu_2,
                   lambda_c, p_shock, mission_time, time_grid):
    """
    Esegue N simulazioni Monte Carlo del sistema point 3.

    Per ogni run raccoglie:
      - t_first_failure : tempo del primo failure di sistema
      - time_up         : tempo totale UP in [0, T_M]
      - n_repairs       : numero di riparazioni completate
      - stato su time_grid (per A(t) istantanea)

    Parameters
    ----------
    n_simulations : int
    lambda_1, lambda_2 : float  — failure rates [h⁻¹]
    mu_1, mu_2         : float  — repair rates  [h⁻¹]
    lambda_c           : float  — shock rate    [h⁻¹]
    p_shock            : float  — prob. guasto per componente UP per shock
    mission_time       : float  — orizzonte     [h]
    time_grid          : np.ndarray — griglia temporale

    Returns
    -------
    failure_times : np.ndarray (N,)
    time_ups      : np.ndarray (N,)
    n_repairs_arr : np.ndarray (N,) int
    states_matrix : np.ndarray (N, len(time_grid)) int8
                    1 = sistema UP, 0 = sistema DOWN
    """
    n_grid        = len(time_grid)
    failure_times = np.empty(n_simulations)
    time_ups      = np.empty(n_simulations)
    n_repairs_arr = np.empty(n_simulations, dtype=int)
    states_matrix = np.zeros((n_simulations, n_grid), dtype=np.int8)

    for i in range(n_simulations):
        res = simulate_system_once(
            lambda_1, lambda_2, mu_1, mu_2,
            lambda_c, p_shock, mission_time
        )
        failure_times[i] = res["t_first_failure"]
        time_ups[i]      = res["time_up"]
        n_repairs_arr[i] = res["n_repairs"]
        # Ricostruisce lo stato su time_grid dalla lista sparsa di transizioni
        states_matrix[i] = system_state_on_grid(res["state_at"], time_grid)

    return failure_times, time_ups, n_repairs_arr, states_matrix


# ── Stime statistiche ─────────────────────────────────────────────────────────

def estimate_reliability(failure_times, time_grid):
    """
    R(t) = P(t_first_failure > t), stimata come media campionaria.

    R_std(t) = deviazione standard della stima MC:
               sqrt(R(t) * (1-R(t)) / N)
               derivata dalla varianza di una variabile di Bernoulli.

    Parameters
    ----------
    failure_times : np.ndarray (N,)
    time_grid     : np.ndarray (n_grid,)

    Returns
    -------
    R     : np.ndarray (n_grid,)
    R_std : np.ndarray (n_grid,)
    """
    n     = len(failure_times)
    R     = np.array([np.mean(failure_times > t) for t in time_grid])
    R_std = np.sqrt(R * (1 - R) / n)
    return R, R_std


def estimate_availability_pointwise(states_matrix):
    """
    A(t) = media su tutte le run dello stato binario in ogni punto della griglia.
    A_std(t) = deviazione standard della stima MC (formula di Bernoulli).

    Parameters
    ----------
    states_matrix : np.ndarray (N, n_grid) int8

    Returns
    -------
    A_t     : np.ndarray (n_grid,)
    A_t_std : np.ndarray (n_grid,)
    """
    n       = states_matrix.shape[0]
    A_t     = np.mean(states_matrix, axis=0)
    A_t_std = np.sqrt(A_t * (1 - A_t) / n)
    return A_t, A_t_std


def estimate_average_availability(time_ups, mission_time, n_simulations):
    """
    Ā = E[time_up] / mission_time — average availability su [0, T_M].

    La std della stima è la std della media campionaria di time_up / T_M.

    Returns
    -------
    A_avg     : float
    A_avg_std : float
    """
    A_avg     = np.mean(time_ups) / mission_time
    A_avg_std = (np.std(time_ups, ddof=1) / np.sqrt(n_simulations)) / mission_time
    return A_avg, A_avg_std


def estimate_mttf(failure_times, mission_time):
    """
    MTTF = media dei tempi di primo failure del sistema.

    Le run in cui il sistema non ha mai fallito entro T_M vengono censurate
    a T_M: questo produce una stima per difetto (lower bound) del MTTF reale.
    La censura è necessaria perché la simulazione termina a mission_time.

    Returns
    -------
    mttf     : float
    mttf_std : float
    """
    n           = len(failure_times)
    ft_censored = np.where(np.isinf(failure_times), mission_time, failure_times)
    mttf        = np.mean(ft_censored)
    mttf_std    = np.std(ft_censored, ddof=1) / np.sqrt(n)
    return mttf, mttf_std


def estimate_n_repairs(n_repairs_arr):
    """
    E[N_rep] = numero medio di riparazioni completate per run.

    Returns
    -------
    mean_rep : float
    std_rep  : float
    """
    n        = len(n_repairs_arr)
    mean_rep = np.mean(n_repairs_arr)
    std_rep  = np.std(n_repairs_arr, ddof=1) / np.sqrt(n)
    return mean_rep, std_rep


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Seed fisso per riproducibilità, coerente con gli altri point del progetto
    np.random.seed(SEED)

    time_grid = np.linspace(0, MISSION_TIME, N_GRID)

    print(f"Esecuzione {N_SIMULATIONS:,} simulazioni Monte Carlo (Point 3)...")
    failure_times, time_ups, n_repairs_arr, states_matrix = run_simulation(
        N_SIMULATIONS, LAMBDA_1, LAMBDA_2, MU_1, MU_2,
        LAMBDA_C, P_SHOCK, MISSION_TIME, time_grid
    )
    print("Simulazioni completate.")

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

    # ── Figure 1: R(t) con banda di incertezza ────────────────────────────────
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