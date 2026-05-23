import numpy as np
import matplotlib.pyplot as plt

from src.point2.system import simulate_system_once, _system_is_up
from src.common.sampling import sample_exponential

# =============================================================================
# POINT 2 — Monte Carlo con riparazione
#
# Stime richieste dal testo:
#   i.   R(t)              — time-dependent reliability
#   ii.  MTTF              — Mean Time To First Failure
#   iii. A(t)              — instantaneous availability
#   iv.  Ā                 — average availability su [0, T_M]
#   v.   E[N_rep]          — numero atteso di riparazioni in [0, T_M]
#   vi.  std di tutto quanto sopra
#
# Plot prodotti (richiesti dal template):
#   Figure 1 — R(t) con banda di incertezza
#   Figure 2 — R_std(t)  (standard deviation di R(t))
#   Figure 3 — A(t) con banda di incertezza
#   Figure 4 — A_std(t)  (standard deviation di A(t))
# =============================================================================


# ── Parametri ─────────────────────────────────────────────────────────────────

LAMBDA_1 = 1e-3    # failure rate A, B, C [h⁻¹]
LAMBDA_2 = 2e-3    # failure rate D, E    [h⁻¹]
MU_1     = 1e-2    # repair rate  A, B, C [h⁻¹]
MU_2     = 2e-2    # repair rate  D, E    [h⁻¹]

MISSION_TIME  = 1000
N_SIMULATIONS = 10000
N_GRID        = 200


# ── Simulazione con storia completa ───────────────────────────────────────────

def run_simulation_with_states(n_simulations, lambda_1, lambda_2,
                                mu_1, mu_2, mission_time, time_grid):
    """
    Esegue N simulazioni event-driven e per ogni run registra:
      - t_first_failure : tempo del primo failure di sistema
      - time_up         : tempo totale UP nella missione
      - n_repairs       : numero di riparazioni completate
      - states_matrix   : stato del sistema (1=UP, 0=DOWN) su time_grid

    Parameters
    ----------
    n_simulations : int
    lambda_1, lambda_2 : float  — failure rates [h⁻¹]
    mu_1, mu_2         : float  — repair rates  [h⁻¹]
    mission_time       : float  — orizzonte [h]
    time_grid          : np.ndarray — punti in cui campionare lo stato

    Returns
    -------
    failure_times : np.ndarray (N,)
    time_ups      : np.ndarray (N,)
    n_repairs_arr : np.ndarray (N,)
    states_matrix : np.ndarray (N, len(time_grid))  int8
    """
    n_grid = len(time_grid)
    states_matrix = np.zeros((n_simulations, n_grid), dtype=np.int8)
    failure_times = np.empty(n_simulations)
    time_ups      = np.empty(n_simulations)
    n_repairs_arr = np.empty(n_simulations, dtype=int)

    lmbda_map = {"A": lambda_1, "B": lambda_1, "C": lambda_1,
                 "D": lambda_2, "E": lambda_2}
    mu_map    = {"A": mu_1,     "B": mu_1,     "C": mu_1,
                 "D": mu_2,     "E": mu_2}

    for i in range(n_simulations):

        # Stato iniziale: tutti UP
        states = {c: "UP" for c in ["A", "B", "C", "D", "E"]}
        # Prossimo evento per ogni componente (guasto, dato che parte UP)
        t_next = {c: sample_exponential(lmbda_map[c]) for c in states}

        t               = 0.0
        t_first_failure = float("inf")
        time_up         = 0.0
        n_rep           = 0
        sys_up          = True   # all'inizio tutti UP → sistema UP
        grid_idx        = 0

        while t < mission_time:
            next_comp = min(t_next, key=t_next.get)
            t_event   = t_next[next_comp]
            t_clipped = min(t_event, mission_time)

            # Marca i punti della griglia nell'intervallo corrente
            while grid_idx < n_grid and time_grid[grid_idx] <= t_clipped:
                states_matrix[i, grid_idx] = int(sys_up)
                grid_idx += 1

            # Accumula tempo UP
            if sys_up:
                time_up += t_clipped - t

            t = t_clipped
            if t >= mission_time:
                break

            # Elabora evento
            comp = next_comp
            if states[comp] == "UP":
                # Guasto: schedula riparazione
                states[comp] = "DOWN"
                t_next[comp] = t + sample_exponential(mu_map[comp])
            else:
                # Riparazione completata: schedula prossimo guasto
                states[comp] = "UP"
                n_rep += 1
                t_next[comp] = t + sample_exponential(lmbda_map[comp])

            # Aggiorna stato sistema e registra primo failure
            was_up = sys_up
            sys_up = _system_is_up(states)
            if was_up and not sys_up and t_first_failure == float("inf"):
                t_first_failure = t

        # Riempi punti griglia oltre mission_time
        while grid_idx < n_grid:
            states_matrix[i, grid_idx] = int(sys_up)
            grid_idx += 1

        failure_times[i] = t_first_failure
        time_ups[i]      = time_up
        n_repairs_arr[i] = n_rep

    return failure_times, time_ups, n_repairs_arr, states_matrix


# ── Stime statistiche ─────────────────────────────────────────────────────────

def estimate_reliability(failure_times, time_grid):
    """
    R(t) = P(t_first_failure > t) stimata come media campionaria.
    R_std(t) = deviazione standard della stima MC (formula di Bernoulli).

    Returns
    -------
    R     : np.ndarray — reliability stimata su time_grid
    R_std : np.ndarray — std della stima (non std del campione grezzo)
    """
    n = len(failure_times)
    R     = np.array([np.mean(failure_times > t) for t in time_grid])
    R_std = np.sqrt(R * (1 - R) / n)
    return R, R_std


def estimate_availability_pointwise(states_matrix):
    """
    A(t) = media su tutte le run dello stato in ogni punto della griglia.
    A_std(t) = deviazione standard della stima MC.

    Parameters
    ----------
    states_matrix : np.ndarray (N, n_grid)  int8

    Returns
    -------
    A_t     : np.ndarray (n_grid,)
    A_t_std : np.ndarray (n_grid,)
    """
    n = states_matrix.shape[0]
    A_t     = np.mean(states_matrix, axis=0)
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
    MTTF = media dei tempi di primo failure.
    Run senza failure entro T_M vengono censurate a T_M (stima conservativa).

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
    E[N_rep] = media delle riparazioni completate per run.

    Returns
    -------
    mean_rep : float
    std_rep  : float
    """
    n        = len(n_repairs_arr)
    mean_rep = np.mean(n_repairs_arr)
    std_rep  = np.std(n_repairs_arr, ddof=1) / np.sqrt(n)
    return mean_rep, std_rep


# ── Soluzione analitica — sanity check ───────────────────────────────────────

def analytical_reliability_norepair(time_grid, lambda_1, lambda_2):
    """
    R(t) analitica del sistema SENZA repair (= point 1).
    Usata come lower bound / sanity check: con repair R(t) deve stare sopra.

    R_ABC(t) = 3·e^{-2λ₁t}·(1-e^{-λ₁t}) + e^{-3λ₁t}   [2oo3]
    R_DE(t)  = 1-(1-e^{-λ₂t})²                            [parallelo]
    R_sys(t) = R_ABC(t) · R_DE(t)                          [serie]
    """
    R1    = np.exp(-lambda_1 * time_grid)
    R_ABC = 3 * R1**2 * (1 - R1) + R1**3
    R2    = np.exp(-lambda_2 * time_grid)
    R_DE  = 1 - (1 - R2)**2
    return R_ABC * R_DE


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    np.random.seed(42)

    time_grid = np.linspace(0, MISSION_TIME, N_GRID)

    print(f"Esecuzione {N_SIMULATIONS:,} simulazioni Monte Carlo...")
    failure_times, time_ups, n_repairs_arr, states_matrix = run_simulation_with_states(
        N_SIMULATIONS, LAMBDA_1, LAMBDA_2, MU_1, MU_2, MISSION_TIME, time_grid
    )
    print("Simulazioni completate.")

    # ── Calcolo stime ────────────────────────────────────────────────────────

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