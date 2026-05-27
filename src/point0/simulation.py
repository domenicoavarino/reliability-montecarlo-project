import numpy as np
import random
import matplotlib.pyplot as plt
#su questo file si farà MonteCarlo + gli output
#ripete tante simulazioni → raccoglie risultati → calcola statistiche → fa grafici

from src.point0.system import simulate_2oo3_once
#Definisco la funzione run_simulation che esegue N simulazioni Monte Carlo.
#Ad ogni iterazione viene chiamata la funzione simulate_2oo3_once (definita in system.py),
#che genera tre tempi di guasto casuali (T_A, T_B, T_C), li ordina e restituisce
#il secondo tempo (tempo di guasto del sistema 2oo3).
#Questo valore viene salvato nella variabile t e aggiunto alla lista results.
#Alla fine, results contiene tutti i tempi di guasto del sistema per le N simulazioni.
def run_simulation(n_simulations, lmbda):
    """
    Esegue N simulazioni Monte Carlo
    """
    results = [] #lista dove salverai i time failure, ovvero tutti i tempi di guasto del sistema.
#in results fondamentalmente vado a mettere tutti i tempi calcolati dalla funzione "simulate_2oo3_once"definita in system.py. quella funzione genera 3 tempi e outputta il secondo (che è quello di failure). quindi in results vado a fare la lista con tutti i time failure
    for _ in range(n_simulations):
        t = simulate_2oo3_once(lmbda)
        results.append(t)

    return np.array(results)


def estimate_reliability_and_std(failure_times, time_grid): 
    """
    Stima R(t) e la deviazione standard
    """
    n = len(failure_times)

    R = np.array([
        np.mean(failure_times > t) # per ogni failure times > di t, restituisce true o false
        for t in time_grid
    ])
#time grid è una lista in cui vuoi vedere quanto il sistema è ancora vivo
    R_std = np.sqrt(R * (1 - R) / n) #è l’incertezza del Monte Carlo, più simulazioni → meno errore

    return R, R_std
#Esempio: [1200, 800, 500] > 1000 → [True, False, False]  
# np.mean(...)

# True = 1
# False = 0

#Quindi:

#(1 + 0 + 0) / 3 = 0.33
#R(t) = probabilità che il sistema sopravviva oltre t
def estimate_mttf_and_std(failure_times):
    """
    Stima MTTF e deviazione standard
    """
    n = len(failure_times)

    mttf = np.mean(failure_times)
    std = np.std(failure_times, ddof=1) / np.sqrt(n)

    return mttf, std


def analytical_reliability_2oo3(time_grid, lmbda):
    """
    Soluzione teorica per confronto
    """
    R_comp = np.exp(-lmbda * time_grid)

    R_sys = (
        3 * R_comp**2 * (1 - R_comp)
        + R_comp**3
    )

    return R_sys


# ================= MAIN =================

if __name__ == "__main__":
    np.random.seed(42)  #aggiunto seed(42) per riproducibilità
    lmbda = 1e-3
    mission_time = 1000
    n_simulations = 100000

    # Monte Carlo
    failure_times = run_simulation(n_simulations, lmbda)

    # griglia temporale
    time_grid = np.linspace(0, mission_time, 200)

    # R(t)
    R, R_std = estimate_reliability_and_std(failure_times, time_grid)

    # confronto teorico
    R_exact = analytical_reliability_2oo3(time_grid, lmbda)

    # MTTF
    mttf, mttf_std = estimate_mttf_and_std(failure_times)

    # ================= PLOT =================
#in questo grafico ho stimato R(t) con Monte Carlo e verificato che coincide con la soluzione analitica, validando il modello.
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

    # Calcolo di R_mission e della sua incertezza al tempo T_M
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