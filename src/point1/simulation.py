import numpy as np
import matplotlib.pyplot as plt
#su questo file si farà MonteCarlo + gli output
#ripete tante simulazioni → raccoglie risultati → calcola statistiche → fa grafici

from src.point1.system import simulate_system_once
#Definisco la funzione run_simulation che esegue N simulazioni Monte Carlo.
#Ad ogni iterazione viene chiamata la funzione simulate_2oo3_once (definita in system.py),
#che genera tre tempi di guasto casuali (T_A, T_B, T_C), li ordina e restituisce
#il secondo tempo (tempo di guasto del sistema 2oo3).
np.random.seed(42)  #aggiunto seed(42) per riproducibilità
#Questo valore viene salvato nella variabile t e aggiunto alla lista results.
#Alla fine, results contiene tutti i tempi di guasto del sistema per le N simulazioni.
def run_simulation(n_simulations, lambda_1, lambda_2):
    """
    Esegue N simulazioni Monte Carlo del sistema completo.
    """

    results = []

    for _ in range(n_simulations):
        t = simulate_system_once(lambda_1, lambda_2)
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





# ================= MAIN =================

if __name__ == "__main__":

    lambda_1 = 1e-3      # A, B, C
    lambda_2 = 2e-3      # D, E

    mission_time = 1000
    n_simulations = 1_000

    failure_times = run_simulation(n_simulations, lambda_1, lambda_2)

    # griglia temporale
    time_grid = np.linspace(0, mission_time, 200)

    # R(t)
    R, R_std = estimate_reliability_and_std(failure_times, time_grid)

    
    

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

    

    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.title("Full system reliability")
    plt.legend()
    plt.grid()
    plt.show()

    # ================= PRINT =================

    R_mission = np.mean(failure_times > mission_time)
    R_mission_std = np.sqrt(R_mission * (1 - R_mission) / n_simulations)

    print("\n===== RESULTS =====")
    print(f"R({mission_time}) = {R_mission:.4f} ± {R_mission_std:.4e}")
    print(f"MTTF = {mttf:.2f} ± {mttf_std:.2f}")


    # ================= INTERPRETAZIONE RISULTATI =================
# Nel point1 il sistema è composto da:
# - sottosistema ABC (2oo3)
# - sottosistema DE in serie
#
# L'aggiunta del blocco DE in serie rende il sistema meno affidabile,
# perché un sistema in serie fallisce quando fallisce uno qualsiasi dei blocchi.
#
# Per questo motivo ci aspettiamo:
# - R(t) più bassa rispetto al point0
# - MTTF più basso (il sistema si rompe prima)
#
# I risultati ottenuti sono coerenti con questa logica.

# ================= INTERPRETAZIONE GRAFICO =================
# La curva blu rappresenta la stima Monte Carlo del sistema completo (ABC + DE).
# Questa curva decresce più velocemente perché il sistema è meno affidabile.
#
# La curva arancione rappresenta la soluzione analitica del solo sistema 2oo3 (point0).
#
# ATTENZIONE:
# Le due curve NON devono coincidere, perché descrivono sistemi diversi:
# - Monte Carlo: sistema completo (ABC + DE)
# - Analitica: solo ABC (2oo3)
#
# La differenza tra le curve è quindi corretta e attesa.

# R(t) rappresenta la probabilità che il sistema sia ancora funzionante al tempo t.
# Viene stimata come:
# numero di simulazioni in cui il sistema sopravvive oltre t / numero totale di simulazioni
# COMMENTO SU MTTF (mettilo vicino alla funzione o al print)
# MTTF (Mean Time To Failure) rappresenta il tempo medio di guasto del sistema.
# È la media di tutti i tempi di failure ottenuti dalle simulazioni Monte Carlo.
#
# Nel point1 ci aspettiamo un MTTF più basso rispetto al point0,
# perché il sistema include un blocco in serie aggiuntivo (DE).
#COMMENTO SULL’INCERTEZZA
# R_std rappresenta l'incertezza statistica della stima Monte Carlo.
# Aumentando il numero di simulazioni, questa incertezza diminuisce.