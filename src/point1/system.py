from src.common.sampling import sample_exponential

# questo file descrive la logica del sistema al point1

def simulate_system_once(lmbda):
    """
    Simula UNA volta il sistema completo:

    - Sottosistema ABC (2oo3)
    - Sottosistema DE (serie)

    ritorna:
        tempo di failure del sistema totale
    """

    # =========================
    # BLOCCO ABC (2oo3)
    # =========================

    t_A = sample_exponential(lmbda)
    t_B = sample_exponential(lmbda)
    t_C = sample_exponential(lmbda)

    times_ABC = [t_A, t_B, t_C]
    times_ABC.sort()

    # il 2oo3 fallisce al secondo guasto
    t_ABC = times_ABC[1]

    # =========================
    # BLOCCO DE (serie)
    # =========================

    t_D = sample_exponential(lmbda)
    t_E = sample_exponential(lmbda)

    # sistema in serie → fallisce al primo guasto
    t_DE = min(t_D, t_E)

    # =========================
    # SISTEMA TOTALE
    # =========================

    # ABC in serie con DE → fallisce quando uno dei due fallisce
    t_system = min(t_ABC, t_DE)

    return t_system

# in questo system.py:
# Calcolo il tempo di failure del sottosistema ABC (2oo3) come ho fatto nel point0
# calcolo il tempo di failure del sottosistema DE (serie).
# Il sistema totale è in serie tra ABC e DE,
# quindi fallisce quando fallisce il primo dei due:
# per questo si prende il minimo tra t_ABC e t_DE.