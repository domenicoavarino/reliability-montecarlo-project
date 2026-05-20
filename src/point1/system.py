from src.common.sampling import sample_exponential

# questo file descrive la logica del sistema al point1


def simulate_system_once(lambda_1, lambda_2):
    """
    Simula UNA volta il sistema completo senza riparazioni:

    - Sottosistema ABC: 2oo3
    - Sottosistema DE: parallelo
    - ABC e DE sono in serie

    ritorna:
        tempo di failure del sistema totale
    """

    # =========================
    # BLOCCO ABC (2oo3)
    # =========================

    t_A = sample_exponential(lambda_1)
    t_B = sample_exponential(lambda_1)
    t_C = sample_exponential(lambda_1)

    times_ABC = [t_A, t_B, t_C]
    times_ABC.sort()

    # il 2oo3 fallisce al secondo guasto
    t_ABC = times_ABC[1]

    # =========================
    # BLOCCO DE (parallelo)
    # =========================

    t_D = sample_exponential(lambda_2)
    t_E = sample_exponential(lambda_2)

    # sistema in parallelo: fallisce quando falliscono entrambi
    t_DE = max(t_D, t_E)

    # =========================
    # SISTEMA TOTALE
    # =========================

    # ABC in serie con DE: il sistema totale fallisce
    # quando fallisce il primo tra ABC e DE
    t_system = min(t_ABC, t_DE)

    return t_system