from src.common.sampling import sample_exponential

# =============================================================================
# POINT 2 — Sistema completo (ABC 2oo3 + DE parallelo) CON RIPARAZIONE
#
# Ipotesi (da testo):
#   - Ogni componente ha un team dedicato → fino a 5 riparazioni simultanee
#   - Il repair inizia immediatamente dopo il guasto
#   - I componenti continuano a guastarsi anche quando il SISTEMA è DOWN
#   - Failure rate: λ₁ = 1e-3 h⁻¹ (A,B,C) | λ₂ = 2e-3 h⁻¹ (D,E)
#   - Repair rate:  μ₁ = 1e-2 h⁻¹ (A,B,C) | μ₂ = 2e-2 h⁻¹ (D,E)
# =============================================================================


def _system_is_up(states):
    """
    Determina se il sistema è UP dato lo stato dei 5 componenti.

    Parameters
    ----------
    states : dict
        Dizionario {componente: 'UP'|'DOWN'} per A, B, C, D, E.

    Returns
    -------
    bool
        True se il sistema è operativo.
    """
    # 2oo3: almeno 2 tra A, B, C devono essere UP
    abc_up = sum(states[c] == "UP" for c in ["A", "B", "C"])
    # parallelo: almeno 1 tra D, E deve essere UP
    de_up = sum(states[c] == "UP" for c in ["D", "E"])

    return abc_up >= 2 and de_up >= 1


def simulate_system_once(lambda_1, lambda_2, mu_1, mu_2, mission_time):
    """
    Simula UNA run del sistema completo con riparazione su [0, mission_time].

    Strategia event-driven:
      - Per ogni componente, generiamo il prossimo evento (guasto o riparazione)
        e saltiamo direttamente all'evento più vicino nel tempo.
      - Teniamo traccia di: stato corrente di ogni componente, se il sistema
        ha mai fallito (per R(t)), e quanto tempo il sistema è UP (per A(t)).

    Parameters
    ----------
    lambda_1 : float  Failure rate componenti A, B, C [h⁻¹]
    lambda_2 : float  Failure rate componenti D, E [h⁻¹]
    mu_1     : float  Repair rate componenti A, B, C [h⁻¹]
    mu_2     : float  Repair rate componenti D, E [h⁻¹]
    mission_time : float  Orizzonte temporale [h]

    Returns
    -------
    dict con:
        t_first_failure : float  — tempo del primo failure di sistema (∞ se mai fallito)
        time_up         : float  — tempo totale in cui il sistema è UP
        n_repairs       : int    — numero di riparazioni completate
        system_up_at    : callable — funzione che dice se il sistema era UP a un dato t
                          (nota: questa run NON memorizza la storia continua;
                           per A(t) usiamo la stima Monte Carlo su molte run)
    """

    # --- Stato iniziale ---
    # Tutti UP a t=0
    states = {c: "UP" for c in ["A", "B", "C", "D", "E"]}

    # Parametri per ogni componente
    lmbda = {"A": lambda_1, "B": lambda_1, "C": lambda_1,
             "D": lambda_2, "E": lambda_2}
    mu    = {"A": mu_1,     "B": mu_1,     "C": mu_1,
             "D": mu_2,     "E": mu_2}

    # Prossimo evento per ciascun componente:
    #   se UP  → prossimo guasto   (tempo assoluto)
    #   se DOWN → prossima riparazione (tempo assoluto)
    t_next = {c: sample_exponential(lmbda[c]) for c in states}

    t = 0.0
    t_first_failure = float("inf")
    time_up = 0.0
    n_repairs = 0

    sys_up_now = _system_is_up(states)  # True all'inizio (tutti UP)

    while t < mission_time:
        # Trova il prossimo evento tra tutti i componenti
        next_comp = min(t_next, key=t_next.get)
        t_event = t_next[next_comp]

        # Tronca all'orizzonte
        t_event_clipped = min(t_event, mission_time)

        # Accumula tempo UP/DOWN nell'intervallo [t, t_event_clipped]
        if sys_up_now:
            time_up += t_event_clipped - t

        t = t_event_clipped

        if t >= mission_time:
            break  # Fine simulazione

        # --- Elabora l'evento ---
        comp = next_comp
        if states[comp] == "UP":
            # Guasto del componente
            states[comp] = "DOWN"
            t_next[comp] = t + sample_exponential(mu[comp])  # schedula repair
        else:
            # Riparazione completata
            states[comp] = "UP"
            n_repairs += 1
            t_next[comp] = t + sample_exponential(lmbda[comp])  # schedula prossimo guasto

        # Aggiorna stato del sistema
        sys_was_up = sys_up_now
        sys_up_now = _system_is_up(states)

        # Primo failure del sistema (transizione UP→DOWN)
        if sys_was_up and not sys_up_now:
            if t_first_failure == float("inf"):
                t_first_failure = t

    return {
        "t_first_failure": t_first_failure,
        "time_up": time_up,
        "n_repairs": n_repairs,
    }