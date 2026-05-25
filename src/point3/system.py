import numpy as np
from collections import deque
from src.common.sampling import sample_exponential

# =============================================================================
# POINT 3 — Sistema completo CON RIPARAZIONE A CODA + SHOCK ESTERNO
#
# Differenze rispetto al point 2:
#
#   MODIFICA 1 — Repair con coda (un team per sottosistema):
#     - 1 team dedicato al sottosistema ABC (rate μ₁), ripara uno alla volta.
#     - 1 team dedicato al sottosistema DE  (rate μ₂), ripara uno alla volta.
#     - Se più componenti dello stesso sottosistema sono DOWN simultaneamente,
#       quelli in attesa vengono messi in coda FIFO (ordine di guasto).
#     - Il repair del prossimo in coda inizia solo quando il team termina
#       quello in corso.
#
#   MODIFICA 2 — Shock esterno:
#     - Arriva con rate λ_c = 2e-3 h⁻¹ (processo di Poisson).
#     - All'arrivo dello shock, ogni componente attualmente UP fallisce
#       indipendentemente con probabilità p = 0.1.
#     - Componenti già DOWN non sono influenzati dallo shock.
#     - Uno shock può causare 0, 1, ... , 5 guasti simultanei.
#
# Invariato rispetto al point 2:
#   - Topologia sistema: ABC (2oo3) in serie con DE (parallelo).
#   - I componenti continuano a guastarsi anche quando il sistema è DOWN.
#   - Failure rate: λ₁ = 1e-3 h⁻¹ (A,B,C) | λ₂ = 2e-3 h⁻¹ (D,E).
#   - Repair rate:  μ₁ = 1e-2 h⁻¹ (A,B,C) | μ₂ = 2e-2 h⁻¹ (D,E).
# =============================================================================


def _system_is_up(states):
    """
    Determina se il sistema è UP dato lo stato dei 5 componenti.

    Il sistema è UP se e solo se:
      - almeno 2 dei 3 componenti {A, B, C} sono UP  (2oo3)
      - almeno 1 dei 2 componenti {D, E} è UP         (parallelo)

    Parameters
    ----------
    states : dict  {componente: 'UP'|'DOWN'}

    Returns
    -------
    bool
    """
    abc_up = sum(states[c] == "UP" for c in ["A", "B", "C"])
    de_up  = sum(states[c] == "UP" for c in ["D", "E"])
    return abc_up >= 2 and de_up >= 1


def simulate_system_once(lambda_1, lambda_2, mu_1, mu_2,
                          lambda_c, p_shock, mission_time):
    """
    Simula UNA run del sistema completo con coda di repair e shock esterno.

    Strategia event-driven con event calendar:
      Manteniamo un dizionario degli eventi futuri schedulati.
      Ad ogni iterazione saltiamo al prossimo evento nel tempo, che può essere:
        (a) guasto spontaneo di un componente UP
        (b) completamento di una riparazione (solo per il componente in repair)
        (c) arrivo di uno shock esterno

      Logica della coda di repair:
        - repair_queue_ABC e repair_queue_DE sono code FIFO di componenti
          in attesa di riparazione (nell'ordine in cui si sono guastati).
        - being_repaired_ABC / being_repaired_DE indicano il componente
          attualmente in riparazione (None se il team è libero).
        - Quando un componente si guasta:
            → se il team è libero: inizia subito la riparazione.
            → se il team è occupato: il componente entra in coda.
        - Quando una riparazione finisce:
            → il componente torna UP e viene schedulato il suo prossimo guasto.
            → se la coda non è vuota: inizia subito la riparazione del prossimo.
            → se la coda è vuota: il team diventa libero.

      Logica dello shock:
        - Schedula il primo shock a t = sample_exponential(lambda_c).
        - Quando lo shock arriva, per ogni componente UP campiona Bernoulli(p).
        - I componenti colpiti vengono messi DOWN e aggiunti alla coda.
        - Schedula il prossimo shock.

    Parameters
    ----------
    lambda_1    : float  — failure rate A, B, C [h⁻¹]
    lambda_2    : float  — failure rate D, E    [h⁻¹]
    mu_1        : float  — repair rate  A, B, C [h⁻¹]
    mu_2        : float  — repair rate  D, E    [h⁻¹]
    lambda_c    : float  — shock arrival rate   [h⁻¹]
    p_shock     : float  — prob. di guasto per componente UP al momento dello shock
    mission_time: float  — orizzonte temporale  [h]

    Returns
    -------
    dict con:
        t_first_failure : float — tempo primo failure di sistema (inf se mai)
        time_up         : float — tempo totale UP del sistema
        n_repairs       : int   — numero riparazioni completate in [0, T_M]
        state_at        : list  — lista di (t, sys_up) per ricostruire la storia
                          [usata in simulation.py per A(t) su time_grid]
    """

    # ── Stato iniziale ─────────────────────────────────────────────────────
    # Tutti e 5 i componenti partono UP a t=0.
    states = {c: "UP" for c in ["A", "B", "C", "D", "E"]}

    # Mappe parametri per ogni componente
    lmbda_map = {"A": lambda_1, "B": lambda_1, "C": lambda_1,
                 "D": lambda_2, "E": lambda_2}
    mu_map    = {"A": mu_1,     "B": mu_1,     "C": mu_1,
                 "D": mu_2,     "E": mu_2}

    # ── Code di repair ─────────────────────────────────────────────────────
    # repair_queue_*: componenti DOWN in attesa (FIFO, ordine di guasto).
    # being_repaired_*: componente attualmente in riparazione (None = team libero).
    repair_queue_ABC = deque()
    repair_queue_DE  = deque()
    being_repaired_ABC = None
    being_repaired_DE  = None

    # ── Event calendar ─────────────────────────────────────────────────────
    # t_fail[c]: prossimo guasto spontaneo del componente c (solo se UP).
    # t_repair_ABC / t_repair_DE: fine riparazione in corso (inf se nessuna).
    # t_shock: prossimo shock esterno.
    INF = float("inf")
    t_fail      = {c: sample_exponential(lmbda_map[c]) for c in states}
    t_repair_ABC = INF   # nessuna riparazione in corso inizialmente
    t_repair_DE  = INF
    t_shock      = sample_exponential(lambda_c)  # primo shock

    # ── Variabili di tracking ───────────────────────────────────────────────
    t               = 0.0
    t_first_failure = INF
    time_up         = 0.0
    n_repairs       = 0
    sys_up          = True   # sistema UP all'inizio

    # state_at: lista di (t_evento, sys_up_dopo_evento) per ricostruire A(t).
    # Inizia con il punto t=0.
    state_at = [(0.0, True)]

    # ── Loop principale event-driven ────────────────────────────────────────
    while t < mission_time:

        # Trova il prossimo evento: minimo tra tutti gli eventi schedulati.
        # Gli eventi di guasto valgono solo per i componenti UP.
        candidate_fails = {c: t_fail[c] for c in states if states[c] == "UP"}

        all_events = list(candidate_fails.values()) + [
            t_repair_ABC, t_repair_DE, t_shock
        ]
        t_next_event = min(all_events)
        t_clipped    = min(t_next_event, mission_time)

        # Accumula tempo UP nell'intervallo [t, t_clipped]
        if sys_up:
            time_up += t_clipped - t

        t = t_clipped
        if t >= mission_time:
            break

        # ── Identifica il tipo di evento ────────────────────────────────────
        # Usiamo una soglia numerica per gestire possibili tie floating-point.
        EPS = 1e-12

        # (a) Fine riparazione ABC
        if abs(t - t_repair_ABC) < EPS:
            comp = being_repaired_ABC
            states[comp] = "UP"
            n_repairs    += 1
            # Schedula il prossimo guasto spontaneo del componente appena riparato
            t_fail[comp]     = t + sample_exponential(lmbda_map[comp])
            # Avvia la prossima riparazione in coda ABC (se presente)
            if repair_queue_ABC:
                being_repaired_ABC = repair_queue_ABC.popleft()
                t_repair_ABC       = t + sample_exponential(mu_map[being_repaired_ABC])
            else:
                being_repaired_ABC = None
                t_repair_ABC       = INF

        # (b) Fine riparazione DE
        elif abs(t - t_repair_DE) < EPS:
            comp = being_repaired_DE
            states[comp] = "UP"
            n_repairs    += 1
            t_fail[comp]     = t + sample_exponential(lmbda_map[comp])
            if repair_queue_DE:
                being_repaired_DE = repair_queue_DE.popleft()
                t_repair_DE       = t + sample_exponential(mu_map[being_repaired_DE])
            else:
                being_repaired_DE = None
                t_repair_DE       = INF

        # (c) Shock esterno
        elif abs(t - t_shock) < EPS:
            # Per ogni componente UP: fallisce con probabilità p_shock
            for comp in ["A", "B", "C", "D", "E"]:
                if states[comp] == "UP" and np.random.random() < p_shock:
                    # Componente colpito dallo shock → va DOWN
                    states[comp] = "DOWN"
                    t_fail[comp] = INF  # non genera più guasti spontanei finché DOWN

                    # Aggiunge alla coda del sottosistema corretto
                    if comp in ["A", "B", "C"]:
                        if being_repaired_ABC is None:
                            # Team libero: inizia subito
                            being_repaired_ABC = comp
                            t_repair_ABC       = t + sample_exponential(mu_map[comp])
                        else:
                            repair_queue_ABC.append(comp)
                    else:  # D o E
                        if being_repaired_DE is None:
                            being_repaired_DE = comp
                            t_repair_DE       = t + sample_exponential(mu_map[comp])
                        else:
                            repair_queue_DE.append(comp)

            # Schedula il prossimo shock
            t_shock = t + sample_exponential(lambda_c)

        # (d) Guasto spontaneo di un componente
        else:
            # Trova il componente UP che si guasta in questo momento
            comp = min(candidate_fails, key=candidate_fails.get)
            states[comp] = "DOWN"
            t_fail[comp] = INF  # non genera guasti spontanei finché in repair

            # Aggiunge alla coda del sottosistema corretto
            if comp in ["A", "B", "C"]:
                if being_repaired_ABC is None:
                    being_repaired_ABC = comp
                    t_repair_ABC       = t + sample_exponential(mu_map[comp])
                else:
                    repair_queue_ABC.append(comp)
            else:  # D o E
                if being_repaired_DE is None:
                    being_repaired_DE = comp
                    t_repair_DE       = t + sample_exponential(mu_map[comp])
                else:
                    repair_queue_DE.append(comp)

        # ── Aggiorna stato del sistema dopo l'evento ─────────────────────────
        was_up = sys_up
        sys_up = _system_is_up(states)

        # Registra transizione per ricostruzione A(t)
        if sys_up != was_up:
            state_at.append((t, sys_up))

        # Primo failure del sistema (prima transizione UP→DOWN)
        if was_up and not sys_up and t_first_failure == INF:
            t_first_failure = t

    # Punto finale per chiudere l'ultima fase
    state_at.append((mission_time, sys_up))

    return {
        "t_first_failure": t_first_failure,
        "time_up":         time_up,
        "n_repairs":       n_repairs,
        "state_at":        state_at,
    }


def system_state_on_grid(state_at, time_grid):
    """
    Ricostruisce lo stato del sistema (1=UP, 0=DOWN) su una griglia temporale
    a partire dalla lista sparsa di transizioni prodotta da simulate_system_once.

    Strategia: scorre la griglia e la lista state_at in parallelo (O(N+G)).

    Parameters
    ----------
    state_at  : list of (float, bool) — transizioni (t, sys_up)
    time_grid : np.ndarray            — griglia temporale

    Returns
    -------
    np.ndarray int8 di shape (len(time_grid),)
    """
    import numpy as np
    result   = np.empty(len(time_grid), dtype=np.int8)
    s_idx    = 0                          # indice in state_at
    cur_val  = int(state_at[0][1])        # stato al tempo 0

    for g_idx, tg in enumerate(time_grid):
        # Avanza in state_at finché il prossimo cambio è ≤ tg
        while s_idx + 1 < len(state_at) and state_at[s_idx + 1][0] <= tg:
            s_idx  += 1
            cur_val = int(state_at[s_idx][1])
        result[g_idx] = cur_val

    return result