import numpy as np
from collections import deque
from src.common.sampling import sample_exponential

"""
SYSTEM LOGIC MODULE - POINT 3
-------------------------------------
This module implements the advanced Discrete Event Simulation (DES) logic for 
the Point 3 system. It extends the repairable model by introducing limited 
maintenance resources and Common Cause Failures (CCF).

Key Engineering Modifications (from project guidelines):
1. Limited Maintenance Teams (Queuing Logic):
   - Only 1 repair team is dedicated to Subsystem ABC (repair rate μ1).
   - Only 1 repair team is dedicated to Subsystem DE (repair rate μ2).
   - If multiple components in a subsystem fail, they are placed in a FIFO 
     (First-In, First-Out) waiting queue.
2. External Shocks (Common Cause Failures):
   - Shocks hit the system following a Poisson process with occurrence rate λ_c.
   - Upon a shock, each CURRENTLY WORKING component fails independently with 
     probability p = 0.1 (Bernoulli trial).
   - Components already in a DOWN state are unaffected.
"""


def _system_is_up(states):
    """
    Evaluates the global operational state of the system based on the 
    current physical states of its 5 individual components.

    Parameters
    ----------
    states : dict
        A dictionary mapping each component ('A', 'B', 'C', 'D', 'E') 
        to its current state ('UP' or 'DOWN').

    Returns
    -------
    bool
        True if the global system is operational (ABC is 2oo3 AND DE is 1oo2).
    """
    abc_up = sum(states[c] == "UP" for c in ["A", "B", "C"])
    de_up  = sum(states[c] == "UP" for c in ["D", "E"])
    return abc_up >= 2 and de_up >= 1


def simulate_system_once(lambda_1, lambda_2, mu_1, mu_2,
                          lambda_c, p_shock, mission_time):
    """
    Simulates a single Monte Carlo run of the system featuring repair queues 
    and external shocks.

    Algorithm (Advanced Event-Driven Strategy):
    The event calendar tracks three types of future events:
      (a) Spontaneous failure of an UP component.
      (b) Completion of a repair process (only if the team is currently busy).
      (c) Arrival of the next external shock.
    At each step, the simulation clock jumps to the nearest event in time.

    Parameters
    ----------
    lambda_1    : float  — Failure rate A, B, C [h^-1]
    lambda_2    : float  — Failure rate D, E    [h^-1]
    mu_1        : float  — Repair rate A, B, C  [h^-1]
    mu_2        : float  — Repair rate D, E     [h^-1]
    lambda_c    : float  — Shock arrival rate   [h^-1]
    p_shock     : float  — Probability of component failure during a shock
    mission_time: float  — Maximum time horizon [h]

    Returns
    -------
    dict
        Contains 't_first_failure', 'time_up', 'n_repairs', and 'state_at' 
        (a sparse list of temporal state transitions used to reconstruct A(t)).
    """

    # ── Initialization ────────────────────────────────────────────────────────
    
    # All components start perfectly operational at t=0
    states = {c: "UP" for c in ["A", "B", "C", "D", "E"]}

    
    lmbda_map = {"A": lambda_1, "B": lambda_1, "C": lambda_1,
                 "D": lambda_2, "E": lambda_2}
    mu_map    = {"A": mu_1,     "B": mu_1,     "C": mu_1,
                 "D": mu_2,     "E": mu_2}

    # ── Repair Queues Setup ───────────────────────────────────────────────────
    # Using collections.deque for O(1) append and popleft FIFO operations
    repair_queue_ABC = deque()
    repair_queue_DE  = deque()
    # Track which component is currently occupying the repair team (None = Free)
    being_repaired_ABC = None
    being_repaired_DE  = None

    # ── Event calendar ─────────────────────────────────────────────────────
    INF = float("inf")
    # t_fail: next spontaneous failure for each component (INF if already DOWN)
    t_fail      = {c: sample_exponential(lmbda_map[c]) for c in states}
    # t_repair: completion time of the ongoing repair (INF if team is idle)
    t_repair_ABC = INF   
    t_repair_DE  = INF
    # t_shock: next arrival time of the external Poisson shock
    t_shock      = sample_exponential(lambda_c)  # primo shock

    # ── Tracking Variables ───────────────────────────────────────────────
    t               = 0.0
    t_first_failure = INF
    time_up         = 0.0
    n_repairs       = 0
    sys_up          = True   # sistema UP all'inizio

    # state_at logs (time, system_status) to reconstruct the history for A(t)
    state_at = [(0.0, True)]

    # ── Main Event-Driven Loop ────────────────────────────────────────
    while t < mission_time:

        # Filter out components that are DOWN (they cannot fail spontaneously)
        candidate_fails = {c: t_fail[c] for c in states if states[c] == "UP"}
        # Compile all scheduled future events
        all_events = list(candidate_fails.values()) + [
            t_repair_ABC, t_repair_DE, t_shock
        ]
        # Identify the chronological next event and truncate at mission horizon
        t_next_event = min(all_events)
        t_clipped    = min(t_next_event, mission_time)

        # Accumulate operational metrics for the elapsed time interval
        if sys_up:
            time_up += t_clipped - t
        # Advance simulation clock
        t = t_clipped
        if t >= mission_time:
            break

        # ── Event Classification & Processing ─────────────────────────────────
        # EPS (Epsilon) is a numerical tolerance threshold used to safely handle 
        # floating-point equality comparisons and prevent tie-breaking bugs.
        EPS = 1e-12

        # EVENT A: Repair Team ABC finishes fixing a component
        if abs(t - t_repair_ABC) < EPS:
            comp = being_repaired_ABC
            states[comp] = "UP"
            n_repairs    += 1
            # Schedule the next spontaneous failure for the newly fixed component
            t_fail[comp]     = t + sample_exponential(lmbda_map[comp])
            # Check the FIFO queue: if a component is waiting, start repairing it
            if repair_queue_ABC:
                being_repaired_ABC = repair_queue_ABC.popleft()
                t_repair_ABC       = t + sample_exponential(mu_map[being_repaired_ABC])
            else:
                being_repaired_ABC = None
                t_repair_ABC       = INF

        # EVENT B: Repair Team DE finishes fixing a component
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

        # EVENT C: External Shock Arrival (Common Cause Failure)
        elif abs(t - t_shock) < EPS:
            # Perform a Bernoulli trial (prob = p_shock) for each UP component
            for comp in ["A", "B", "C", "D", "E"]:
                if states[comp] == "UP" and np.random.random() < p_shock:
                    
                    states[comp] = "DOWN"
                    t_fail[comp] = INF  # Suspends spontaneous failure clock

                    # Route the failed component to the correct subsystem logic
                    if comp in ["A", "B", "C"]:
                        if being_repaired_ABC is None:
                            # Team libero: inizia subito
                            being_repaired_ABC = comp
                            t_repair_ABC       = t + sample_exponential(mu_map[comp])
                        else:
                            repair_queue_ABC.append(comp)
                    else:  # D or E
                        if being_repaired_DE is None:
                            being_repaired_DE = comp
                            t_repair_DE       = t + sample_exponential(mu_map[comp])
                        else:
                            repair_queue_DE.append(comp)

            # Schedule the next external shock
            t_shock = t + sample_exponential(lambda_c)

        # EVENT D: Spontaneous Component Failure
        else:
            
            comp = min(candidate_fails, key=candidate_fails.get)
            states[comp] = "DOWN"
            t_fail[comp] = INF  

            
            if comp in ["A", "B", "C"]:
                if being_repaired_ABC is None:
                    being_repaired_ABC = comp
                    t_repair_ABC       = t + sample_exponential(mu_map[comp])
                else:
                    repair_queue_ABC.append(comp) # Enqueue if team is busy
            else:  # D or E
                if being_repaired_DE is None:
                    being_repaired_DE = comp
                    t_repair_DE       = t + sample_exponential(mu_map[comp])
                else:
                    repair_queue_DE.append(comp)

        # ── Global State Update ─────────────────────────
        was_up = sys_up
        sys_up = _system_is_up(states)

        # Log transition if the system state changed (used for A(t))
        if sys_up != was_up:
            state_at.append((t, sys_up))

        # Record the exact time of the FIRST global failure (censoring tracker)
        if was_up and not sys_up and t_first_failure == INF:
            t_first_failure = t

    # Append the final boundary condition to close the interval
    state_at.append((mission_time, sys_up))

    return {
        "t_first_failure": t_first_failure,
        "time_up":         time_up,
        "n_repairs":       n_repairs,
        "state_at":        state_at,
    }


def system_state_on_grid(state_at, time_grid):
    """
    Projects the sparse list of state transitions onto a uniform time grid.
    This is necessary to easily compute the ensemble average for Instantaneous 
    Availability A(t) across thousands of Monte Carlo runs.

    Algorithm:
    Uses a highly efficient two-pointer approach (O(N+G) complexity) to traverse 
    the transition list and the time grid concurrently.

    Parameters
    ----------
    state_at  : list of tuple
        Sparse chronological transitions format: (time_of_event, boolean_sys_up).
    time_grid : np.ndarray
        The fixed temporal grid to project the states onto.

    Returns
    -------
    np.ndarray
        An array of int8 (1=UP, 0=DOWN) evaluating the state at each grid point.
    """
    
    result   = np.empty(len(time_grid), dtype=np.int8)
    s_idx    = 0                          # Pointer for state_at list
    cur_val  = int(state_at[0][1])        # Initial system state at t=0

    for g_idx, tg in enumerate(time_grid):
        # Advance the state pointer as long as the next transition happened 
        # BEFORE or EXACTLY AT the current grid timestamp (tg)
        while s_idx + 1 < len(state_at) and state_at[s_idx + 1][0] <= tg:
            s_idx  += 1
            cur_val = int(state_at[s_idx][1])
        result[g_idx] = cur_val

    return result