from src.common.sampling import sample_exponential

"""
SYSTEM LOGIC MODULE - POINT 2
-------------------------------------
This module implements the Discrete Event Simulation (DES) logic for the 
Point 2 system. It models the full architecture (Subsystem ABC in series 
with Subsystem DE) incorporating independent repair processes.

Key Engineering Assumptions (from project guidelines):
- Continuous monitoring: Repair starts immediately upon component failure.
- Independent repair teams: Up to 5 repairs can occur simultaneously.
- Components continue to operate (and can thus fail) even when the global 
  system is in a DOWN state.
"""

def _system_is_up(states):
    """
    Evaluates the global operational state of the system based on the 
    current physical states of its 5 individual components.

    Topology Logic:
    - Subsystem ABC (2-out-of-3): Requires at least 2 components UP.
    - Subsystem DE (Parallel/1-out-of-2): Requires at least 1 component UP.
    - Global System (Series): Requires BOTH subsystems to be UP simultaneously.

    Parameters
    ----------
    states : dict
        A dictionary mapping each component ('A', 'B', 'C', 'D', 'E') 
        to its current state ('UP' or 'DOWN').

    Returns
    -------
    bool
        True if the global system is operational, False otherwise.
    """
    # Check Subsystem ABC (2oo3 architecture)
    abc_up = sum(states[c] == "UP" for c in ["A", "B", "C"])
    # Check Subsystem DE (Parallel architecture)
    de_up = sum(states[c] == "UP" for c in ["D", "E"])
    # The series connection requires both subsystems to be strictly functional
    return abc_up >= 2 and de_up >= 1


def simulate_system_once(lambda_1, lambda_2, mu_1, mu_2, mission_time):
    """
    Simulates a single Monte Carlo run of the repairable system over the 
    specified mission time horizon [0, T_M].

    Algorithm (Event-Driven Strategy):
    Instead of discretizing time into fixed steps (dt), the simulation maintains 
    an 'event calendar' (t_next). It computes the absolute time of the next 
    state transition (either a failure or a repair) for each component and 
    jumps the simulation clock directly to the nearest future event. 

    Parameters
    ----------
    lambda_1 : float
        Failure rate for components A, B, and C [h^-1].
    lambda_2 : float
        Failure rate for components D and E [h^-1].
    mu_1     : float
        Repair rate for components A, B, and C [h^-1].
    mu_2     : float
        Repair rate for components D and E [h^-1].
    mission_time : float
        The maximum time horizon for the mission [h].

    Returns
    -------
    dict
        A dictionary containing the simulated performance metrics:
        - 't_first_failure': Absolute time of the first global system failure.
        - 'time_up': Total cumulative time the system spent in the UP state.
        - 'n_repairs': Total number of successfully completed repair interventions.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    # The system starts at t=0 in the fully UP state with all components working
    states = {c: "UP" for c in ["A", "B", "C", "D", "E"]}

    # Parameter mapping for vectorized-like access
    lmbda = {"A": lambda_1, "B": lambda_1, "C": lambda_1,
             "D": lambda_2, "E": lambda_2}
    mu    = {"A": mu_1,     "B": mu_1,     "C": mu_1,
             "D": mu_2,     "E": mu_2}

    # Event Calendar: maps each component to its next scheduled state transition
    # Since all components start UP, the first events are spontaneous failures
    t_next = {c: sample_exponential(lmbda[c]) for c in states}
    # Simulation trackers
    t = 0.0
    t_first_failure = float("inf")
    time_up = 0.0
    n_repairs = 0
    # Current global state
    sys_up_now = _system_is_up(states)  # True all'inizio (tutti UP)
    # =========================================================================
    # EVENT-DRIVEN SIMULATION LOOP
    # =========================================================================
    while t < mission_time:
        # 1. Identify the nearest upcoming event in the calendar
        next_comp = min(t_next, key=t_next.get)
        t_event = t_next[next_comp]

        # 2. Truncate the event time if it exceeds the mission horizon
        t_event_clipped = min(t_event, mission_time)

        # 3. Accumulate operational metrics for the elapsed time interval
        if sys_up_now:
            time_up += t_event_clipped - t
        # Advance the simulation clock
        t = t_event_clipped
        # Stop if we reached the end of the mission
        if t >= mission_time:
            break  

        # 4. Process the scheduled event and trigger the state transition
        comp = next_comp
        if states[comp] == "UP":
            # Event: Component Failure
            states[comp] = "DOWN"
            # Schedule the completion of the immediate repair process
            t_next[comp] = t + sample_exponential(mu[comp])  # schedula repair
        else:
            # Event: Component Repair Completion
            states[comp] = "UP"
            n_repairs += 1
            # Schedule the next spontaneous failure
            t_next[comp] = t + sample_exponential(lmbda[comp])  

        # 5. Evaluate the global system status after the component transition
        sys_was_up = sys_up_now
        sys_up_now = _system_is_up(states)

        # Record the exact time of the FIRST global failure (UP -> DOWN transition)
        # This is critical for computing the censured MTTF later
        if sys_was_up and not sys_up_now:
            if t_first_failure == float("inf"):
                t_first_failure = t
    # Return the metrics collected during the single history
    return {
        "t_first_failure": t_first_failure,
        "time_up": time_up,
        "n_repairs": n_repairs,
    }