from src.common.sampling import sample_exponential

"""
SYSTEM LOGIC MODULE - POINT 1
-------------------------------------
This module defines the topological failure logic for the Point 1 system.
The architecture consists of a 2-out-of-3 (2oo3) subsystem (components A, B, C)
connected in series with a parallel subsystem (components D, E), without any repair.
"""

def simulate_system_once(lambda_1, lambda_2):
    """
    Simulates a single Monte Carlo run (history) for the complete system 
    without repair operations.

    System Topology:
    - Subsystem ABC: 2-out-of-3 architecture.
    - Subsystem DE: Parallel architecture.
    - Global System: Subsystem ABC and Subsystem DE connected in series.

    Parameters
    ----------
    lambda_1 : float
        Failure rate for components A, B, and C [h^-1].
    lambda_2 : float
        Failure rate for components D and E [h^-1].

    Returns
    -------
    float
        The absolute failure time of the global system [h].
    """

    # =========================================================================
    # 1. SUBSYSTEM ABC (2-out-of-3 Architecture)
    # =========================================================================
    # Sample independent failure times for components A, B, and C

    t_A = sample_exponential(lambda_1)
    t_B = sample_exponential(lambda_1)
    t_C = sample_exponential(lambda_1)

    times_ABC = [t_A, t_B, t_C]
    times_ABC.sort()

    # A 2oo3 system requires at least 2 working components.
    # Therefore, it fails exactly when the 2nd component fails.
    # In a 0-indexed sorted array, the 2nd chronological failure is at index 1.
    t_ABC = times_ABC[1]

    # =========================================================================
    # 2. SUBSYSTEM DE (Parallel Architecture)
    # =========================================================================
    # Sample independent failure times for components D and E

    t_D = sample_exponential(lambda_2)
    t_E = sample_exponential(lambda_2)

    # A parallel system fails ONLY when ALL of its components have failed.
    # Mathematically, its failure time is the maximum of the individual failure times.
    t_DE = max(t_D, t_E)

    # =========================================================================
    # 3. TOTAL SYSTEM (Series Connection)
    # =========================================================================
    # A series system fails as soon as ANY of its internal subsystems fails.
    # Mathematically, the overall failure time is the minimum between the 
    # failure times of Subsystem ABC and Subsystem DE.
    t_system = min(t_ABC, t_DE)

    return t_system