"""
SYSTEM LOGIC MODULE - POINT 0
-------------------------------------
This module defines the topological failure logic for the Point 0 system,
which consists exclusively of a 2-out-of-3 (2oo3) subsystem composed of 
components A, B, and C without any repair interventions.
"""
from src.common.sampling import sample_exponential 

def simulate_2oo3_once(lmbda):
    """
    Simulates a single run (one history) of the 2-out-of-3 subsystem.
    
    Engineering Logic:
    A 2oo3 system requires at least 2 components to be operational (UP) 
    for the system to function. Therefore, the system fails exactly when 
    the 2nd component fails. 
    
    Algorithm:
    1. Sample independent failure times for components A, B, and C.
    2. Sort the failure times in ascending order.
    3. The system failure time corresponds to the 2nd chronological failure 
       (which is at index 1 of the sorted array).
    
    Parameters
    ----------
    lmbda : float
        The constant failure rate (λ) of components A, B, and C [h^-1].
        
    Returns
    -------
    float
        The absolute time of system failure [h].
    """

    # 1. Generate independent exponential failure times for each component
    t_A = sample_exponential(lmbda)
    t_B = sample_exponential(lmbda)
    t_C = sample_exponential(lmbda)

    # 2. Store the failure times in a list to evaluate their chronological sequence
    times = [t_A, t_B, t_C]

    # 3. Sort the times in ascending order 
    #    (times[0] = 1st failure, times[1] = 2nd failure, times[2] = 3rd failure)
    times.sort()

    # 4. Return the time of the 2nd failure, which triggers the 2oo3 system breakdown
    return times[1]
# =========================================================================
    # LOGICAL SUMMARY OF THE 2oo3 SYSTEM SIMULATION
    # =========================================================================
    # 1. The sample_exponential(lmbda) function from the sampling module 
    #    generates random failure times using the theoretical inverse transform 
    #    formula based on the failure rate parameter (lambda).
    #
    # 2. Here, it is called three times to simulate the individual, independent 
    #    failure times of the three system components: T_A, T_B, and T_C.
    #
    # 3. These three times are stored in a list and sorted in ascending order.
    #
    # 4. Since the architecture is a 2-out-of-3 (2oo3) system, it fails only 
    #    when at least two components have failed. In other words, the system 
    #    survives the first component failure but breaks down exactly upon 
    #    the second failure.
    #
    # 5. Consequently, the overall system failure time corresponds to the 
    #    second smallest time in the chronological sequence, which is 
    #    extracted using times[1].
    # =========================================================================