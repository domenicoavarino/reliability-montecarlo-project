import numpy as np

def sample_exponential(lmbda):
    """
    Genera tempi esponenziali degli eventi a partire da una U(0,1)
    e dal parametro lambda (failure/repair rate).
    """
    u = np.random.random() 
    return -np.log(1 - u) / lmbda

if __name__ == "__main__":
    np.random.seed(42)      #seed fissato per riproducibilità
    for _ in range(5):
        print(sample_exponential(0.1))  #genero 5 numeri casuali