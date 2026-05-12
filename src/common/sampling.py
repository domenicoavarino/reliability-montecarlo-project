import random
import math


def sample_exponential(lmbda):
    u = random.random()
    return -math.log(1 - u) / lmbda  #formula teorica che genera Tempi esponenziali degli eventi (T) a partire da u (numero generato casualmente) e lambda (failure rate)
# Questo blocco viene eseguito solo
# se il file viene lanciato direttamente.
# Serve per testare la funzione
# sample_exponential().
if __name__ == "__main__":

    # Genera e stampa 5 campioni casuali
    # da una distribuzione esponenziale


    for _ in range(5):
        print(sample_exponential(0.1))

        #Questo codice genera tempi casuali esponenziali, ovvvero possibili failure times.