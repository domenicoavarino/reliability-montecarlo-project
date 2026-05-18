#questo file si occupa della logica del sistema, risponde alla "domanda quando fallisce il sistema?"

from src.common.sampling import sample_exponential #Qui stai importando la funzione che hai scritto tu (sampling.py).
#Definisci una funzione :simulate → simulazione , 2oo3 → 2 out of 3, once → UNA simulazione
#lmbda = tasso di guasto (failure rate) che tradotto “simula una volta un sistema con 3 componenti”
def simulate_2oo3_once(lmbda):
    """
    Simula UNA volta il sistema 2oo3 (A, B, C)

    ritorna:
        tempo di failure del sistema
    """

    # tempi di guasto dei 3 componenti
    t_A = sample_exponential(lmbda)#Generi il tempo di guasto del componente A
    t_B = sample_exponential(lmbda)#Generi il tempo di guasto del componente B
    t_C = sample_exponential(lmbda)#Generi il tempo di guasto del componente C

    # mettiamo in lista, Perché?così puoi lavorarci facilmente (ordinare, ecc.)
    times = [t_A, t_B, t_C]

    # ordiniamo i tempi
    times.sort()

    # il sistema fallisce al secondo guasto
    return times[1]
#Restituisci il secondo guastoPerché?Perché:primo guasto → sistema ancora vivo,secondo guasto → sistema muore
#quindi quello è il tempo di failure del sistema
#RIASSUNTO LOGICO:

#Questa funzione fa:

#genera 3 tempi casuali, li ordina, prende il secondo e lo restituisce
# La funzione sample_exponential(lmbda) creata in sampling genera tempi di guasto casuali utilizzanda lmbda e la formula teorica data.
# a partire da un parametro lambda (lmbda), che rappresenta il tasso di guasto (failure rate).
#
# In system.py questa funzione viene chiamata tre volte per simulare
# i tempi di guasto dei tre componenti del sistema:
# T_A, T_B e T_C.
#
# I tre tempi vengono poi inseriti in una lista e ordinati in ordine crescente.
#
# Poiché il sistema è di tipo 2oo3 (2 out of 3), il sistema fallisce
# quando si guastano almeno due componenti.
#
# Di conseguenza, il tempo di guasto del sistema corrisponde
# al secondo tempo più piccolo (secondo guasto), cioè times[1].