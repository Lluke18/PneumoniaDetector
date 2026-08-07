import numpy as np
import matplotlib.pyplot as plt


# A: 3 exemple (zile diferite), fiecare cu 2 informații (ex: plouă, timp liber)
A = np.array([[1.5, 2.0],
              [2.0, 1.0],
              [3.5, 3.0]])

# W: 2 ponderi (cât de mult contează fiecare din cele 2 informații)
W = np.array([0.5, -0.2])

# b: Bias-ul nostru (un singur număr)
b = 0.1

def combinatia_liniara(A, W, b):
    """
    Calculează înmulțirea matricială dintre intrări și ponderi, apoi adună bias-ul.
    """
    # TODO: Implementează formula z = A * W + b folosind numpy
    z = A @ W + b

    return z

def step_function(z):
    # np.where verifică condiția: dacă z >= 0 pune 1, altfel pune 0
    return np.where(z >= 0, 1, 0)

# Generăm numere de la -10 la 10 pentru a desena graficul
z_valori = np.linspace(-10, 10, 100)
y_treapta = step_function(z_valori)

plt.plot(z_valori, y_treapta, linewidth=3, color='blue')
plt.title("Funcția Treaptă (Step)")
plt.grid(True)
plt.show()