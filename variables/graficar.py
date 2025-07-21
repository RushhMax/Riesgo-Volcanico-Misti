import matplotlib.pyplot as plt

# Visualización
def graficar_fuzzy(x, funciones, etiquetas, titulo, xlabel):
    plt.figure()
    for f, label in zip(funciones, etiquetas):
        plt.plot(x, f, label=label)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel("Grado de pertenencia")
    plt.legend()
    plt.grid(True)
    plt.show()