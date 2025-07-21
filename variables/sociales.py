import numpy as np
import matplotlib.pyplot as plt
from variables.graficar import graficar_fuzzy
import skfuzzy as fuzz

# Variable 5: Proximidad (km)
x_prox = np.arange(0, 21, 1)
prox_lejana = fuzz.trimf(x_prox, [10, 20, 20])
prox_media = fuzz.trimf(x_prox, [5, 10, 15])
prox_cercana = fuzz.trimf(x_prox, [0, 0, 10])

# Variable 6: Planes de evacuación (puntuación 0–10)
x_planes = np.arange(0, 11, 1)
plan_inexistente = fuzz.trimf(x_planes, [0, 0, 3])
plan_parcial = fuzz.trimf(x_planes, [2, 5, 7])
plan_completo = fuzz.trimf(x_planes, [6, 10, 10])

# Variable 7: Nivel de preparación (0–5)
x_preparacion = np.arange(0, 5.1, 0.1)
prep_bajo = fuzz.trimf(x_preparacion, [0, 0, 2])
prep_medio = fuzz.trimf(x_preparacion, [1.5, 2.5, 3.5])
prep_alto = fuzz.trimf(x_preparacion, [3, 5, 5])

# Variable 8: Densidad poblacional (hab/km²)
x_densidad = np.arange(0, 30001, 500)
dens_baja = fuzz.trimf(x_densidad, [0, 0, 8000])
dens_media = fuzz.trimf(x_densidad, [5000, 15000, 20000])
dens_alta = fuzz.trimf(x_densidad, [15000, 30000, 30000])

# Graficar
# graficar_fuzzy(x_prox, [prox_lejana, prox_media, prox_cercana], ['Lejana', 'Media', 'Cercana'], 'Proximidad a población', 'Distancia (km)')
# graficar_fuzzy(x_planes, [plan_inexistente, plan_parcial, plan_completo], ['Inexistente', 'Parcial', 'Completo'], 'Planes de evacuación', 'Puntuación')
# graficar_fuzzy(x_preparacion, [prep_bajo, prep_medio, prep_alto], ['Bajo', 'Medio', 'Alto'], 'Nivel de preparación', 'Puntuación')
# graficar_fuzzy(x_densidad, [dens_baja, dens_media, dens_alta], ['Baja', 'Media', 'Alta'], 'Densidad poblacional', 'hab/km²')