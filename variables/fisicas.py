import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from graficar import graficar_fuzzy

# Variable 1: Sismicidad (eventos por día)
x_sismicidad = np.arange(0, 21, 1)

sism_baja = fuzz.trimf(x_sismicidad, [0, 0, 5])
sism_moderada = fuzz.trimf(x_sismicidad, [3, 8, 12])
sism_alta = fuzz.trimf(x_sismicidad, [10, 20, 20])

# Variable 2: Emisión de gases (ppm)
x_gases = np.arange(0, 5001, 100)
gases_normal = fuzz.trimf(x_gases, [0, 0, 1500])
gases_elevada = fuzz.trimf(x_gases, [1000, 5000, 5000])

# Variable 3: Deformación del suelo (mm)
x_deform = np.arange(0, 51, 1)
def_nula = fuzz.trimf(x_deform, [0, 0, 5])
def_leve = fuzz.trimf(x_deform, [3, 20, 30])
def_signif = fuzz.trimf(x_deform, [25, 50, 50])

# Variable 4: Actividad histórica (índice 0–10)
x_hist = np.arange(0, 11, 1)
hist_baja = fuzz.trimf(x_hist, [0, 0, 2])
hist_moderada = fuzz.trimf(x_hist, [1, 5, 7])
hist_alta = fuzz.trimf(x_hist, [6, 10, 10])

# Mostrar las gráficas
graficar_fuzzy(x_sismicidad, [sism_baja, sism_moderada, sism_alta], ['Baja', 'Moderada', 'Alta'], 'Sismicidad', 'Eventos/día')
graficar_fuzzy(x_gases, [gases_normal, gases_elevada], ['Normal', 'Elevada'], 'Emisión de gases', 'ppm SO2')
graficar_fuzzy(x_deform, [def_nula, def_leve, def_signif], ['Nula', 'Leve', 'Significativa'], 'Deformación del suelo', 'mm')
graficar_fuzzy(x_hist, [hist_baja, hist_moderada, hist_alta], ['Baja', 'Moderada', 'Alta'], 'Actividad histórica', 'Índice')
