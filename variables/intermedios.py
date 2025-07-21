from graficar import graficar_fuzzy
import numpy as np
import skfuzzy as fuzz

# Variable intermedia: Amenaza volcánica (0–10)
x_amenaza = np.arange(0, 11, 1)
amenaza_baja = fuzz.trimf(x_amenaza, [0, 0, 4])
amenaza_media = fuzz.trimf(x_amenaza, [3, 5, 7])
amenaza_alta = fuzz.trimf(x_amenaza, [6, 10, 10])

# Graficar
graficar_fuzzy(x_amenaza, [amenaza_baja, amenaza_media, amenaza_alta], ['Baja', 'Media', 'Alta'], 'Amenaza volcánica', 'Nivel')


# Variable intermedia: Vulnerabilidad social (0–10)
x_vuln = np.arange(0, 11, 1)
vuln_baja = fuzz.trimf(x_vuln, [0, 0, 4])
vuln_media = fuzz.trimf(x_vuln, [3, 5, 7])
vuln_alta = fuzz.trimf(x_vuln, [6, 10, 10])

# Graficar
graficar_fuzzy(x_vuln, [vuln_baja, vuln_media, vuln_alta], ['Baja', 'Media', 'Alta'], 'Vulnerabilidad social', 'Nivel')

# Variable final: Riesgo volcánico (0–10)
x_riesgo = np.arange(0, 11, 1)
riesgo_bajo = fuzz.trimf(x_riesgo, [0, 0, 4])
riesgo_medio = fuzz.trimf(x_riesgo, [3, 5, 7])
riesgo_alto = fuzz.trimf(x_riesgo, [6, 10, 10])

# Graficar
graficar_fuzzy(x_riesgo, [riesgo_bajo, riesgo_medio, riesgo_alto], ['Bajo', 'Medio', 'Alto'], 'Riesgo volcánico', 'Nivel')
