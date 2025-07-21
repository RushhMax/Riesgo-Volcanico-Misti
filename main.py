from modelos.amenaza import calcular_amenaza
from modelos.vulnerabilidad import calcular_vulnerabilidad
from modelos.riesgo import calcular_riesgo

# Ejemplo de valores de entrada
sismicidad_val = 15         # eventos/día
gases_val = 4000            # ppm
deformacion_val = 30        # mm
historia_val = 8            # nivel histórico de actividad (0–10)

densidad_val = 10000        # personas/km²
preparacion_val = 2         # 0–10 (mala a excelente)
proximidad_val = 10         # km al cráter
evacuacion_val = 5          # 0–10 (pobre a buena planificación)

# Paso 1: Calcular amenaza difusa
valor_amenaza = calcular_amenaza(
    sis=sismicidad_val,
    gas=gases_val,
    deform=deformacion_val,
    hist=historia_val
)

# Paso 2: Calcular vulnerabilidad difusa
valor_vulnerabilidad = calcular_vulnerabilidad(
    dens=densidad_val,
    prep=preparacion_val,
    prox=proximidad_val,
    evac=evacuacion_val
)

# Paso 3: Evaluar el riesgo a partir de amenaza y vulnerabilidad
valor_riesgo = calcular_riesgo(
    amenaza=valor_amenaza,
    vulnerabilidad=valor_vulnerabilidad
)

# Mostrar resultados
print(f"Amenaza: {valor_amenaza:.2f}")
print(f"Vulnerabilidad: {valor_vulnerabilidad:.2f}")
print(f"Riesgo Volcánico Estimado: {valor_riesgo:.2f}")
