# from modelos.amenaza import calcular_amenaza
# from modelos.vulnerabilidad import calcular_vulnerabilidad
# from modelos.riesgo import calcular_riesgo

# # Ejemplo de valores de entrada
# sismicidad_val = 15         # eventos/día
# gases_val = 4000            # ppm
# deformacion_val = 30        # mm
# historia_val = 8            # nivel histórico de actividad (0–10)

# densidad_val = 10000        # personas/km²
# preparacion_val = 2         # 0–10 (mala a excelente)
# proximidad_val = 10         # km al cráter
# evacuacion_val = 5          # 0–10 (pobre a buena planificación)

# # Paso 1: Calcular amenaza difusa
# valor_amenaza = calcular_amenaza(
#     sis=sismicidad_val,
#     gas=gases_val,
#     deform=deformacion_val,
#     hist=historia_val
# )

# # Paso 2: Calcular vulnerabilidad difusa
# valor_vulnerabilidad = calcular_vulnerabilidad(
#     dens=densidad_val,
#     prep=preparacion_val,
#     prox=proximidad_val,
#     evac=evacuacion_val
# )

# # Paso 3: Evaluar el riesgo a partir de amenaza y vulnerabilidad
# valor_riesgo = calcular_riesgo(
#     valor_amenaza=valor_amenaza,
#     valor_vulnerabilidad=valor_vulnerabilidad
# )

# # Mostrar resultados
# print(f"Amenaza: {valor_amenaza:.2f}")
# print(f"Vulnerabilidad: {valor_vulnerabilidad:.2f}")
# print(f"Riesgo Volcánico Estimado: {valor_riesgo:.2f}")
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from modelos.bayes_volcan import crear_red_bayesiana, probabilidad_a_valor_difuso

if __name__ == '__main__':
    # Crear modelo
    modelo = crear_red_bayesiana()
    infer = VariableElimination(modelo)

    # Evidencia conocida
    evidencia = {
        'sismicidad': 2,     # 0=baja, 1=media, 2=alta
        'gases': 1,          # 0=normal, 1=media, 2=alta
        'deformacion': 0,    # 0=nula
        'historia': 1,       # 0=baja, 1=alta
        'densidad': 2,       # 0=baja, 1=media, 2=alta
        'preparacion': 0     # 0=baja preparación
    }

    # Inferir amenaza
    resultado_amenaza = infer.query(variables=['amenaza'], evidence=evidencia)
    print("Distribución de 'amenaza':")
    for i, etiqueta in enumerate(['baja', 'media', 'alta']):
        print(f"  {etiqueta}: {resultado_amenaza.values[i]:.2f}")
    valor_amenaza = probabilidad_a_valor_difuso(resultado_amenaza.values)
    print(f"\nNivel difuso de amenaza (esperado): {valor_amenaza:.2f}")

    # Inferir vulnerabilidad
    resultado_vulnerabilidad = infer.query(variables=['vulnerabilidad'], evidence=evidencia)
    print("\nDistribución de 'vulnerabilidad':")
    for i, etiqueta in enumerate(['baja', 'media', 'alta']):
        print(f"  {etiqueta}: {resultado_vulnerabilidad.values[i]:.2f}")
    valor_vulnerabilidad = probabilidad_a_valor_difuso(resultado_vulnerabilidad.values)
    print(f"\nNivel difuso de vulnerabilidad (esperado): {valor_vulnerabilidad:.2f}")