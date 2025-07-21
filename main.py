from red_bayesiana.red import TrueFuzzyBayesianNetwork
import numpy as np

# Rangos exactos según especificación
PARAMETROS = {
    'sismicidad': {'min': 0, 'max': 20, 'unidad': 'eventos/día'},
    'gases': {'min': 0, 'max': 5000, 'unidad': 'ppm'},
    'deformacion': {'min': 0, 'max': 50, 'unidad': 'mm'},
    'historia': {'min': 0, 'max': 10, 'unidad': 'nivel'},
    'densidad': {'min': 0, 'max': 30000, 'unidad': 'personas/km²'},
    'preparacion': {'min': 0, 'max': 5, 'unidad': 'nivel'},
    'proximidad': {'min': 0, 'max': 20, 'unidad': 'km'},
    'evacuacion': {'min': 0, 'max': 10, 'unidad': 'nivel'}
}

def evaluar_riesgo_volcanico(distrito):
    """Evalúa el riesgo volcánico con rangos estandarizados"""
    print("🌋" * 20)
    print(f"EVALUACIÓN DE RIESGO - {distrito['nombre'].upper()}")
    print("🌋" * 20)
    
    # Crear red bayesiana con parámetros estandarizados
    fbn = TrueFuzzyBayesianNetwork()
    
    # Mostrar parámetros de referencia
    print("\n📐 PARÁMETROS DE REFERENCIA:")
    for param, info in PARAMETROS.items():
        print(f"   {param.capitalize():<12}: {info['min']}-{info['max']} {info['unidad']}")
    
    # Datos actualizados del volcán (simulados dentro de los rangos)
    actividad_actual = {
        'sismicidad': np.clip(np.random.normal(12, 4), PARAMETROS['sismicidad']['min'], PARAMETROS['sismicidad']['max']),
        'gases': np.clip(np.random.normal(3500, 800), PARAMETROS['gases']['min'], PARAMETROS['gases']['max']),
        'deformacion': np.clip(np.random.normal(35, 10), PARAMETROS['deformacion']['min'], PARAMETROS['deformacion']['max'])
    }
    
    # Evidencia completa con rangos validados
    evidence = {
        **actividad_actual,
        'historia': min(distrito['historia'], PARAMETROS['historia']['max']),
        'densidad': min(distrito['densidad'], PARAMETROS['densidad']['max']),
        'preparacion': min(distrito['preparacion'], PARAMETROS['preparacion']['max']),
        'proximidad': min(distrito['proximidad'], PARAMETROS['proximidad']['max']),
        'evacuacion': min(distrito['evacuacion'], PARAMETROS['evacuacion']['max'])
    }
    
    # Mostrar datos de entrada normalizados
    print("\n📊 DATOS NORMALIZADOS:")
    for k, v in evidence.items():
        print(f"   {k.capitalize():<12}: {v:.2f} {PARAMETROS[k]['unidad']}")
    
    # Inferencia de riesgo
    fuzzy_result = fbn.fuzzy_inference(evidence, 'riesgo', verbose=False)
    crisp_risk = fbn.defuzzify_distribution(fuzzy_result, 'centroid')
    
    # Resultado detallado
    print("\n🔍 INTERPRETACIÓN:")
    print(f"   Sismicidad: {v} ({fbn.crisp_to_fuzzy_state('sismicidad', actividad_actual['sismicidad'])})")
    print(f"   Proximidad: {distrito['proximidad']} km ({fbn.crisp_to_fuzzy_state('proximidad', distrito['proximidad'])})")
    print(f"   Preparación: {distrito['preparacion']}/5 ({fbn.crisp_to_fuzzy_state('preparacion', distrito['preparacion'])})")
    
    # Evaluación final con umbrales ajustados
    print("\n🔥 RESULTADO:")
    riesgo_nivel = "ALTO" if crisp_risk > 7 else "MEDIO" if crisp_risk > 4 else "BAJO"
    color = "🔴" if crisp_risk > 7 else "🟡" if crisp_risk > 4 else "🟢"
    print(f"   Riesgo calculado: {crisp_risk:.2f}/10 {color} ({riesgo_nivel})")
    
    # Recomendaciones específicas por rango
    print("\n📌 RECOMENDACIONES:")
    if crisp_risk > 7:
        print("   - Evacuación inmediata de zonas críticas")
        print("   - Activación de albergues temporales")
    elif crisp_risk > 4:
        print("   - Simulacro de evacuación obligatorio")
        print("   - Revisión de rutas de escape")
    else:
        print("   - Monitoreo continuo")
        print("   - Talleres de preparación comunal")
    
    return crisp_risk


DISTRITOS = {
    "Alto Selva Alegre": {
        "nombre": "Alto Selva Alegre",
        "historia": 9,
        "densidad": 14500,
        "preparacion": 1.8,
        "proximidad": 8.5,
        "evacuacion": 4.2
    },
    "Miraflores": {
        "nombre": "Miraflores",
        "historia": 6,
        "densidad": 22000,
        "preparacion": 4.0,
        "proximidad": 12.1,
        "evacuacion": 7.8
    },
    "Mariano Melgar": {
        "nombre": "Mariano Melgar",
        "historia": 8,
        "densidad": 17500,
        "preparacion": 2.0,
        "proximidad": 9.0,
        "evacuacion": 5.0
    },
    "Paucarpata": {
        "nombre": "Paucarpata",
        "historia": 7,
        "densidad": 16000,
        "preparacion": 3.2,
        "proximidad": 10.0,
        "evacuacion": 6.2
    },
    "Cercado de Arequipa": {
        "nombre": "Cercado de Arequipa",
        "historia": 10,
        "densidad": 25000,
        "preparacion": 2.8,
        "proximidad": 7.5,
        "evacuacion": 7.0
    },
    "Cayma": {
        "nombre": "Cayma",
        "historia": 8,
        "densidad": 9800,
        "preparacion": 3.1,
        "proximidad": 10.2,
        "evacuacion": 6.5
    },
    "Sachaca": {
        "nombre": "Sachaca",
        "historia": 6,
        "densidad": 5200,
        "preparacion": 2.4,
        "proximidad": 11.8,
        "evacuacion": 5.0
    },
    "Jacobo Hunter": {
        "nombre": "Jacobo Hunter",
        "historia": 5,
        "densidad": 7300,
        "preparacion": 3.0,
        "proximidad": 13.1,
        "evacuacion": 4.6
    },
    "Chiguata": {
        "nombre": "Chiguata",
        "historia": 5,
        "densidad": 800,
        "preparacion": 1.2,
        "proximidad": 15.3,
        "evacuacion": 2.7
    },
    "Characato": {
        "nombre": "Characato",
        "historia": 4,
        "densidad": 3100,
        "preparacion": 2.1,
        "proximidad": 16.7,
        "evacuacion": 3.3
    },
    "Socabaya": {
        "nombre": "Socabaya",
        "historia": 6,
        "densidad": 4100,
        "preparacion": 3.4,
        "proximidad": 14.5,
        "evacuacion": 5.6
    }
}


if __name__ == "__main__":
    print("SISTEMA DE ALERTA TEMPRANA VOLCÁN MISTI\n")
    print(f"🔹 Rango sismicidad: {PARAMETROS['sismicidad']['min']}-{PARAMETROS['sismicidad']['max']} {PARAMETROS['sismicidad']['unidad']}")
    print(f"🔹 Rango gases: {PARAMETROS['gases']['min']}-{PARAMETROS['gases']['max']} {PARAMETROS['gases']['unidad']}")
    print(f"🔹 Máxima deformación: {PARAMETROS['deformacion']['max']} {PARAMETROS['deformacion']['unidad']}\n")
    
    # Evaluar todos los distritos
    resultados = {}
    for nombre, datos in DISTRITOS.items():
        resultados[nombre] = evaluar_riesgo_volcanico(datos)
        print("\n" + "="*50 + "\n")
    
    # Resumen comparativo
    print("📈 RESUMEN COMPARATIVO:")
    print(f"{'Distrito':<20} | {'Riesgo':<6} | {'Prox (km)':<9} | {'Densidad':<9} | Prep.")
    print("-"*60)
    for distrito, riesgo in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        datos = DISTRITOS[distrito]
        print(f"{distrito:<20} | {riesgo:.2f}/10 | {datos['proximidad']:>6.1f} km | {datos['densidad']:>8} | {datos['preparacion']}/5")


import matplotlib.pyplot as plt

# 1. Gráfico de barras de riesgo por distrito
def graficar_riesgo_distritos(resultados):
    nombres = list(resultados.keys())
    valores = list(resultados.values())

    plt.figure(figsize=(10, 6))
    barras = plt.bar(nombres, valores, color=[
        'red' if v > 7 else 'orange' if v > 4 else 'green' for v in valores
    ])
    plt.ylim(0, 10)
    plt.title("Nivel de Riesgo Volcánico por Distrito")
    plt.ylabel("Riesgo (0 a 10)")
    plt.xlabel("Distrito")

    # Añadir etiquetas de valor
    for bar in barras:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# 2. Radar chart de un distrito (comparación de variables)
def graficar_radar_distrito(nombre, datos):
    etiquetas = ['historia', 'densidad', 'preparacion', 'proximidad', 'evacuacion']
    valores = [datos[e] for e in etiquetas]

    # Normalizar valores (entre 0 y 1)
    normalizados = [
        valores[i] / PARAMETROS[etiquetas[i]]['max'] for i in range(len(etiquetas))
    ]
    normalizados += [normalizados[0]]  # Cerrar figura

    etiquetas_legibles = [e.capitalize() for e in etiquetas]
    etiquetas_legibles += [etiquetas_legibles[0]]

    angulos = np.linspace(0, 2 * np.pi, len(etiquetas_legibles), endpoint=True)

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})
    ax.plot(angulos, normalizados, marker='o')
    ax.fill(angulos, normalizados, alpha=0.25)
    ax.set_thetagrids(angles=np.degrees(angulos), labels=etiquetas_legibles)
    ax.set_title(f"Perfil del distrito: {nombre}", size=14)
    plt.show()


# 3. Ejecutar gráficos
graficar_riesgo_distritos(resultados)

# (Opcional) Mostrar radar para los 2 más riesgosos
top2 = sorted(resultados.items(), key=lambda x: x[1], reverse=True)[:2]
for nombre, _ in top2:
    graficar_radar_distrito(nombre, DISTRITOS[nombre])