from red_bayesiana.red import TrueFuzzyBayesianNetwork
import numpy as np

def evaluar_riesgo_volcanico(distrito):
    """Evalúa el riesgo volcánico para distritos específicos de Arequipa"""
    print("🌋" * 20)
    print(f"EVALUACIÓN DE RIESGO VOLCÁNICO - DISTRITO: {distrito['nombre']}")
    print(f"Altura: {distrito['altura_msnm']} msnm | Distancia al cráter: {distrito['distancia_km']} km")
    print("🌋" * 20)
    
    # Crear la red bayesiana
    fbn = TrueFuzzyBayesianNetwork()
    
    # Datos específicos del distrito
    print("\n📊 DATOS DE ENTRADA:")
    print(f"   Población: {distrito['poblacion']} hab")
    print(f"   Densidad: {distrito['densidad']} hab/km²")
    print(f"   Nivel de preparación: {distrito['preparacion']}/5")
    print(f"   Planes de evacuación: {distrito['evacuacion']}/10")
    
    # Valores actualizados del volcán (datos simulados)
    actividad_actual = {
        'sismicidad': np.random.randint(0, 20),  # eventos/día
        'gases': np.random.randint(3000, 5000),   # ppm SO2
        'deformacion': np.random.randint(30, 50)  # mm
    }
    
    # Evidencia completa
    evidence = {
        **actividad_actual,
        'historia': 9,  # El Misti tiene historial eruptivo frecuente
        'densidad': distrito['densidad'],
        'preparacion': distrito['preparacion'],
        'proximidad': distrito['distancia_km'],
        'evacuacion': distrito['evacuacion']
    }
    
    # Mostrar actividad volcánica actual
    print("\n⚡ ACTIVIDAD VOLCÁNICA ACTUAL:")
    for k, v in actividad_actual.items():
        print(f"   {k}: {v}")
    
    # Inferencia de riesgo
    fuzzy_result = fbn.fuzzy_inference(evidence, 'riesgo', verbose=True)
    crisp_risk = fbn.defuzzify_distribution(fuzzy_result, 'centroid')
    
    # Resultado detallado
    print("\n🔍 FACTORES CLAVE:")
    print(f"   Amenaza volcánica: {fbn.crisp_to_fuzzy_state('sismicidad', actividad_actual['sismicidad'])}")
    print(f"   Vulnerabilidad: {distrito['vulnerabilidad_descripcion']}")
    
    # Evaluación final
    print("\n🔥 EVALUACIÓN FINAL:")
    print(f"   Valor de riesgo: {crisp_risk:.2f}/10")
    
    if crisp_risk <= 3.5:
        level = "🟢 BAJO"
        action = "Monitoreo rutinario"
    elif crisp_risk <= 6.5:
        level = "🟡 MEDIO"
        action = "Alerta amarilla - Preparar protocolos"
    else:
        level = "🔴 ALTO"
        action = "Alerta roja - Evacuación inmediata"
    
    print(f"   Nivel: {level}")
    print(f"   Acción: {action}")
    
    # Recomendaciones específicas
    print("\n📌 RECOMENDACIONES:")
    if distrito['distancia_km'] < 10:
        print("   - Priorizar evacuación (zona de alto peligro)")
    if distrito['preparacion'] < 2.5:
        print("   - Implementar talleres de preparación comunitaria")
    if distrito['evacuacion'] < 5:
        print("   - Actualizar planes de evacuación con simulacros")
    
    return crisp_risk

# Datos reales de distritos de Arequipa
DISTRITOS = {
    "Alto Selva Alegre": {
        "nombre": "Alto Selva Alegre",
        "altura_msnm": 2450,
        "distancia_km": 8.5,
        "poblacion": 77264,
        "densidad": 4500,
        "preparacion": 1.8,
        "evacuacion": 4.2,
        "vulnerabilidad_descripcion": "Alta densidad cerca del volcán"
    },
    "Cayma": {
        "nombre": "Cayma",
        "altura_msnm": 2400,
        "distancia_km": 10.2,
        "poblacion": 90487,
        "densidad": 3800,
        "preparacion": 3.1,
        "evacuacion": 6.5,
        "vulnerabilidad_descripcion": "Zona residencial con moderada preparación"
    },
    "Yanahuara": {
        "nombre": "Yanahuara",
        "altura_msnm": 2300,
        "distancia_km": 9.8,
        "poblacion": 25685,
        "densidad": 5200,
        "preparacion": 2.5,
        "evacuacion": 5.8,
        "vulnerabilidad_descripcion": "Centro urbano denso"
    },
    "Chiguata": {
        "nombre": "Chiguata",
        "altura_msnm": 2910,
        "distancia_km": 15.3,
        "poblacion": 4123,
        "densidad": 800,
        "preparacion": 1.2,
        "evacuacion": 2.7,
        "vulnerabilidad_descripcion": "Zona rural con baja preparación"
    }
}

if __name__ == "__main__":
    print("ANÁLISIS DE RIESGO VOLCÁNICO - VOLCÁN MISTI\n")
    
    # Evaluar todos los distritos
    resultados = {}
    for nombre, datos in DISTRITOS.items():
        resultados[nombre] = evaluar_riesgo_volcanico(datos)
        print("\n" + "="*50 + "\n")
    
    # Mostrar resumen comparativo
    print("📊 RESUMEN COMPARATIVO DE RIESGO POR DISTRITO:")
    for distrito, riesgo in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        print(f"  {distrito:20}: {riesgo:.2f}/10 {'🔴' if riesgo >6.5 else '🟡' if riesgo >3.5 else '🟢'}")