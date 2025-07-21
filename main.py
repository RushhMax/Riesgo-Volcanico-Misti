from red_bayesiana.red import TrueFuzzyBayesianNetwork

# Función principal de demostración
def demo_true_fuzzy_bayesian_network():
    """Demostración de la Red Bayesiana Difusa"""
    print("🌋" * 20)
    print("SISTEMA DE RED BAYESIANA DIFUSA")
    print("Evaluación de Riesgo Volcánico del Misti")
    print("🌋" * 20)
    
    # Crear la red
    fbn = TrueFuzzyBayesianNetwork()
    
    # Evidencia crisp de ejemplo
    evidence = {
        'sismicidad': 15,       # eventos/día
        'gases': 4000,          # ppm
        'deformacion': 30,      # mm
        'historia': 8,          # nivel histórico (0-10)
        'densidad': 10000,      # personas/km²
        'preparacion': 2,       # nivel de preparación (0-5)
        'proximidad': 10,       # km al cráter
        'evacuacion': 5         # nivel de evacuación (0-10)
    }
    
    print(f"\n📋 DATOS DE ENTRADA:")
    for var, val in evidence.items():
        print(f"   {var}: {val}")
    
    # Realizar inferencia difusa
    fuzzy_result = fbn.fuzzy_inference(evidence, 'riesgo', verbose=True)
    
    # Defuzzificar resultado
    crisp_risk = fbn.defuzzify_distribution(fuzzy_result, 'centroid')
    
    print(f"\n🔥 EVALUACIÓN FINAL DEL RIESGO:")
    print(f"   Valor difuso defuzzificado: {crisp_risk:.2f}/10")
    
    if crisp_risk <= 3:
        level = "🟢 BAJO"
        action = "Monitoreo rutinario"
    elif crisp_risk <= 6:
        level = "🟡 MEDIO" 
        action = "Alerta y preparación"
    else:
        level = "🔴 ALTO"
        action = "Evacuación recomendada"
    
    print(f"   Nivel de riesgo: {level}")
    print(f"   Acción recomendada: {action}")
    
    return fbn, fuzzy_result, crisp_risk

# Ejemplo de uso
if __name__ == "__main__":
    fbn, result, risk_value = demo_true_fuzzy_bayesian_network()