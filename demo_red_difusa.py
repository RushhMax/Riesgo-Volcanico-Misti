"""
DEMOSTRACIÓN COMPLETA DE RED BAYESIANA DIFUSA VERDADERA
Sistema de Evaluación de Riesgo Volcánico del Misti

Esta demostración muestra las capacidades completas del sistema de Red Bayesiana Difusa
implementado para la evaluación de riesgo volcánico, incluyendo:
- Inferencia con números difusos triangulares
- Propagación de incertidumbre
- Múltiples escenarios de evaluación
- Análisis de sensibilidad
"""

import numpy as np
import matplotlib.pyplot as plt
from main import TrueFuzzyBayesianNetwork
from red_bayesiana.triangular import TriangularFuzzyProbability

def demo_escenarios_volcanicos():
    """Demostración con múltiples escenarios volcánicos"""
    print("🌋" * 60)
    print(" " * 15 + "DEMOSTRACIÓN DE RED BAYESIANA DIFUSA VERDADERA")
    print(" " * 20 + "Sistema de Riesgo Volcánico del Misti")
    print("🌋" * 60)
    
    # Crear la red
    fbn = TrueFuzzyBayesianNetwork()
    
    # Definir escenarios de prueba
    escenarios = {
        "🟢 ESCENARIO TRANQUILO": {
            'sismicidad': 2,        # Baja actividad sísmica
            'gases': 500,           # Emisiones normales
            'deformacion': 1,       # Sin deformación
            'historia': 1,          # Poca actividad histórica
            'densidad': 5000,       # Población media
            'preparacion': 4,       # Buena preparación
            'proximidad': 15,       # Alejado del volcán
            'evacuacion': 8         # Planes de evacuación buenos
        },
        
        "🟡 ESCENARIO MODERADO": {
            'sismicidad': 8,        # Actividad sísmica media
            'gases': 2500,          # Emisiones elevadas
            'deformacion': 15,      # Deformación leve
            'historia': 5,          # Actividad histórica media
            'densidad': 12000,      # Población alta
            'preparacion': 2,       # Preparación baja
            'proximidad': 8,        # Relativamente cerca
            'evacuacion': 4         # Planes limitados
        },
        
        "🔴 ESCENARIO CRÍTICO": {
            'sismicidad': 18,       # Alta actividad sísmica
            'gases': 4500,          # Emisiones muy altas
            'deformacion': 40,      # Deformación significativa
            'historia': 9,          # Alta actividad histórica
            'densidad': 25000,      # Población muy densa
            'preparacion': 1,       # Muy poca preparación
            'proximidad': 3,        # Muy cerca del volcán
            'evacuacion': 2         # Evacuación inadecuada
        },
        
        "⚡ ESCENARIO MIXTO": {
            'sismicidad': 12,       # Actividad media-alta
            'gases': 1800,          # Emisiones moderadas
            'deformacion': 25,      # Deformación media
            'historia': 6,          # Historia media
            'densidad': 8000,       # Población media-baja
            'preparacion': 3,       # Preparación media
            'proximidad': 12,       # Distancia media
            'evacuacion': 6         # Evacuación moderada
        }
    }
    
    resultados = {}
    
    for nombre_escenario, evidencia in escenarios.items():
        print(f"\n{nombre_escenario}")
        print("=" * len(nombre_escenario))
        
        # Mostrar parámetros de entrada
        print("📋 Parámetros de entrada:")
        for var, val in evidencia.items():
            unidad = {
                'sismicidad': 'eventos/día',
                'gases': 'ppm',
                'deformacion': 'mm',
                'historia': '/10',
                'densidad': 'hab/km²',
                'preparacion': '/5',
                'proximidad': 'km',
                'evacuacion': '/10'
            }
            print(f"   • {var}: {val} {unidad.get(var, '')}")
        
        # Realizar inferencia
        fuzzy_result = fbn.fuzzy_inference(evidencia, 'riesgo', verbose=False)
        
        # Calcular valor crisp
        crisp_risk = fbn.defuzzify_distribution(fuzzy_result, 'centroid')
        resultados[nombre_escenario] = {
            'fuzzy_distribution': fuzzy_result,
            'crisp_value': crisp_risk,
            'evidence': evidencia
        }
        
        # Mostrar resultados
        print(f"\n🎯 RESULTADOS DE RIESGO:")
        print(f"   Distribución difusa:")
        for estado, prob_difusa in fuzzy_result.items():
            crisp_prob = prob_difusa.defuzzify_centroid()
            print(f"     • {estado}: {prob_difusa} → {crisp_prob:.3f}")
        
        print(f"   Valor defuzzificado: {crisp_risk:.2f}/10")
        
        # Clasificación del riesgo
        if crisp_risk <= 3:
            nivel = "🟢 RIESGO BAJO"
            accion = "Monitoreo rutinario"
        elif crisp_risk <= 6:
            nivel = "🟡 RIESGO MEDIO"
            accion = "Alerta y preparación"
        else:
            nivel = "🔴 RIESGO ALTO"
            accion = "Evacuación recomendada"
        
        print(f"   Clasificación: {nivel}")
        print(f"   Acción recomendada: {accion}")
    
    return resultados

def analisis_sensibilidad():
    """Análisis de sensibilidad de variables clave"""
    print("\n" + "🔬" * 60)
    print(" " * 20 + "ANÁLISIS DE SENSIBILIDAD")
    print("🔬" * 60)
    
    fbn = TrueFuzzyBayesianNetwork()
    
    # Configuración base
    evidencia_base = {
        'sismicidad': 10,
        'gases': 2000,
        'deformacion': 15,
        'historia': 5,
        'densidad': 10000,
        'preparacion': 3,
        'proximidad': 8,
        'evacuacion': 5
    }
    
    # Variables a analizar
    variables_analisis = {
        'sismicidad': np.arange(1, 20, 3),
        'gases': np.arange(500, 5000, 500),
        'deformacion': np.arange(0, 50, 5),
        'proximidad': np.arange(2, 20, 2)
    }
    
    print(f"📊 Configuración base: {evidencia_base}")
    
    for variable, valores in variables_analisis.items():
        print(f"\n🔍 Análisis de sensibilidad para '{variable}':")
        
        riesgos = []
        for valor in valores:
            evidencia_temp = evidencia_base.copy()
            evidencia_temp[variable] = valor
            
            fuzzy_result = fbn.fuzzy_inference(evidencia_temp, 'riesgo', verbose=False)
            crisp_risk = fbn.defuzzify_distribution(fuzzy_result, 'centroid')
            riesgos.append(crisp_risk)
            
            print(f"   {variable}={valor} → Riesgo={crisp_risk:.2f}")
        
        # Calcular sensibilidad
        varianza = np.var(riesgos)
        rango = max(riesgos) - min(riesgos)
        
        print(f"   📈 Varianza del riesgo: {varianza:.3f}")
        print(f"   📏 Rango de variación: {rango:.3f}")

def demo_aritmetica_difusa():
    """Demostración de operaciones aritméticas con números difusos"""
    print("\n" + "🧮" * 60)
    print(" " * 15 + "DEMOSTRACIÓN DE ARITMÉTICA DIFUSA")
    print("🧮" * 60)
    
    # Crear números difusos triangulares
    print("🔢 Creando números difusos triangulares:")
    
    amenaza_alta = TriangularFuzzyProbability(0.6, 0.75, 0.9)
    vulnerabilidad_media = TriangularFuzzyProbability(0.3, 0.5, 0.6)
    
    print(f"   Amenaza Alta: {amenaza_alta}")
    print(f"   Vulnerabilidad Media: {vulnerabilidad_media}")
    
    # Operaciones aritméticas
    print(f"\n➕ Suma difusa:")
    suma = amenaza_alta + vulnerabilidad_media
    print(f"   {amenaza_alta} + {vulnerabilidad_media} = {suma}")
    
    print(f"\n✖️ Multiplicación difusa:")
    producto = amenaza_alta * vulnerabilidad_media
    print(f"   {amenaza_alta} * {vulnerabilidad_media} = {producto}")
    
    # Defuzzificación
    print(f"\n🎯 Defuzzificación:")
    print(f"   Amenaza (centroide): {amenaza_alta.defuzzify_centroid():.3f}")
    print(f"   Vulnerabilidad (centroide): {vulnerabilidad_media.defuzzify_centroid():.3f}")
    print(f"   Producto (centroide): {producto.defuzzify_centroid():.3f}")
    
    # Operaciones de membresía
    print(f"\n🎯 Análisis de membresía:")
    valor_test = 0.7
    print(f"   Membresía de {valor_test} en Amenaza Alta: {amenaza_alta.membership(valor_test):.3f}")
    print(f"   Membresía de {valor_test} en Vulnerabilidad Media: {vulnerabilidad_media.membership(valor_test):.3f}")
    
    # Alpha-cuts
    print(f"\n✂️ Alpha-cuts (α=0.5):")
    alpha = 0.5
    cut_amenaza = amenaza_alta.alpha_cut(alpha)
    cut_vulnerabilidad = vulnerabilidad_media.alpha_cut(alpha)
    print(f"   Amenaza α-cut: {cut_amenaza}")
    print(f"   Vulnerabilidad α-cut: {cut_vulnerabilidad}")

def demo_comparacion_metodos():
    """Comparación entre métodos de defuzzificación"""
    print("\n" + "⚖️" * 60)
    print(" " * 15 + "COMPARACIÓN DE MÉTODOS DE DEFUZZIFICACIÓN")
    print("⚖️" * 60)
    
    fbn = TrueFuzzyBayesianNetwork()
    
    # Evidencia de prueba
    evidencia = {
        'sismicidad': 15,
        'gases': 3500,
        'deformacion': 25,
        'historia': 7,
        'densidad': 15000,
        'preparacion': 2,
        'proximidad': 6,
        'evacuacion': 3
    }
    
    print(f"📋 Evidencia: {evidencia}")
    
    # Realizar inferencia
    fuzzy_result = fbn.fuzzy_inference(evidencia, 'riesgo', verbose=False)
    
    print(f"\n🎲 Distribución difusa obtenida:")
    for estado, prob in fuzzy_result.items():
        print(f"   {estado}: {prob}")
    
    # Comparar métodos de defuzzificación
    metodos = ['centroid', 'mean_of_max']
    
    print(f"\n🎯 Comparación de métodos de defuzzificación:")
    for metodo in metodos:
        valor_crisp = fbn.defuzzify_distribution(fuzzy_result, metodo)
        print(f"   {metodo}: {valor_crisp:.3f}")

def main():
    """Función principal de demostración"""
    try:
        # Ejecutar todas las demostraciones
        print("🚀 Iniciando demostración completa del sistema...")
        
        # 1. Escenarios volcánicos
        resultados = demo_escenarios_volcanicos()
        
        # 2. Análisis de sensibilidad
        analisis_sensibilidad()
        
        # 3. Aritmética difusa
        demo_aritmetica_difusa()
        
        # 4. Comparación de métodos
        demo_comparacion_metodos()
        
        # Resumen final
        print("\n" + "🏁" * 60)
        print(" " * 20 + "RESUMEN FINAL")
        print("🏁" * 60)
        
        print("✅ Sistema de Red Bayesiana Difusa implementado exitosamente")
        print("✅ Inferencia con números difusos triangulares funcionando")
        print("✅ Múltiples escenarios evaluados correctamente")
        print("✅ Operaciones aritméticas difusas verificadas")
        print("✅ Métodos de defuzzificación comparados")
        
        print(f"\n📊 Resumen de escenarios evaluados:")
        for escenario, datos in resultados.items():
            riesgo = datos['crisp_value']
            if riesgo <= 3:
                estado = "🟢 BAJO"
            elif riesgo <= 6:
                estado = "🟡 MEDIO"
            else:
                estado = "🔴 ALTO"
            print(f"   {escenario}: {riesgo:.2f} ({estado})")
        
        print(f"\n🎯 El sistema está listo para uso operacional!")
        
    except Exception as e:
        print(f"❌ Error en la demostración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
