#!/usr/bin/env python3
"""
Ejemplo de uso de la Red Bayesiana Difusa Mejorada
Demuestra las mejoras implementadas y el uso correcto de la red.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from red_bayesiana.red import TrueFuzzyBayesianNetwork
from red_bayesiana.triangular import TriangularFuzzyProbability

def main():
    """Función principal para demostrar el uso mejorado de la red"""
    
    print("🌋 DEMO: Red Bayesiana Difusa para Riesgo Volcánico - VERSIÓN MEJORADA")
    print("=" * 80)
    
    # Crear la red
    red = TrueFuzzyBayesianNetwork()
    
    # 1. Diagnóstico de la red
    print("\n📋 1. DIAGNÓSTICO DE LA RED:")
    issues = red.diagnose_network()
    if issues:
        print("⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ La red está correctamente configurada")
    
    # 2. Información de la red
    print("\n📊 2. INFORMACIÓN DE LA RED:")
    info = red.get_network_info()
    print(f"   Total de nodos: {info['total_nodes']}")
    print(f"   Nodos raíz: {info['root_nodes']}")
    print(f"   Nodos intermedios: {info['intermediate_nodes']}")
    print(f"   Nodos hoja: {info['leaf_nodes']}")
    print(f"   Total reglas CPD: {info['total_cpd_rules']}")
    
    # 3. Casos de prueba
    casos_prueba = [
        {
            'nombre': 'Escenario Alto Riesgo',
            'evidencia': {
                'sismicidad': 15,      # Alta actividad sísmica
                'gases': 3000,         # Emisiones elevadas
                'deformacion': 35,     # Deformación significativa
                'historia': 8,         # Historia alta
                'densidad': 15000,     # Alta densidad poblacional
                'preparacion': 1.5,    # Baja preparación
                'proximidad': 3,       # Muy cerca del volcán
                'evacuacion': 2        # Planes de evacuación deficientes
            }
        },
        {
            'nombre': 'Escenario Bajo Riesgo',
            'evidencia': {
                'sismicidad': 2,       # Baja actividad sísmica
                'gases': 500,          # Emisiones normales
                'deformacion': 2,      # Sin deformación
                'historia': 1,         # Historia baja
                'densidad': 3000,      # Baja densidad poblacional
                'preparacion': 4.5,    # Alta preparación
                'proximidad': 18,      # Lejos del volcán
                'evacuacion': 9        # Excelentes planes
            }
        },
        {
            'nombre': 'Escenario Mixto',
            'evidencia': {
                'sismicidad': 7,       # Actividad media
                'gases': 1500,         # Emisiones elevadas
                'deformacion': 12,     # Deformación leve
                'historia': 5,         # Historia media
                'densidad': 8000,      # Densidad media
                'preparacion': 3,      # Preparación media
                'proximidad': 8,       # Distancia media
                'evacuacion': 5        # Evacuación parcial
            }
        }
    ]
    
    print("\n🧪 3. PRUEBAS DE INFERENCIA:")
    
    for i, caso in enumerate(casos_prueba, 1):
        print(f"\n--- Caso {i}: {caso['nombre']} ---")
        
        try:
            # Realizar inferencia con validación mejorada
            resultado = red.fuzzy_inference(
                evidence_crisp=caso['evidencia'],
                target_variable='riesgo',
                verbose=False  # Cambiar a True para ver detalles
            )
            
            print("📈 Distribución difusa de riesgo:")
            for estado, prob_difusa in resultado['riesgo'].items():
                valor_crisp = prob_difusa.defuzzify_centroid()
                print(f"   {estado}: {prob_difusa} → crisp: {valor_crisp:.3f}")
            
            # Defuzzificar resultado final
            riesgo_crisp = red.defuzzify_distribution(resultado['riesgo'])
            print(f"🎯 Riesgo final (crisp): {riesgo_crisp:.2f}/10")
            
            # Normalizar distribución para verificar consistencia
            distribucion_norm = red._normalize_fuzzy_distribution(resultado['riesgo'])
            suma_centroides = sum(p.defuzzify_centroid() for p in distribucion_norm.values())
            print(f"✓ Suma de centroides normalizados: {suma_centroides:.3f}")
            
        except Exception as e:
            print(f"❌ Error en caso {i}: {str(e)}")
    
    # 4. Demostrar manejo de errores
    print(f"\n🔧 4. DEMOSTRACIÓN DE VALIDACIÓN:")
    
    try:
        # Intentar con datos inválidos
        red.fuzzy_inference({})
    except ValueError as e:
        print(f"✓ Validación exitosa - Error capturado: {e}")
    
    try:
        # Intentar con variable objetivo inexistente
        red.fuzzy_inference({'sismicidad': 5}, target_variable='inexistente')
    except ValueError as e:
        print(f"✓ Validación exitosa - Error capturado: {e}")
    
    try:
        # Intentar con valor no numérico
        red.fuzzy_inference({'sismicidad': 'texto'})
    except TypeError as e:
        print(f"✓ Validación exitosa - Error capturado: {e}")
    
    print(f"\n🎉 Demo completada exitosamente!")
    print("💡 Sugerencias:")
    print("   - Usar verbose=True para ver detalles de la inferencia")
    print("   - La red ahora maneja mejor los casos extremos")
    print("   - Se agregaron métodos de diagnóstico y validación")

if __name__ == "__main__":
    main()
