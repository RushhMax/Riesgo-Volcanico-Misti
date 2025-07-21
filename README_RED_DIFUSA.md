# 🌋 Sistema de Red Bayesiana Difusa para Riesgo Volcánico del Misti

## 📋 Descripción del Proyecto

Este proyecto implementa una **Red Bayesiana Difusa verdadera** para evaluar el riesgo volcánico del volcán Misti en Arequipa, Perú. A diferencia de los sistemas híbridos que combinan lógica difusa con redes bayesianas discretas, esta implementación utiliza números difusos triangulares en toda la estructura de la red.

## 🎯 Características Principales

### ✨ Red Bayesiana Difusa Verdadera
- **Números Difusos Triangulares**: Todos los nodos manejan distribuciones de probabilidad difusas
- **CPDs Difusas**: Tablas de probabilidad condicional con números difusos
- **Inferencia Difusa**: Propagación de incertidumbre mediante aritmética difusa
- **Interpolación Inteligente**: Manejo de combinaciones no definidas explícitamente

### 🧮 Aritmética Difusa Completa
- Suma, multiplicación y división de números difusos triangulares
- Operaciones de membresía y alpha-cuts
- Múltiples métodos de defuzzificación (centroide, mean-of-max)
- Cálculo de centroides para defuzzificación

### 🔗 Estructura de la Red
```
Variables de Entrada:
├── Sismicidad (eventos/día)
├── Gases (ppm)
├── Deformación (mm)
├── Historia (nivel 0-10)
├── Densidad Poblacional (hab/km²)
├── Preparación (nivel 0-5)
├── Proximidad (km al cráter)
└── Evacuación (nivel 0-10)

Variables Intermedias:
├── Amenaza (baja, media, alta)
└── Vulnerabilidad (baja, media, alta)

Variable Objetivo:
└── Riesgo (bajo, medio, alto)
```

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
pip install -r requirements.txt
```

### Dependencias Principales
- `scikit-fuzzy`: Sistemas de inferencia difusa
- `pgmpy`: Redes bayesianas (para comparación)
- `numpy`: Computación numérica
- `matplotlib`: Visualización

### Ejecución Rápida
```bash
# Activar entorno virtual
.venv\Scripts\activate

# Ejecutar demostración básica
python main.py

# Ejecutar demostración completa
python demo_red_difusa.py
```

## 📁 Estructura del Proyecto

```
Riesgo-Volcanico-Misti/
├── main.py                          # Implementación principal de Red Bayesiana Difusa
├── triangular.py                    # Clase para números difusos triangulares
├── demo_red_difusa.py               # Demostración completa del sistema
├── requirements.txt                 # Dependencias del proyecto
├── .gitignore                       # Configuración de Git
├── README_RED_DIFUSA.md             # Documentación de la Red Bayesiana Difusa
├── CONVERSION_COMPLETADA.md         # Resumen de la conversión realizada
└── ANALISIS_TECNICO.md              # Análisis técnico detallado
```

## 🔬 Componentes Técnicos

### 1. Clase `TriangularFuzzyProbability`
```python
# Ejemplo de número difuso triangular
fuzzy_prob = TriangularFuzzyProbability(0.6, 0.75, 0.9)
print(fuzzy_prob)  # (0.600, 0.750, 0.900)

# Operaciones aritméticas
suma = fuzzy_prob1 + fuzzy_prob2
producto = fuzzy_prob1 * fuzzy_prob2

# Defuzzificación
crisp_value = fuzzy_prob.defuzzify_centroid()
```

### 2. Clase `TrueFuzzyBayesianNetwork`
```python
# Crear red
fbn = TrueFuzzyBayesianNetwork()

# Evidencia de entrada
evidence = {
    'sismicidad': 15,    # eventos/día
    'gases': 4000,       # ppm
    'deformacion': 30,   # mm
    'historia': 8,       # nivel histórico
    'densidad': 10000,   # hab/km²
    'preparacion': 2,    # nivel preparación
    'proximidad': 10,    # km al cráter
    'evacuacion': 5      # nivel evacuación
}

# Realizar inferencia difusa
fuzzy_result = fbn.fuzzy_inference(evidence, 'riesgo', verbose=True)
crisp_risk = fbn.defuzzify_distribution(fuzzy_result, 'centroid')
```

### 3. Nodos y CPDs Difusas
- **Nodos Padre**: Distribuciones a priori difusas
- **Nodos Hijo**: CPDs con reglas difusas expertas
- **Interpolación**: Manejo automático de casos no definidos
- **Propagación**: Inferencia hacia adelante con números difusos

## 📊 Ejemplos de Uso

### Escenario Crítico
```python
evidence_critico = {
    'sismicidad': 18,     # Alta actividad sísmica
    'gases': 4500,        # Emisiones muy altas
    'deformacion': 40,    # Deformación significativa
    'historia': 9,        # Alta actividad histórica
    'densidad': 25000,    # Población muy densa
    'preparacion': 1,     # Muy poca preparación
    'proximidad': 3,      # Muy cerca del volcán
    'evacuacion': 2       # Evacuación inadecuada
}

result = fbn.fuzzy_inference(evidence_critico, 'riesgo')
# Resultado: Distribución difusa de riesgo con alta incertidumbre
```

### Análisis de Sensibilidad
```python
# Variar parámetros y observar cambios en el riesgo
for sismicidad in range(1, 20, 3):
    evidence['sismicidad'] = sismicidad
    fuzzy_result = fbn.fuzzy_inference(evidence, 'riesgo')
    crisp_risk = fbn.defuzzify_distribution(fuzzy_result)
    print(f"Sismicidad {sismicidad} → Riesgo {crisp_risk:.2f}")
```

## 🎨 Funcionalidades Avanzadas

### 1. Múltiples Métodos de Defuzzificación
- **Centroide**: Valor medio ponderado
- **Mean-of-Max**: Promedio de valores con máxima membresía

### 2. Interpolación Difusa
- Manejo automático de reglas no definidas
- Búsqueda de reglas similares
- Expansión de incertidumbre para interpolación

### 3. Propagación de Incertidumbre
- Conservación de información difusa en toda la red
- Cálculo de distribuciones difusas completas
- Análisis de sensibilidad automático

## 📈 Resultados y Validación

### Escenarios de Prueba
- **🟢 Tranquilo**: Condiciones normales → Riesgo Medio
- **🟡 Moderado**: Actividad media → Riesgo Medio
- **🔴 Crítico**: Condiciones severas → Riesgo Alto
- **⚡ Mixto**: Combinación variada → Riesgo Variable

### Métricas de Evaluación
- Varianza de riesgo por variable
- Rango de variación
- Consistencia entre métodos de defuzzificación
- Robustez ante datos faltantes

## 🔧 Desarrollo y Extensión

### Agregar Nuevas Variables
1. Definir nuevo nodo en `_create_network()`
2. Configurar distribución a priori en `_setup_fuzzy_distributions()`
3. Añadir reglas expertas en CPDs correspondientes
4. Actualizar sistema de mapeo crisp-to-linguistic

### Nuevas Reglas Difusas
```python
# Ejemplo: Nueva regla para amenaza
fuzzy_rules = {
    ('nueva_condicion', 'gases', 'deformacion', 'historia'): {
        'baja': TriangularFuzzyProbability(0.1, 0.2, 0.3),
        'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
        'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
    }
}
```

## 🏆 Ventajas del Sistema

### Vs. Sistemas Híbridos
- ✅ Manejo nativo de incertidumbre difusa
- ✅ No pérdida de información en discretización
- ✅ Propagación coherente de incertidumbre
- ✅ Interpretabilidad mejorada de resultados

### Vs. Sistemas Clásicos
- ✅ Manejo natural de imprecisión
- ✅ Incorporación de conocimiento experto
- ✅ Robustez ante datos incompletos
- ✅ Análisis de sensibilidad integrado

## 🚨 Limitaciones y Consideraciones

### Limitaciones Actuales
- Reglas limitadas para algunas combinaciones
- Dependencia de conocimiento experto para CPDs
- Complejidad computacional mayor que sistemas discretos

### Trabajo Futuro
- Expansión de base de reglas
- Optimización de algoritmos de inferencia
- Integración con datos históricos reales
- Validación con expertos vulcanólogos

## 👥 Contribución

### Cómo Contribuir
1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Áreas de Contribución
- Mejora de reglas difusas
- Optimización de algoritmos
- Documentación y ejemplos
- Validación con datos reales

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones relacionadas con este sistema de Red Bayesiana Difusa para evaluación de riesgo volcánico.

---

**🌋 Sistema desarrollado para la evaluación de riesgo volcánico del Misti con tecnología de Red Bayesiana Difusa verdadera 🌋**
