# Sistema de Evaluación de Riesgo Volcánico del Misti

Este proyecto implementa un sistema inteligente para evaluar el riesgo volcánico del volcán Misti utilizando múltiples enfoques de inteligencia artificial: **lógica difusa**, **redes bayesianas discretas** y un **sistema híbrido difuso-bayesiano**.

⚠️ **Nota Técnica**: El proyecto combina sistemas difusos con redes bayesianas clásicas, pero NO implementa una verdadera Red Bayesiana Difusa (donde los nodos contienen variables difusas y las CPDs manejan números difusos).

## 🌋 Descripción del Proyecto

El sistema evalúa el riesgo volcánico considerando dos componentes principales:
- **Amenaza Volcánica**: Basada en sismicidad, emisión de gases, deformación del terreno e historia volcánica
- **Vulnerabilidad**: Basada en densidad poblacional, nivel de preparación, proximidad al volcán y planificación de evacuación

## 📁 Estructura del Proyecto

```
Riesgo-Volcanico-Misti/
├── main.py                              # Sistema híbrido difuso-bayesiano
├── demo_completo.py                     # Demostración de todos los enfoques
├── red_bayesiana_difusa_verdadera.py   # Implementación de RBD auténtica
├── requirements.txt                     # Dependencias del proyecto
├── triangular.py                        # Clase para probabilidades difusas triangulares
└── modelos/
    ├── amenaza.py                       # Módulo de cálculo de amenaza (lógica difusa)
    ├── vulnerabilidad.py                # Módulo de cálculo de vulnerabilidad (lógica difusa)
    ├── riesgo.py                        # Módulo de cálculo de riesgo (lógica difusa)
    └── bayes_volcan.py                  # Red bayesiana discreta simple
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/RushhMax/Riesgo-Volcanico-Misti.git
cd Riesgo-Volcanico-Misti
```

2. **Crear y activar entorno virtual:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1    # En Windows PowerShell
# o
.venv\Scripts\activate.bat    # En Windows CMD
# o
source .venv/bin/activate     # En Linux/Mac
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 📊 Uso del Sistema

### Ejecución Básica

**Sistema híbrido completo:**
```bash
python main.py
```

**Demostración completa de todos los enfoques:**
```bash
python demo_completo.py
```

### Uso de Módulos Individuales

**Cálculo de amenaza:**
```python
from modelos.amenaza import calcular_amenaza
amenaza = calcular_amenaza(
    sis=15,    # eventos sísmicos/día
    gas=4000,  # ppm de gases
    deform=30, # mm de deformación
    hist=8     # nivel histórico (0-10)
)
print(f"Amenaza: {amenaza:.2f}/10")
```

**Cálculo de vulnerabilidad:**
```python
from modelos.vulnerabilidad import calcular_vulnerabilidad
vulnerabilidad = calcular_vulnerabilidad(
    dens=10000,  # personas/km²
    prep=2,      # nivel de preparación (0-5)
    prox=10,     # km al cráter
    evac=5       # planificación evacuación (0-10)
)
print(f"Vulnerabilidad: {vulnerabilidad:.2f}/10")
```

**Cálculo de riesgo:**
```python
from modelos.riesgo import calcular_riesgo
riesgo = calcular_riesgo(
    valor_amenaza=8.19,
    valor_vulnerabilidad=5.0
)
print(f"Riesgo: {riesgo:.2f}/10")
```

## 🧠 Enfoques Implementados

### 1. Lógica Difusa Modular
- **Archivos**: `modelos/amenaza.py`, `modelos/vulnerabilidad.py`, `modelos/riesgo.py`
- **Características**: Sistemas independientes con reglas difusas específicas
- **Ventajas**: Interpretabilidad, facilidad de ajuste
- **Tipo**: Sistemas de inferencia difusa tipo Mamdani

### 2. Red Bayesiana Discreta
- **Archivo**: `modelos/bayes_volcan.py`
- **Características**: Inferencia probabilística con CPDs discretas predefinidas
- **Ventajas**: Manejo de incertidumbre, inferencia bidireccional
- **Tipo**: Red bayesiana clásica con variables discretas

### 3. Sistema Híbrido Difuso-Bayesiano
- **Archivo**: `main.py`
- **Características**: Combina sistemas difusos con inferencia bayesiana discreta
- **Ventajas**: Mejor precisión, manejo de múltiples tipos de incertidumbre
- **Tipo**: Híbrido (NO es una verdadera Red Bayesiana Difusa)

### 4. Red Bayesiana Difusa Auténtica (Implementación Teórica)
- **Archivo**: `red_bayesiana_difusa_verdadera.py`
- **Características**: Nodos con variables difusas, CPDs con números difusos
- **Ventajas**: Propagación completa de incertidumbre difusa
- **Tipo**: Verdadera Red Bayesiana Difusa con inferencia difusa

## 📈 Parámetros de Entrada

| Variable | Rango | Unidad | Descripción |
|----------|-------|---------|-------------|
| Sismicidad | 0-20 | eventos/día | Número de eventos sísmicos diarios |
| Gases | 0-5000 | ppm | Concentración de gases volcánicos |
| Deformación | 0-50 | mm | Deformación del terreno |
| Historia | 0-10 | nivel | Actividad histórica del volcán |
| Densidad | 0-30000 | personas/km² | Densidad poblacional |
| Preparación | 0-5 | nivel | Nivel de preparación comunitaria |
| Proximidad | 0-20 | km | Distancia al cráter |
| Evacuación | 0-10 | nivel | Calidad de planes de evacuación |

## 📊 Interpretación de Resultados

### Niveles de Riesgo
- **0-3**: 🟢 **BAJO** - Monitoreo rutinario
- **4-6**: 🟡 **MEDIO** - Alerta y preparación
- **7-10**: 🔴 **ALTO** - Evacuación recomendada

### Salida del Sistema
El sistema proporciona:
- Valores numéricos de amenaza, vulnerabilidad y riesgo (0-10)
- Distribuciones de probabilidad discretas
- Recomendaciones de acción específicas

## 🔧 Dependencias Principales

- **numpy**: Operaciones numéricas
- **scikit-fuzzy**: Lógica difusa
- **pgmpy**: Redes bayesianas
- **matplotlib**: Visualización (opcional)

## 👥 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

- **Autor**: RushhMax
- **GitHub**: [https://github.com/RushhMax](https://github.com/RushhMax)
- **Repositorio**: [Riesgo-Volcanico-Misti](https://github.com/RushhMax/Riesgo-Volcanico-Misti)

## 🎯 Casos de Uso

### Ejemplo 1: Escenario de Alto Riesgo
```python
evidence = {
    'sismicidad': 18,     # Alta actividad sísmica
    'gases': 4500,        # Emisiones elevadas
    'deformacion': 45,    # Deformación significativa
    'historia': 9,        # Historia alta
    'densidad': 25000,    # Alta densidad poblacional
    'preparacion': 1,     # Preparación muy baja
    'proximidad': 5,      # Muy cerca del volcán
    'evacuacion': 2       # Evacuación deficiente
}
# Resultado esperado: Riesgo ALTO (8-10)
```

### Ejemplo 2: Escenario de Bajo Riesgo
```python
evidence = {
    'sismicidad': 2,      # Baja actividad sísmica
    'gases': 500,         # Emisiones normales
    'deformacion': 1,     # Sin deformación
    'historia': 3,        # Historia baja
    'densidad': 2000,     # Baja densidad poblacional
    'preparacion': 4,     # Buena preparación
    'proximidad': 18,     # Lejos del volcán
    'evacuacion': 8       # Buena evacuación
}
# Resultado esperado: Riesgo BAJO (0-3)
```

## 🔍 Validación y Testing

Para validar el sistema:
```bash
# Ejecutar todos los módulos
python -c "from modelos.amenaza import calcular_amenaza; print('Amenaza OK')"
python -c "from modelos.vulnerabilidad import calcular_vulnerabilidad; print('Vulnerabilidad OK')"
python -c "from modelos.riesgo import calcular_riesgo; print('Riesgo OK')"
python -c "from modelos.bayes_volcan import crear_red_bayesiana; print('Bayesiano OK')"
python -c "from main import FuzzyBayesianNetwork; print('Híbrido OK')"
```

---
*Desarrollado para la evaluación de riesgo volcánico del Misti, Arequipa, Perú*
