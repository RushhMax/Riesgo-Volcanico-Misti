import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Definir las variables de entrada y salida
densidad = ctrl.Antecedent(np.arange(0, 30001, 500), 'densidad')
preparacion = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'preparacion')
proximidad = ctrl.Antecedent(np.arange(0, 21, 1), 'proximidad')
evacuacion = ctrl.Antecedent(np.arange(0, 11, 1), 'evacuacion')
vulnerabilidad = ctrl.Consequent(np.arange(0, 11, 1), 'vulnerabilidad')

# 2. Funciones de membresía para cada variable

# Densidad
densidad['baja'] = fuzz.trimf(densidad.universe, [0, 0, 8000])
densidad['media'] = fuzz.trimf(densidad.universe, [5000, 15000, 20000])
densidad['alta'] = fuzz.trimf(densidad.universe, [15000, 30000, 30000])

# Preparación
preparacion['muy bajo'] = fuzz.trimf(preparacion.universe, [0, 0, 1])
preparacion['bajo'] = fuzz.trimf(preparacion.universe, [0.5, 1.5, 2.5])
preparacion['medio'] = fuzz.trimf(preparacion.universe, [2, 2.5, 3.5])
preparacion['alto'] = fuzz.trimf(preparacion.universe, [3, 4, 5])
preparacion['muy alto'] = fuzz.trimf(preparacion.universe, [4, 5, 5])

# Proximidad
proximidad['lejana'] = fuzz.trimf(proximidad.universe, [10, 20, 20])
proximidad['media'] = fuzz.trimf(proximidad.universe, [5, 10, 15])
proximidad['cercana'] = fuzz.trimf(proximidad.universe, [0, 0, 10])

# Evacuación
evacuacion['inexistente'] = fuzz.trimf(evacuacion.universe, [0, 0, 3])
evacuacion['parcial'] = fuzz.trimf(evacuacion.universe, [2, 5, 7])
evacuacion['completo'] = fuzz.trimf(evacuacion.universe, [6, 10, 10])

# Vulnerabilidad (salida)
vulnerabilidad['baja'] = fuzz.trimf(vulnerabilidad.universe, [0, 0, 4])
vulnerabilidad['media'] = fuzz.trimf(vulnerabilidad.universe, [3, 5, 7])
vulnerabilidad['alta'] = fuzz.trimf(vulnerabilidad.universe, [6, 10, 10])

# 3. Reglas difusas
reglas = [
    ctrl.Rule(densidad['alta'] & preparacion['muy bajo'] & proximidad['cercana'] & evacuacion['inexistente'], vulnerabilidad['alta']),
    ctrl.Rule(densidad['media'] & preparacion['bajo'] & proximidad['media'] & evacuacion['parcial'], vulnerabilidad['media']),
    ctrl.Rule(densidad['baja'] & preparacion['alto'] & proximidad['lejana'] & evacuacion['completo'], vulnerabilidad['baja']),
    ctrl.Rule(densidad['alta'] & preparacion['medio'] & proximidad['media'] & evacuacion['parcial'], vulnerabilidad['media']),
    ctrl.Rule(densidad['media'] & preparacion['muy bajo'] & proximidad['cercana'] & evacuacion['inexistente'], vulnerabilidad['alta']),
]

# 4. Sistema de control y simulación
sistema_vuln_ctrl = ctrl.ControlSystem(reglas)
sistema_vuln = ctrl.ControlSystemSimulation(sistema_vuln_ctrl)

# 5. Función para evaluar vulnerabilidad
def calcular_vulnerabilidad(dens, prep, prox, evac):
    sistema = ctrl.ControlSystemSimulation(sistema_vuln_ctrl)  # nueva instancia
    sistema.input['densidad'] = dens
    sistema.input['preparacion'] = prep
    sistema.input['proximidad'] = prox
    sistema.input['evacuacion'] = evac

    try:
        sistema.compute()
        return sistema.output['vulnerabilidad']
    except Exception as e:
        print(f"Error: {e}")
        return None
