import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Crear variables difusas
x_riesgo = np.arange(0, 11, 1)
x_amenaza = np.arange(0, 11, 1)
x_vulnerabilidad = np.arange(0, 11, 1)

amenaza = ctrl.Antecedent(x_amenaza, 'amenaza')
vulnerabilidad = ctrl.Antecedent(x_vulnerabilidad, 'vulnerabilidad')
riesgo = ctrl.Consequent(x_riesgo, 'riesgo')

# Funciones de membresía para amenaza
amenaza['baja'] = fuzz.trimf(x_amenaza, [0, 0, 4])
amenaza['media'] = fuzz.trimf(x_amenaza, [2, 5, 8])
amenaza['alta'] = fuzz.trimf(x_amenaza, [6, 10, 10])

# Funciones de membresía para vulnerabilidad
vulnerabilidad['baja'] = fuzz.trimf(x_vulnerabilidad, [0, 0, 4])
vulnerabilidad['media'] = fuzz.trimf(x_vulnerabilidad, [2, 5, 8])
vulnerabilidad['alta'] = fuzz.trimf(x_vulnerabilidad, [6, 10, 10])

# Funciones de membresía para riesgo
riesgo['bajo'] = fuzz.trimf(x_riesgo, [0, 0, 4])
riesgo['medio'] = fuzz.trimf(x_riesgo, [2, 5, 8])
riesgo['alto'] = fuzz.trimf(x_riesgo, [6, 10, 10])

# Reglas difusas
reglas = [
    ctrl.Rule(antecedent=(amenaza['baja'] & vulnerabilidad['baja']), consequent=riesgo['bajo']),
    ctrl.Rule(antecedent=(amenaza['baja'] & vulnerabilidad['media']), consequent=riesgo['medio']),
    ctrl.Rule(antecedent=(amenaza['baja'] & vulnerabilidad['alta']), consequent=riesgo['medio']),

    ctrl.Rule(antecedent=(amenaza['media'] & vulnerabilidad['baja']), consequent=riesgo['medio']),
    ctrl.Rule(antecedent=(amenaza['media'] & vulnerabilidad['media']), consequent=riesgo['medio']),
    ctrl.Rule(antecedent=(amenaza['media'] & vulnerabilidad['alta']), consequent=riesgo['alto']),

    ctrl.Rule(antecedent=(amenaza['alta'] & vulnerabilidad['baja']), consequent=riesgo['medio']),
    ctrl.Rule(antecedent=(amenaza['alta'] & vulnerabilidad['media']), consequent=riesgo['alto']),
    ctrl.Rule(antecedent=(amenaza['alta'] & vulnerabilidad['alta']), consequent=riesgo['alto']),
]

# Sistema y simulación
sistema_ctrl = ctrl.ControlSystem(reglas)
sistema_riesgo = ctrl.ControlSystemSimulation(sistema_ctrl)

def calcular_riesgo(valor_amenaza, valor_vulnerabilidad, graficar=False):
    sistema_riesgo.input['amenaza'] = valor_amenaza
    sistema_riesgo.input['vulnerabilidad'] = valor_vulnerabilidad
    sistema_riesgo.compute()
    resultado = sistema_riesgo.output['riesgo']

    if graficar:
        riesgo.view(sim=sistema_riesgo)
        plt.title("Resultado del Riesgo Difuso")
        plt.show()

    return resultado
