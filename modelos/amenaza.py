import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variables de entrada
sismicidad = ctrl.Antecedent(np.arange(0, 21, 1), 'sismicidad')
gases = ctrl.Antecedent(np.arange(0, 5001, 100), 'gases')
deformacion = ctrl.Antecedent(np.arange(0, 51, 1), 'deformacion')
historia = ctrl.Antecedent(np.arange(0, 11, 1), 'historia')

# Variable de salida
amenaza = ctrl.Consequent(np.arange(0, 11, 1), 'amenaza')

# Funciones de membresía
sismicidad['baja'] = fuzz.trimf(sismicidad.universe, [0, 0, 5])
sismicidad['media'] = fuzz.trimf(sismicidad.universe, [3, 8, 12])
sismicidad['alta'] = fuzz.trimf(sismicidad.universe, [10, 20, 20])

gases['normal'] = fuzz.trimf(gases.universe, [0, 0, 1500])
gases['elevada'] = fuzz.trimf(gases.universe, [1000, 5000, 5000])

deformacion['nula'] = fuzz.trimf(deformacion.universe, [0, 0, 5])
deformacion['leve'] = fuzz.trimf(deformacion.universe, [3, 20, 30])
deformacion['significativa'] = fuzz.trimf(deformacion.universe, [25, 50, 50])

historia['baja'] = fuzz.trimf(historia.universe, [0, 0, 2])
historia['media'] = fuzz.trimf(historia.universe, [1, 5, 7])
historia['alta'] = fuzz.trimf(historia.universe, [6, 10, 10])

amenaza['baja'] = fuzz.trimf(amenaza.universe, [0, 0, 4])
amenaza['media'] = fuzz.trimf(amenaza.universe, [3, 5, 7])
amenaza['alta'] = fuzz.trimf(amenaza.universe, [6, 10, 10])

# Reglas corregidas (solo se usan etiquetas que sí existen)
rules = [
    ctrl.Rule(sismicidad['alta'] & gases['elevada'] & deformacion['significativa'] & historia['alta'], amenaza['alta']),
    ctrl.Rule(sismicidad['media'] & gases['normal'] & deformacion['leve'] & historia['media'], amenaza['media']),
    ctrl.Rule(sismicidad['baja'] & gases['normal'] & deformacion['nula'] & historia['baja'], amenaza['baja']),
    ctrl.Rule(sismicidad['alta'] & gases['elevada'] & deformacion['leve'] & historia['media'], amenaza['media']),
    ctrl.Rule(sismicidad['media'] & gases['normal'] & deformacion['nula'] & historia['baja'], amenaza['baja']),
]

# Sistema de control
sistema_amenaza_ctrl = ctrl.ControlSystem(rules)
sistema_amenaza = ctrl.ControlSystemSimulation(sistema_amenaza_ctrl)

# Función de evaluación
def calcular_amenaza(sis, gas, deform, hist):
    sistema_amenaza.input['sismicidad'] = sis
    sistema_amenaza.input['gases'] = gas
    sistema_amenaza.input['deformacion'] = deform
    sistema_amenaza.input['historia'] = hist
    sistema_amenaza.compute()
    return sistema_amenaza.output['amenaza']


