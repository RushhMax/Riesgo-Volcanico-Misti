# from modelos.amenaza import calcular_amenaza
# from modelos.vulnerabilidad import calcular_vulnerabilidad
# from modelos.riesgo import calcular_riesgo

# # Ejemplo de valores de entrada
# sismicidad_val = 15         # eventos/día
# gases_val = 4000            # ppm
# deformacion_val = 30        # mm
# historia_val = 8            # nivel histórico de actividad (0–10)

# densidad_val = 10000        # personas/km²
# preparacion_val = 2         # 0–10 (mala a excelente)
# proximidad_val = 10         # km al cráter
# evacuacion_val = 5          # 0–10 (pobre a buena planificación)

# # Paso 1: Calcular amenaza difusa
# valor_amenaza = calcular_amenaza(
#     sis=sismicidad_val,
#     gas=gases_val,
#     deform=deformacion_val,
#     hist=historia_val
# )

# # Paso 2: Calcular vulnerabilidad difusa
# valor_vulnerabilidad = calcular_vulnerabilidad(
#     dens=densidad_val,
#     prep=preparacion_val,
#     prox=proximidad_val,
#     evac=evacuacion_val
# )

# # Paso 3: Evaluar el riesgo a partir de amenaza y vulnerabilidad
# valor_riesgo = calcular_riesgo(
#     valor_amenaza=valor_amenaza,
#     valor_vulnerabilidad=valor_vulnerabilidad
# )

# # Mostrar resultados
# print(f"Amenaza: {valor_amenaza:.2f}")
# print(f"Vulnerabilidad: {valor_vulnerabilidad:.2f}")
# print(f"Riesgo Volcánico Estimado: {valor_riesgo:.2f}")


# from pgmpy.models import DiscreteBayesianNetwork
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
# from modelos.bayes_volcan import crear_red_bayesiana, probabilidad_a_valor_difuso

# if __name__ == '__main__':
#     # Crear modelo
#     modelo = crear_red_bayesiana()
#     infer = VariableElimination(modelo)

#     # Evidencia conocida
#     evidencia = {
#         'sismicidad': 2,     # 0=baja, 1=media, 2=alta
#         'gases': 1,          # 0=normal, 1=media, 2=alta
#         'deformacion': 0,    # 0=nula
#         'historia': 1,       # 0=baja, 1=alta
#         'densidad': 2,       # 0=baja, 1=media, 2=alta
#         'preparacion': 0     # 0=baja preparación
#     }

#     # Inferir amenaza
#     resultado_amenaza = infer.query(variables=['amenaza'], evidence=evidencia)
#     print("Distribución de 'amenaza':")
#     for i, etiqueta in enumerate(['baja', 'media', 'alta']):
#         print(f"  {etiqueta}: {resultado_amenaza.values[i]:.2f}")
#     valor_amenaza = probabilidad_a_valor_difuso(resultado_amenaza.values)
#     print(f"\nNivel difuso de amenaza (esperado): {valor_amenaza:.2f}")

#     # Inferir vulnerabilidad
#     resultado_vulnerabilidad = infer.query(variables=['vulnerabilidad'], evidence=evidencia)
#     print("\nDistribución de 'vulnerabilidad':")
#     for i, etiqueta in enumerate(['baja', 'media', 'alta']):
#         print(f"  {etiqueta}: {resultado_vulnerabilidad.values[i]:.2f}")
#     valor_vulnerabilidad = probabilidad_a_valor_difuso(resultado_vulnerabilidad.values)
#     print(f"\nNivel difuso de vulnerabilidad (esperado): {valor_vulnerabilidad:.2f}")


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class FuzzyBayesianNetwork:
    def __init__(self):
        # 1. Definir la estructura de la red bayesiana
        self.model = DiscreteBayesianNetwork([
            ('sismicidad', 'amenaza'),
            ('gases', 'amenaza'),
            ('deformacion', 'amenaza'),
            ('historia', 'amenaza'),
            ('densidad', 'vulnerabilidad'),
            ('preparacion', 'vulnerabilidad'),
            ('proximidad', 'vulnerabilidad'),  # Add these missing parents
            ('evacuacion', 'vulnerabilidad'),  # for vulnerabilidad
            ('amenaza', 'riesgo'),
            ('vulnerabilidad', 'riesgo')
        ])
        
        # 2. Configurar los sistemas difusos para cada variable
        self._setup_fuzzy_systems()
        
        # 3. Definir las CPDs basadas en lógica difusa
        self._setup_cpds()
        
        # 4. Verificar el modelo
        self.model.check_model()
        
        # 5. Crear el motor de inferencia
        self.inference = VariableElimination(self.model)
    
    def _setup_fuzzy_systems(self):
        """Configura los sistemas de inferencia difusa para cada variable"""
        # Sistema difuso para amenaza
        self.amenaza_system = self._create_amenaza_system()
        
        # Sistema difuso para vulnerabilidad
        self.vulnerabilidad_system = self._create_vulnerabilidad_system()
        
        # Sistema difuso para riesgo
        self.riesgo_system = self._create_riesgo_system()
    
    def _create_amenaza_system(self):
        """Crea el sistema difuso para calcular amenaza"""
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

        # Reglas
        rules = [
            ctrl.Rule(sismicidad['alta'] & gases['elevada'] & deformacion['significativa'] & historia['alta'], amenaza['alta']),
            ctrl.Rule(sismicidad['media'] & gases['normal'] & deformacion['leve'] & historia['media'], amenaza['media']),
            ctrl.Rule(sismicidad['baja'] & gases['normal'] & deformacion['nula'] & historia['baja'], amenaza['baja']),
            ctrl.Rule(sismicidad['alta'] & gases['elevada'] & deformacion['leve'] & historia['media'], amenaza['media']),
            ctrl.Rule(sismicidad['media'] & gases['normal'] & deformacion['nula'] & historia['baja'], amenaza['baja']),
        ]

        return ctrl.ControlSystem(rules)
    
    def _create_vulnerabilidad_system(self):
        """Crea el sistema difuso para calcular vulnerabilidad"""
        # Variables de entrada
        densidad = ctrl.Antecedent(np.arange(0, 30001, 500), 'densidad')
        preparacion = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'preparacion')
        proximidad = ctrl.Antecedent(np.arange(0, 21, 1), 'proximidad')
        evacuacion = ctrl.Antecedent(np.arange(0, 11, 1), 'evacuacion')
        
        # Variable de salida
        vulnerabilidad = ctrl.Consequent(np.arange(0, 11, 1), 'vulnerabilidad')
        
        # Funciones de membresía
        densidad['baja'] = fuzz.trimf(densidad.universe, [0, 0, 8000])
        densidad['media'] = fuzz.trimf(densidad.universe, [5000, 15000, 20000])
        densidad['alta'] = fuzz.trimf(densidad.universe, [15000, 30000, 30000])

        preparacion['muy bajo'] = fuzz.trimf(preparacion.universe, [0, 0, 1])
        preparacion['bajo'] = fuzz.trimf(preparacion.universe, [0.5, 1.5, 2.5])
        preparacion['medio'] = fuzz.trimf(preparacion.universe, [2, 2.5, 3.5])
        preparacion['alto'] = fuzz.trimf(preparacion.universe, [3, 4, 5])
        preparacion['muy alto'] = fuzz.trimf(preparacion.universe, [4, 5, 5])

        proximidad['lejana'] = fuzz.trimf(proximidad.universe, [10, 20, 20])
        proximidad['media'] = fuzz.trimf(proximidad.universe, [5, 10, 15])
        proximidad['cercana'] = fuzz.trimf(proximidad.universe, [0, 0, 10])

        evacuacion['inexistente'] = fuzz.trimf(evacuacion.universe, [0, 0, 3])
        evacuacion['parcial'] = fuzz.trimf(evacuacion.universe, [2, 5, 7])
        evacuacion['completo'] = fuzz.trimf(evacuacion.universe, [6, 10, 10])

        vulnerabilidad['baja'] = fuzz.trimf(vulnerabilidad.universe, [0, 0, 4])
        vulnerabilidad['media'] = fuzz.trimf(vulnerabilidad.universe, [3, 5, 7])
        vulnerabilidad['alta'] = fuzz.trimf(vulnerabilidad.universe, [6, 10, 10])

        # Reglas
        rules = [
            ctrl.Rule(densidad['alta'] & preparacion['muy bajo'] & proximidad['cercana'] & evacuacion['inexistente'], vulnerabilidad['alta']),
            ctrl.Rule(densidad['media'] & preparacion['bajo'] & proximidad['media'] & evacuacion['parcial'], vulnerabilidad['media']),
            ctrl.Rule(densidad['baja'] & preparacion['alto'] & proximidad['lejana'] & evacuacion['completo'], vulnerabilidad['baja']),
            ctrl.Rule(densidad['alta'] & preparacion['medio'] & proximidad['media'] & evacuacion['parcial'], vulnerabilidad['media']),
            ctrl.Rule(densidad['media'] & preparacion['muy bajo'] & proximidad['cercana'] & evacuacion['inexistente'], vulnerabilidad['alta']),
        ]

        return ctrl.ControlSystem(rules)
    
    def _create_riesgo_system(self):
        """Crea el sistema difuso para calcular riesgo"""
        # Variables de entrada
        amenaza = ctrl.Antecedent(np.arange(0, 11, 1), 'amenaza')
        vulnerabilidad = ctrl.Antecedent(np.arange(0, 11, 1), 'vulnerabilidad')
        
        # Variable de salida
        riesgo = ctrl.Consequent(np.arange(0, 11, 1), 'riesgo')
        
        # Funciones de membresía
        amenaza['baja'] = fuzz.trimf(amenaza.universe, [0, 0, 4])
        amenaza['media'] = fuzz.trimf(amenaza.universe, [3, 5, 7])
        amenaza['alta'] = fuzz.trimf(amenaza.universe, [6, 10, 10])

        vulnerabilidad['baja'] = fuzz.trimf(vulnerabilidad.universe, [0, 0, 4])
        vulnerabilidad['media'] = fuzz.trimf(vulnerabilidad.universe, [3, 5, 7])
        vulnerabilidad['alta'] = fuzz.trimf(vulnerabilidad.universe, [6, 10, 10])

        riesgo['bajo'] = fuzz.trimf(riesgo.universe, [0, 0, 4])
        riesgo['medio'] = fuzz.trimf(riesgo.universe, [3, 5, 7])
        riesgo['alto'] = fuzz.trimf(riesgo.universe, [6, 10, 10])

        # Reglas
        rules = [
            ctrl.Rule(amenaza['baja'] & vulnerabilidad['baja'], riesgo['bajo']),
            ctrl.Rule(amenaza['baja'] & vulnerabilidad['media'], riesgo['medio']),
            ctrl.Rule(amenaza['baja'] & vulnerabilidad['alta'], riesgo['medio']),
            ctrl.Rule(amenaza['media'] & vulnerabilidad['baja'], riesgo['medio']),
            ctrl.Rule(amenaza['media'] & vulnerabilidad['media'], riesgo['medio']),
            ctrl.Rule(amenaza['media'] & vulnerabilidad['alta'], riesgo['alto']),
            ctrl.Rule(amenaza['alta'] & vulnerabilidad['baja'], riesgo['medio']),
            ctrl.Rule(amenaza['alta'] & vulnerabilidad['media'], riesgo['alto']),
            ctrl.Rule(amenaza['alta'] & vulnerabilidad['alta'], riesgo['alto']),
        ]

        return ctrl.ControlSystem(rules)
    
    def _setup_cpds(self):
        """Configura las CPDs basadas en los sistemas difusos"""
        # Para nodos padres (distribuciones difusas)
        cpd_sismicidad = self._create_fuzzy_cpd('sismicidad', ['baja', 'media', 'alta'], 
                                            [(0, 0, 5), (3, 8, 12), (10, 20, 20)])
        cpd_gases = self._create_fuzzy_cpd('gases', ['normal', 'elevada'], 
                                        [(0, 0, 1500), (1000, 5000, 5000)])
        cpd_deformacion = self._create_fuzzy_cpd('deformacion', ['nula', 'leve', 'significativa'], 
                                            [(0, 0, 5), (3, 20, 30), (25, 50, 50)])
        cpd_historia = self._create_fuzzy_cpd('historia', ['baja', 'media', 'alta'], 
                                            [(0, 0, 2), (1, 5, 7), (6, 10, 10)])
        cpd_densidad = self._create_fuzzy_cpd('densidad', ['baja', 'media', 'alta'], 
                                            [(0, 0, 8000), (5000, 15000, 20000), (15000, 30000, 30000)])
        cpd_preparacion = self._create_fuzzy_cpd('preparacion', ['muy bajo', 'bajo', 'medio', 'alto', 'muy alto'], 
                                            [(0, 0, 1), (0.5, 1.5, 2.5), (2, 2.5, 3.5), (3, 4, 5), (4, 5, 5)])
        cpd_proximidad = self._create_fuzzy_cpd('proximidad', ['lejana', 'media', 'cercana'],
                                            [(10, 20, 20), (5, 10, 15), (0, 0, 10)])
        cpd_evacuacion = self._create_fuzzy_cpd('evacuacion', ['inexistente', 'parcial', 'completo'],
                                            [(0, 0, 3), (2, 5, 7), (6, 10, 10)])
        
        # Para nodos hijos (usando funciones difusas)
        cpd_amenaza = self._create_fuzzy_child_cpd('amenaza', ['baja', 'media', 'alta'], 
                                                ['sismicidad', 'gases', 'deformacion', 'historia'], 
                                                self.amenaza_system)
        
        cpd_vulnerabilidad = self._create_fuzzy_child_cpd('vulnerabilidad', ['baja', 'media', 'alta'], 
                                                        ['densidad', 'preparacion', 'proximidad', 'evacuacion'], 
                                                        self.vulnerabilidad_system)
        
        cpd_riesgo = self._create_fuzzy_child_cpd('riesgo', ['bajo', 'medio', 'alto'], 
                                                ['amenaza', 'vulnerabilidad'], 
                                                self.riesgo_system)
        
        # Agregar todas las CPDs al modelo
        self.model.add_cpds(cpd_sismicidad, cpd_gases, cpd_deformacion, cpd_historia,
                        cpd_densidad, cpd_preparacion, cpd_proximidad, cpd_evacuacion,
                        cpd_amenaza, cpd_vulnerabilidad, cpd_riesgo)
    def _create_fuzzy_cpd(self, variable, states, ranges):
        """Crea una CPD basada en funciones de membresía difusas para variables de entrada"""
        # Para variables de entrada, usamos distribuciones basadas en las funciones de membresía
        values = []
        for r in ranges:
            # Crear una distribución triangular centrada en el punto medio del rango
            midpoint = (r[0] + r[2]) / 2
            values.append([midpoint])
        
        # Normalizar los valores para que sumen 1
        total = sum(v[0] for v in values)
        normalized_values = [[v[0]/total] for v in values]
        
        return TabularCPD(variable=variable, variable_card=len(states), values=normalized_values)
    
    def _create_fuzzy_child_cpd(self, variable, states, parents, fuzzy_system):
        """Crea una CPD para un nodo hijo usando un sistema difuso"""
        # Define the number of states for each parent
        parent_states = {
            'sismicidad': 3,
            'gases': 2,  # gases has 2 states
            'deformacion': 3,
            'historia': 3,
            'densidad': 3,
            'preparacion': 5,  # preparacion has 5 states
            'proximidad': 3,
            'evacuacion': 3,
            'amenaza': 3,
            'vulnerabilidad': 3,
            'riesgo': 3
        }
        
        parent_cards = [parent_states[parent] for parent in parents]
        
        # Create all combinations of parent states
        # This is simplified - in a real implementation you would evaluate the fuzzy system
        # for each combination to get proper probabilities
        num_combinations = np.prod(parent_cards)
        values = [
            [0.8]*num_combinations,  # baja/bajo
            [0.15]*num_combinations,  # media/medio
            [0.05]*num_combinations   # alta/alto
        ][:len(states)]  # Adjust based on number of states
        
        return TabularCPD(
            variable=variable,
            variable_card=len(states),
            evidence=parents,
            evidence_card=parent_cards,
            values=values
        )
    
    def evaluate_risk(self, evidence, verbose=False):
        """
        Evalúa el riesgo volcánico basado en evidencia proporcionada.
        
        Args:
            evidence (dict): Diccionario con los valores de las variables observadas
            verbose (bool): Si True, muestra información detallada del proceso
            
        Returns:
            dict: Distribución de probabilidad del riesgo
        """
        # 1. Evaluar los sistemas difusos con la evidencia proporcionada
        if verbose:
            print("Evaluando sistemas difusos...")
        
        # Evaluar amenaza si tenemos todos sus padres
        amenaza_parents = {'sismicidad', 'gases', 'deformacion', 'historia'}
        if amenaza_parents.issubset(evidence.keys()):
            sim = ctrl.ControlSystemSimulation(self.amenaza_system)
            sim.input['sismicidad'] = evidence['sismicidad']
            sim.input['gases'] = evidence['gases']
            sim.input['deformacion'] = evidence['deformacion']
            sim.input['historia'] = evidence['historia']
            sim.compute()
            amenaza_value = sim.output['amenaza']
            evidence['amenaza'] = amenaza_value
            if verbose:
                print(f"Valor difuso de amenaza: {amenaza_value:.2f}")
        
        # Evaluar vulnerabilidad si tenemos todos sus padres
        vulnerabilidad_parents = {'densidad', 'preparacion', 'proximidad', 'evacuacion'}
        if vulnerabilidad_parents.issubset(evidence.keys()):
            sim = ctrl.ControlSystemSimulation(self.vulnerabilidad_system)
            sim.input['densidad'] = evidence['densidad']
            sim.input['preparacion'] = evidence['preparacion']
            sim.input['proximidad'] = evidence['proximidad']
            sim.input['evacuacion'] = evidence['evacuacion']
            sim.compute()
            vulnerabilidad_value = sim.output['vulnerabilidad']
            evidence['vulnerabilidad'] = vulnerabilidad_value
            if verbose:
                print(f"Valor difuso de vulnerabilidad: {vulnerabilidad_value:.2f}")
        
        # Evaluar riesgo si tenemos amenaza y vulnerabilidad (but don't add to evidence)
        riesgo_parents = {'amenaza', 'vulnerabilidad'}
        if riesgo_parents.issubset(evidence.keys()):
            sim = ctrl.ControlSystemSimulation(self.riesgo_system)
            sim.input['amenaza'] = evidence['amenaza']
            sim.input['vulnerabilidad'] = evidence['vulnerabilidad']
            sim.compute()
            riesgo_value = sim.output['riesgo']
            if verbose:
                print(f"Valor difuso de riesgo: {riesgo_value:.2f}")
        
        # 2. Realizar inferencia en la red bayesiana
        if verbose:
            print("\nRealizando inferencia bayesiana...")
        
        # Convertir valores continuos a discretos para la inferencia bayesiana
        discretized_evidence = {}
        for var, value in evidence.items():
            if var in ['amenaza', 'vulnerabilidad']:  # Don't include riesgo here
                # Discretizar basado en las funciones de membresía
                if value < 4:
                    discretized_evidence[var] = 'baja' if var != 'riesgo' else 'bajo'
                elif value < 7:
                    discretized_evidence[var] = 'media' if var != 'riesgo' else 'medio'
                else:
                    discretized_evidence[var] = 'alta' if var != 'riesgo' else 'alto'
            elif var not in ['riesgo']:  # Exclude riesgo from evidence
                discretized_evidence[var] = value
        
        if verbose:
            print("Evidencia discretizada:", discretized_evidence)
        
        # Realizar la consulta de inferencia (now riesgo is not in evidence)
        query_result = self.inference.query(variables=['riesgo'], evidence=discretized_evidence)
        
        return {
            'fuzzy_values': {
                'amenaza': evidence.get('amenaza'),
                'vulnerabilidad': evidence.get('vulnerabilidad'),
                'riesgo': riesgo_value  # From fuzzy calculation
            },
            'discrete_probabilities': query_result
        }
# Ejemplo de uso
if __name__ == "__main__":
    # Crear la red bayesiana difusa
    fbn = FuzzyBayesianNetwork()
    
    # Definir evidencia (valores de entrada)
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
    
    # Evaluar el riesgo
    result = fbn.evaluate_risk(evidence, verbose=True)
    
    # Mostrar resultados
    print("\nResultados:")
    print("Valores difusos calculados:")
    for var, value in result['fuzzy_values'].items():
        if var in ['amenaza', 'vulnerabilidad', 'riesgo']:
            print(f"- {var}: {value:.2f}")
    
    print("\nDistribución de probabilidad discreta para el riesgo:")
    print(result['discrete_probabilities'])