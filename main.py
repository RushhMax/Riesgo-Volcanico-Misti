import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from triangular import TriangularFuzzyProbability
import itertools

class FuzzyBayesianNode:
    """Nodo de una Red Bayesiana Difusa"""
    
    def __init__(self, name, states, parents=None):
        self.name = name
        self.states = states  # Lista de etiquetas de estados ['bajo', 'medio', 'alto']
        self.parents = parents or []
        self.fuzzy_cpd = {}  # CPD con números difusos
        self.fuzzy_prior = {}  # Distribución a priori difusa
    
    def set_fuzzy_prior(self, fuzzy_distribution):
        """Establecer distribución a priori difusa"""
        self.fuzzy_prior = fuzzy_distribution
    
    def set_fuzzy_cpd(self, fuzzy_cpd):
        """Establecer CPD difusa"""
        self.fuzzy_cpd = fuzzy_cpd

class TrueFuzzyBayesianNetwork:
    """Red Bayesiana Difusa verdadera con inferencia difusa completa"""
    
    def __init__(self):
        self.nodes = {}
        self.fuzzy_systems = {}
        self._create_network()
    
    def _create_network(self):
        """Crear la estructura de la red con nodos difusos"""
        
        # Crear nodos con estados lingüísticos
        self.nodes['sismicidad'] = FuzzyBayesianNode('sismicidad', ['baja', 'media', 'alta'])
        self.nodes['gases'] = FuzzyBayesianNode('gases', ['normal', 'elevada'])
        self.nodes['deformacion'] = FuzzyBayesianNode('deformacion', ['nula', 'leve', 'significativa'])
        self.nodes['historia'] = FuzzyBayesianNode('historia', ['baja', 'media', 'alta'])
        
        self.nodes['densidad'] = FuzzyBayesianNode('densidad', ['baja', 'media', 'alta'])
        self.nodes['preparacion'] = FuzzyBayesianNode('preparacion', ['muy bajo', 'bajo', 'medio', 'alto', 'muy alto'])
        self.nodes['proximidad'] = FuzzyBayesianNode('proximidad', ['lejana', 'media', 'cercana'])
        self.nodes['evacuacion'] = FuzzyBayesianNode('evacuacion', ['inexistente', 'parcial', 'completo'])
        
        # Nodos con padres
        self.nodes['amenaza'] = FuzzyBayesianNode('amenaza', ['baja', 'media', 'alta'], 
                                                ['sismicidad', 'gases', 'deformacion', 'historia'])
        self.nodes['vulnerabilidad'] = FuzzyBayesianNode('vulnerabilidad', ['baja', 'media', 'alta'],
                                                        ['densidad', 'preparacion', 'proximidad', 'evacuacion'])
        self.nodes['riesgo'] = FuzzyBayesianNode('riesgo', ['bajo', 'medio', 'alto'],
                                               ['amenaza', 'vulnerabilidad'])
        
        # Configurar distribuciones difusas
        self._setup_fuzzy_distributions()
        
        # Configurar sistemas difusos para mapeo
        self._setup_fuzzy_systems()
    
    def _setup_fuzzy_distributions(self):
        """Configurar distribuciones a priori y CPDs difusas"""
        
        # Distribuciones a priori difusas para nodos raíz
        self.nodes['sismicidad'].set_fuzzy_prior({
            'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'media': TriangularFuzzyProbability(0.2, 0.3, 0.4),
            'alta': TriangularFuzzyProbability(0.1, 0.2, 0.3)
        })
        
        self.nodes['gases'].set_fuzzy_prior({
            'normal': TriangularFuzzyProbability(0.5, 0.6, 0.7),
            'elevada': TriangularFuzzyProbability(0.3, 0.4, 0.5)
        })
        
        self.nodes['deformacion'].set_fuzzy_prior({
            'nula': TriangularFuzzyProbability(0.4, 0.5, 0.6),
            'leve': TriangularFuzzyProbability(0.2, 0.3, 0.4),
            'significativa': TriangularFuzzyProbability(0.1, 0.2, 0.3)
        })
        
        self.nodes['historia'].set_fuzzy_prior({
            'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'media': TriangularFuzzyProbability(0.2, 0.3, 0.4),
            'alta': TriangularFuzzyProbability(0.1, 0.2, 0.3)
        })
        
        self.nodes['densidad'].set_fuzzy_prior({
            'baja': TriangularFuzzyProbability(0.2, 0.3, 0.4),
            'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
        })
        
        self.nodes['preparacion'].set_fuzzy_prior({
            'muy bajo': TriangularFuzzyProbability(0.1, 0.2, 0.3),
            'bajo': TriangularFuzzyProbability(0.2, 0.25, 0.3),
            'medio': TriangularFuzzyProbability(0.2, 0.25, 0.3),
            'alto': TriangularFuzzyProbability(0.15, 0.2, 0.25),
            'muy alto': TriangularFuzzyProbability(0.05, 0.1, 0.15)
        })
        
        self.nodes['proximidad'].set_fuzzy_prior({
            'lejana': TriangularFuzzyProbability(0.2, 0.3, 0.4),
            'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'cercana': TriangularFuzzyProbability(0.2, 0.3, 0.4)
        })
        
        self.nodes['evacuacion'].set_fuzzy_prior({
            'inexistente': TriangularFuzzyProbability(0.2, 0.3, 0.4),
            'parcial': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'completo': TriangularFuzzyProbability(0.2, 0.3, 0.4)
        })
        
        # Distribuciones a priori para nodos intermedios (por si acaso)
        self.nodes['amenaza'].set_fuzzy_prior({
            'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
        })
        
        self.nodes['vulnerabilidad'].set_fuzzy_prior({
            'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
        })
        
        self.nodes['riesgo'].set_fuzzy_prior({
            'bajo': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'medio': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'alto': TriangularFuzzyProbability(0.2, 0.3, 0.4)
        })
        
        # CPDs difusas para nodos hijos
        self._setup_amenaza_fuzzy_cpd()
        self._setup_vulnerabilidad_fuzzy_cpd()
        self._setup_riesgo_fuzzy_cpd()
    
    def _setup_amenaza_fuzzy_cpd(self):
        """Configurar CPD difusa para amenaza"""
        # Reglas difusas basadas en conocimiento experto
        fuzzy_rules = {
            # (sismicidad, gases, deformacion, historia): {baja, media, alta}
            ('alta', 'elevada', 'significativa', 'alta'): {
                'baja': TriangularFuzzyProbability(0.0, 0.05, 0.1),
                'media': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'alta': TriangularFuzzyProbability(0.6, 0.75, 0.9)
            },
            ('baja', 'normal', 'nula', 'baja'): {
                'baja': TriangularFuzzyProbability(0.6, 0.8, 0.9),
                'media': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'alta': TriangularFuzzyProbability(0.0, 0.05, 0.1)
            },
            ('media', 'normal', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.1, 0.2, 0.3)
            },
            # Reglas adicionales para cubrir más combinaciones
            ('alta', 'elevada', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'media': TriangularFuzzyProbability(0.3, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
            },
            ('media', 'normal', 'nula', 'baja'): {
                'baja': TriangularFuzzyProbability(0.5, 0.7, 0.8),
                'media': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'alta': TriangularFuzzyProbability(0.0, 0.1, 0.2)
            }
        }
        self.nodes['amenaza'].set_fuzzy_cpd(fuzzy_rules)
    
    def _setup_vulnerabilidad_fuzzy_cpd(self):
        """Configurar CPD difusa para vulnerabilidad"""
        fuzzy_rules = {
            # (densidad, preparacion, proximidad, evacuacion): {baja, media, alta}
            ('alta', 'muy bajo', 'cercana', 'inexistente'): {
                'baja': TriangularFuzzyProbability(0.0, 0.05, 0.1),
                'media': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'alta': TriangularFuzzyProbability(0.6, 0.75, 0.9)
            },
            ('baja', 'alto', 'lejana', 'completo'): {
                'baja': TriangularFuzzyProbability(0.6, 0.8, 0.9),
                'media': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'alta': TriangularFuzzyProbability(0.0, 0.05, 0.1)
            },
            ('media', 'bajo', 'media', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.1, 0.2, 0.3)
            },
            ('alta', 'medio', 'media', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'media': TriangularFuzzyProbability(0.3, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
            },
            ('media', 'muy bajo', 'cercana', 'inexistente'): {
                'baja': TriangularFuzzyProbability(0.0, 0.1, 0.2),
                'media': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'alta': TriangularFuzzyProbability(0.4, 0.6, 0.8)
            }
        }
        self.nodes['vulnerabilidad'].set_fuzzy_cpd(fuzzy_rules)
    
    def _setup_riesgo_fuzzy_cpd(self):
        """Configurar CPD difusa para riesgo"""
        fuzzy_rules = {
            # (amenaza, vulnerabilidad): {bajo, medio, alto}
            ('baja', 'baja'): {
                'bajo': TriangularFuzzyProbability(0.7, 0.8, 0.9),
                'medio': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'alto': TriangularFuzzyProbability(0.0, 0.05, 0.1)
            },
            ('baja', 'media'): {
                'bajo': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'medio': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alto': TriangularFuzzyProbability(0.0, 0.1, 0.2)
            },
            ('baja', 'alta'): {
                'bajo': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'medio': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alto': TriangularFuzzyProbability(0.1, 0.2, 0.3)
            },
            ('media', 'baja'): {
                'bajo': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'medio': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alto': TriangularFuzzyProbability(0.0, 0.1, 0.2)
            },
            ('media', 'media'): {
                'bajo': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'medio': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alto': TriangularFuzzyProbability(0.1, 0.2, 0.3)
            },
            ('media', 'alta'): {
                'bajo': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'medio': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'alto': TriangularFuzzyProbability(0.4, 0.55, 0.7)
            },
            ('alta', 'baja'): {
                'bajo': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'medio': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alto': TriangularFuzzyProbability(0.1, 0.2, 0.3)
            },
            ('alta', 'media'): {
                'bajo': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'medio': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'alto': TriangularFuzzyProbability(0.4, 0.55, 0.7)
            },
            ('alta', 'alta'): {
                'bajo': TriangularFuzzyProbability(0.0, 0.05, 0.1),
                'medio': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'alto': TriangularFuzzyProbability(0.6, 0.75, 0.9)
            }
        }
        self.nodes['riesgo'].set_fuzzy_cpd(fuzzy_rules)
    
    def _setup_fuzzy_systems(self):
        """Configurar sistemas difusos para mapeo de valores crisp a estados lingüísticos"""
        # Sistema para mapear sismicidad

        # Sismicidad (eventos/día)
        self.fuzzy_systems['sismicidad'] = {
            'ranges': {
                'baja': (0, 5),
                'media': (3, 12),
                'alta': (10, 20)
            },
            'unqiverse': np.arange(0, 21, 1)
        }

        # Emisión de gases (ppm SO₂)
        self.fuzzy_systems['gases'] = {
            'ranges': {
                'normal': (0, 1500),
                'elevada': (1000, 5000)
            },
            'universe': np.arange(0, 5001, 100)
        }

        # Deformación del terreno (mm)
        self.fuzzy_systems['deformacion'] = {
            'ranges': {
                'nula': (0, 5),
                'leve': (3, 30),
                'significativa': (25, 50)
            },
            'universe': np.arange(0, 51, 1)
        }

        # Actividad histórica (nivel 0–10)
        self.fuzzy_systems['historia'] = {
            'ranges': {
                'baja': (0, 2),
                'media': (1, 7),
                'alta': (6, 10)
            },
            'universe': np.arange(0, 11, 1)
        }

        # Densidad poblacional (personas/km²)
        self.fuzzy_systems['densidad'] = {
            'ranges': {
                'baja': (0, 5000),
                'media': (4000, 15000),
                'alta': (12000, 30000)
            },
            'universe': np.arange(0, 30001, 500)
        }

        # Nivel de preparación comunitaria (0–5)
        self.fuzzy_systems['preparacion'] = {
            'ranges': {
                'baja': (0, 1.5),
                'media': (1, 3.5),
                'alta': (3, 5)
            },
            'universe': np.arange(0, 5.1, 0.1)
        }

        # Proximidad al cráter (km)
        self.fuzzy_systems['proximidad'] = {
            'ranges': {
                'cercana': (0, 5),
                'media': (4, 12),
                'lejana': (10, 20)
            },
            'universe': np.arange(0, 21, 1)
        }

        # Calidad de planes de evacuación (nivel 0–10)
        self.fuzzy_systems['evacuacion'] = {
            'ranges': {
                'deficiente': (0, 3),
                'aceptable': (2, 7),
                'buena': (6, 10)
            },
            'universe': np.arange(0, 11, 1)
        }
        
    def crisp_to_fuzzy_state(self, variable, crisp_value):
        """Convierte un valor crisp a estado lingüístico difuso"""
        if variable not in self.fuzzy_systems:
            return 'medio'  # Estado por defecto
        
        ranges = self.fuzzy_systems[variable]['ranges']
        max_membership = 0
        best_state = list(ranges.keys())[0]
        
        for state, (low, high) in ranges.items():
            # Calcular membresía triangular simple
            if low <= crisp_value <= high:
                if crisp_value <= (low + high) / 2:
                    membership = (crisp_value - low) / ((low + high) / 2 - low) if (low + high) / 2 != low else 1
                else:
                    membership = (high - crisp_value) / (high - (low + high) / 2) if high != (low + high) / 2 else 1
                
                if membership > max_membership:
                    max_membership = membership
                    best_state = state
        
        return best_state
    
    def fuzzy_inference(self, evidence_crisp, target_variable='riesgo', verbose=False):
        """
        Realizar inferencia difusa completa en la red
        
        Args:
            evidence_crisp: Diccionario con evidencia en valores crisp
            target_variable: Variable objetivo para la inferencia
            verbose: Si mostrar información detallada
            
        Returns:
            Distribución difusa para la variable objetivo
        """
        if verbose:
            print("🌋 INICIANDO INFERENCIA DIFUSA BAYESIANA")
            print("=" * 50)
        
        # Paso 1: Convertir evidencia crisp a estados lingüísticos
        evidence_linguistic = {}
        for var, value in evidence_crisp.items():
            if var in self.fuzzy_systems:
                state = self.crisp_to_fuzzy_state(var, value)
                evidence_linguistic[var] = state
                if verbose:
                    print(f"📊 {var}: {value} → '{state}'")
        
        if verbose:
            print(f"\n🔍 Evidencia lingüística: {evidence_linguistic}")
        
        # Paso 2: Realizar inferencia difusa hacia adelante
        inferred_states = self._forward_fuzzy_inference(evidence_linguistic, verbose)
        
        # Paso 3: Obtener distribución difusa final
        if target_variable in inferred_states:
            result = inferred_states[target_variable]
        else:
            # Si no se pudo inferir, usar distribución a priori
            result = self.nodes[target_variable].fuzzy_prior if target_variable in self.nodes else {}
        
        if verbose:
            print(f"\n🎯 RESULTADO FINAL para '{target_variable}':")
            for state, fuzzy_prob in result.items():
                crisp_val = fuzzy_prob.defuzzify_centroid()
                print(f"   {state}: {fuzzy_prob} → crisp: {crisp_val:.3f}")
        
        return result
    
    def _forward_fuzzy_inference(self, evidence_linguistic, verbose=False):
        """Inferencia difusa hacia adelante usando propagación de creencias difusas"""
        inferred = evidence_linguistic.copy()
        
        # Inferir amenaza si tenemos sus padres
        amenaza_parents = ['sismicidad', 'gases', 'deformacion', 'historia']
        if all(parent in inferred for parent in amenaza_parents):
            parent_states = tuple(inferred[parent] for parent in amenaza_parents)
            
            if parent_states in self.nodes['amenaza'].fuzzy_cpd:
                inferred_amenaza = self.nodes['amenaza'].fuzzy_cpd[parent_states]
                if verbose:
                    print(f"\n⚡ Inferencia de AMENAZA:")
                    print(f"   Padres: {parent_states}")
                    for state, prob in inferred_amenaza.items():
                        print(f"   amenaza({state}): {prob}")
                
                # Seleccionar estado más probable (defuzzificación)
                best_state = max(inferred_amenaza.keys(), 
                               key=lambda s: inferred_amenaza[s].defuzzify_centroid())
                inferred['amenaza'] = best_state
            else:
                # Usar interpolación difusa para casos no definidos
                inferred_amenaza = self._interpolate_fuzzy_cpd('amenaza', parent_states)
                if verbose:
                    print(f"\n⚡ Inferencia de AMENAZA (interpolada):")
                    print(f"   Padres: {parent_states}")
                    for state, prob in inferred_amenaza.items():
                        print(f"   amenaza({state}): {prob}")
                
                best_state = max(inferred_amenaza.keys(), 
                               key=lambda s: inferred_amenaza[s].defuzzify_centroid())
                inferred['amenaza'] = best_state
        
        # Inferir vulnerabilidad si tenemos sus padres
        vuln_parents = ['densidad', 'preparacion', 'proximidad', 'evacuacion']
        if all(parent in inferred for parent in vuln_parents):
            parent_states = tuple(inferred[parent] for parent in vuln_parents)
            
            if parent_states in self.nodes['vulnerabilidad'].fuzzy_cpd:
                inferred_vuln = self.nodes['vulnerabilidad'].fuzzy_cpd[parent_states]
                if verbose:
                    print(f"\n🛡️ Inferencia de VULNERABILIDAD:")
                    print(f"   Padres: {parent_states}")
                    for state, prob in inferred_vuln.items():
                        print(f"   vulnerabilidad({state}): {prob}")
                
                best_state = max(inferred_vuln.keys(), 
                               key=lambda s: inferred_vuln[s].defuzzify_centroid())
                inferred['vulnerabilidad'] = best_state
            else:
                # Usar interpolación difusa para casos no definidos
                inferred_vuln = self._interpolate_fuzzy_cpd('vulnerabilidad', parent_states)
                if verbose:
                    print(f"\n🛡️ Inferencia de VULNERABILIDAD (interpolada):")
                    print(f"   Padres: {parent_states}")
                    for state, prob in inferred_vuln.items():
                        print(f"   vulnerabilidad({state}): {prob}")
                
                best_state = max(inferred_vuln.keys(), 
                               key=lambda s: inferred_vuln[s].defuzzify_centroid())
                inferred['vulnerabilidad'] = best_state
        
        # Inferir riesgo si tenemos amenaza y vulnerabilidad
        if 'amenaza' in inferred and 'vulnerabilidad' in inferred:
            parent_states = (inferred['amenaza'], inferred['vulnerabilidad'])
            
            if parent_states in self.nodes['riesgo'].fuzzy_cpd:
                riesgo_distribution = self.nodes['riesgo'].fuzzy_cpd[parent_states]
                if verbose:
                    print(f"\n🔥 Inferencia de RIESGO:")
                    print(f"   Padres: {parent_states}")
                
                return {'riesgo': riesgo_distribution}
            else:
                # Usar interpolación difusa para casos no definidos
                riesgo_distribution = self._interpolate_fuzzy_cpd('riesgo', parent_states)
                if verbose:
                    print(f"\n🔥 Inferencia de RIESGO (interpolada):")
                    print(f"   Padres: {parent_states}")
                
                return {'riesgo': riesgo_distribution}
        
        # Si no podemos inferir riesgo, usar distribución por defecto
        return {'riesgo': {
            'bajo': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'medio': TriangularFuzzyProbability(0.3, 0.4, 0.5),
            'alto': TriangularFuzzyProbability(0.2, 0.2, 0.4)
        }}
    
    def _interpolate_fuzzy_cpd(self, node_name, parent_states):
        """Interpolar CPD difusa para combinaciones no definidas"""
        if node_name not in self.nodes:
            return {}
        
        node = self.nodes[node_name]
        
        # Si no hay CPD definida, usar distribución por defecto
        if not node.fuzzy_cpd:
            if node_name == 'amenaza':
                return {
                    'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                    'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                    'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
                }
            elif node_name == 'vulnerabilidad':
                return {
                    'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                    'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                    'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
                }
            elif node_name == 'riesgo':
                return {
                    'bajo': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                    'medio': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                    'alto': TriangularFuzzyProbability(0.2, 0.3, 0.4)
                }
        
        # Buscar la regla más similar y usar interpolación simple
        best_match = None
        best_similarity = 0
        
        for rule_states, distribution in node.fuzzy_cpd.items():
            # Calcular similitud simple (conteo de estados coincidentes)
            similarity = sum(1 for a, b in zip(parent_states, rule_states) if a == b)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = distribution
        
        # Si encontramos una regla similar, usarla con ligera modificación
        if best_match:
            interpolated = {}
            for state, fuzzy_num in best_match.items():
                # Añadir ligera incertidumbre a la interpolación
                a, m, b = fuzzy_num.a, fuzzy_num.m, fuzzy_num.b
                # Expandir ligeramente el triángulo para reflejar incertidumbre
                interpolated[state] = TriangularFuzzyProbability(
                    max(0, a - 0.05), m, min(1, b + 0.05)
                )
            return interpolated
        
        # Si no hay reglas similares, usar distribución uniforme difusa
        num_states = len(node.states)
        uniform_prob = 1.0 / num_states
        
        result = {}
        for state in node.states:
            result[state] = TriangularFuzzyProbability(
                max(0, uniform_prob - 0.1),
                uniform_prob,
                min(1, uniform_prob + 0.1)
            )
        
        return result
    
    def defuzzify_distribution(self, fuzzy_distribution, method='centroid'):
        """
        Defuzzificar una distribución difusa a un valor crisp
        
        Args:
            fuzzy_distribution: Diccionario con estados y números difusos
            method: Método de defuzzificación ('centroid', 'mean_of_max')
            
        Returns:
            Valor crisp defuzzificado
        """
        # Verificar que la distribución no esté vacía
        if not fuzzy_distribution:
            return 5.0  # Valor por defecto
        
        if method == 'centroid':
            total_weight = 0
            weighted_sum = 0
            
            for state, fuzzy_num in fuzzy_distribution.items():
                # Mapear estado a valor numérico
                state_value = self._state_to_numeric(state)
                weight = fuzzy_num.defuzzify_centroid()
                
                total_weight += weight
                weighted_sum += state_value * weight
            
            return weighted_sum / total_weight if total_weight > 0 else 5.0
        
        elif method == 'mean_of_max':
            if not fuzzy_distribution.values():
                return 5.0
            
            max_prob = max(fuzzy_distribution.values(), 
                          key=lambda x: x.defuzzify_centroid())
            max_states = [state for state, prob in fuzzy_distribution.items() 
                         if prob.defuzzify_centroid() == max_prob.defuzzify_centroid()]
            
            # Promedio de estados con máxima probabilidad
            return np.mean([self._state_to_numeric(state) for state in max_states])
        
        return 5.0  # Valor por defecto
    
    def _state_to_numeric(self, state):
        """Mapear estados lingüísticos a valores numéricos"""
        mapping = {
            'bajo': 2, 'baja': 2, 'muy bajo': 1, 'muy baja': 1,
            'medio': 5, 'media': 5, 'normal': 5,
            'alto': 8, 'alta': 8, 'muy alto': 9, 'muy alta': 9,
            'nula': 1, 'leve': 4, 'significativa': 8,
            'lejana': 8, 'cercana': 2,
            'inexistente': 1, 'parcial': 5, 'completo': 9,
            'elevada': 8
        }
        return mapping.get(state, 5)  # Valor por defecto

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