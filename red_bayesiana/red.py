import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from red_bayesiana.triangular import TriangularFuzzyProbability
from red_bayesiana.nodo import FuzzyBayesianNode 

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
        """Configurar CPD difusa para amenaza con probabilidades mejoradas"""
        fuzzy_rules = {
            # Combinaciones extremas
            ('alta', 'elevada', 'significativa', 'alta'): {
                'baja': TriangularFuzzyProbability(0.0, 0.02, 0.05),
                'media': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'alta': TriangularFuzzyProbability(0.8, 0.88, 0.95)
            },
            ('baja', 'normal', 'nula', 'baja'): {
                'baja': TriangularFuzzyProbability(0.85, 0.9, 0.95),
                'media': TriangularFuzzyProbability(0.05, 0.08, 0.1),
                'alta': TriangularFuzzyProbability(0.0, 0.02, 0.05)
            },
            
            # Combinaciones con alta sismicidad
            ('alta', 'elevada', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.3, 0.35, 0.4)
            },
            ('alta', 'normal', 'significativa', 'alta'): {
                'baja': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'alta': TriangularFuzzyProbability(0.4, 0.5, 0.6)
            },
            
            # Combinaciones con media sismicidad
            ('media', 'elevada', 'significativa', 'alta'): {
                'baja': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.3, 0.4, 0.5)
            },
            ('media', 'normal', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'media': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alta': TriangularFuzzyProbability(0.1, 0.15, 0.2)
            },
            ('media', 'elevada', 'nula', 'baja'): {
                'baja': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'media': TriangularFuzzyProbability(0.3, 0.35, 0.4),
                'alta': TriangularFuzzyProbability(0.05, 0.1, 0.15)
            },
            
            # Combinaciones con baja sismicidad pero otros factores altos
            ('baja', 'elevada', 'significativa', 'alta'): {
                'baja': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.2, 0.3, 0.4)
            },
            ('baja', 'normal', 'significativa', 'alta'): {
                'baja': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'alta': TriangularFuzzyProbability(0.1, 0.15, 0.2)
            },
            
            # Casos intermedios adicionales
            ('alta', 'normal', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.2, 0.25, 0.3),
                'media': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alta': TriangularFuzzyProbability(0.15, 0.2, 0.25)
            },
            ('media', 'elevada', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.15, 0.2, 0.25),
                'media': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alta': TriangularFuzzyProbability(0.2, 0.25, 0.3)
            },
            
            # Caso especial: baja sismicidad pero gases elevados y deformación
            ('baja', 'elevada', 'leve', 'media'): {
                'baja': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'media': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'alta': TriangularFuzzyProbability(0.1, 0.15, 0.2)
            }
        }
        self.nodes['amenaza'].set_fuzzy_cpd(fuzzy_rules)
    
    def _setup_vulnerabilidad_fuzzy_cpd(self):
        """Configurar CPD difusa para vulnerabilidad con probabilidades mejoradas"""
        fuzzy_rules = {
            # Escenario peor caso: alta densidad, nula preparación, cercanía y sin evacuación
            ('alta', 'muy bajo', 'cercana', 'inexistente'): {
                'baja': TriangularFuzzyProbability(0.0, 0.02, 0.05),
                'media': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'alta': TriangularFuzzyProbability(0.8, 0.88, 0.95)
            },
            
            # Escenario mejor caso: baja densidad, máxima preparación, lejanía y evacuación completa
            ('baja', 'muy alto', 'lejana', 'completo'): {
                'baja': TriangularFuzzyProbability(0.85, 0.9, 0.95),
                'media': TriangularFuzzyProbability(0.05, 0.08, 0.12),
                'alta': TriangularFuzzyProbability(0.0, 0.02, 0.05)
            },
            
            # Alta densidad pero buena preparación y evacuación
            ('alta', 'alto', 'cercana', 'completo'): {
                'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.1, 0.15, 0.2)
            },
            
            # Densidad media con preparación media y evacuación parcial
            ('media', 'medio', 'media', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.25, 0.35, 0.45),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.15, 0.2, 0.25)
            },
            
            # Alta densidad con preparación media
            ('alta', 'medio', 'media', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.3, 0.35, 0.4)
            },
            
            # Baja densidad pero mala preparación
            ('baja', 'bajo', 'lejana', 'inexistente'): {
                'baja': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'media': TriangularFuzzyProbability(0.3, 0.35, 0.4),
                'alta': TriangularFuzzyProbability(0.05, 0.1, 0.15)
            },
            
            # Densidad media con preparación baja
            ('media', 'bajo', 'cercana', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.15, 0.2, 0.25),
                'media': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alta': TriangularFuzzyProbability(0.2, 0.25, 0.3)
            },
            
            # Todos los factores en nivel medio
            ('media', 'medio', 'media', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'media': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alta': TriangularFuzzyProbability(0.1, 0.15, 0.2)
            },
            
            # Alta densidad pero lejanía compensatoria
            ('alta', 'medio', 'lejana', 'parcial'): {
                'baja': TriangularFuzzyProbability(0.3, 0.4, 0.5),
                'media': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alta': TriangularFuzzyProbability(0.1, 0.15, 0.2)
            }
        }
        self.nodes['vulnerabilidad'].set_fuzzy_cpd(fuzzy_rules)

    def _setup_riesgo_fuzzy_cpd(self):
        """Configurar CPD difusa para riesgo con mayor granularidad"""
        fuzzy_rules = {
            # Escenarios base
            ('baja', 'baja'): {
                'bajo': TriangularFuzzyProbability(0.75, 0.85, 0.92),
                'medio': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'alto': TriangularFuzzyProbability(0.0, 0.03, 0.06)
            },
            ('baja', 'media'): {
                'bajo': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'medio': TriangularFuzzyProbability(0.25, 0.35, 0.45),
                'alto': TriangularFuzzyProbability(0.05, 0.1, 0.15)
            },
            ('baja', 'alta'): {
                'bajo': TriangularFuzzyProbability(0.25, 0.35, 0.45),
                'medio': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'alto': TriangularFuzzyProbability(0.15, 0.2, 0.25)
            },
            ('media', 'baja'): {
                'bajo': TriangularFuzzyProbability(0.4, 0.5, 0.6),
                'medio': TriangularFuzzyProbability(0.35, 0.45, 0.55),
                'alto': TriangularFuzzyProbability(0.05, 0.1, 0.15)
            },
            ('media', 'media'): {
                'bajo': TriangularFuzzyProbability(0.15, 0.25, 0.35),
                'medio': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alto': TriangularFuzzyProbability(0.15, 0.25, 0.35)
            },
            ('media', 'alta'): {
                'bajo': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'medio': TriangularFuzzyProbability(0.35, 0.45, 0.55),
                'alto': TriangularFuzzyProbability(0.4, 0.5, 0.6)
            },
            ('alta', 'baja'): {
                'bajo': TriangularFuzzyProbability(0.15, 0.25, 0.35),
                'medio': TriangularFuzzyProbability(0.45, 0.55, 0.65),
                'alto': TriangularFuzzyProbability(0.15, 0.25, 0.35)
            },
            ('alta', 'media'): {
                'bajo': TriangularFuzzyProbability(0.05, 0.1, 0.15),
                'medio': TriangularFuzzyProbability(0.25, 0.35, 0.45),
                'alto': TriangularFuzzyProbability(0.5, 0.6, 0.7)
            },
            ('alta', 'alta'): {
                'bajo': TriangularFuzzyProbability(0.0, 0.02, 0.05),
                'medio': TriangularFuzzyProbability(0.1, 0.15, 0.2),
                'alto': TriangularFuzzyProbability(0.75, 0.85, 0.95)
            },
            
            # Casos especiales
            ('media-alta', 'baja'): {  # Transición entre media y alta amenaza
                'bajo': TriangularFuzzyProbability(0.1, 0.2, 0.3),
                'medio': TriangularFuzzyProbability(0.5, 0.6, 0.7),
                'alto': TriangularFuzzyProbability(0.2, 0.3, 0.4)
            },
            ('alta', 'media-alta'): {  # Vulnerabilidad en transición
                'bajo': TriangularFuzzyProbability(0.0, 0.05, 0.1),
                'medio': TriangularFuzzyProbability(0.2, 0.3, 0.4),
                'alto': TriangularFuzzyProbability(0.6, 0.7, 0.8)
            },
            ('baja-media', 'alta'): {  # Amenaza en transición
                'bajo': TriangularFuzzyProbability(0.15, 0.25, 0.35),
                'medio': TriangularFuzzyProbability(0.45, 0.55, 0.65),
                'alto': TriangularFuzzyProbability(0.2, 0.3, 0.4)
            }
        }
        self.nodes['riesgo'].set_fuzzy_cpd(fuzzy_rules)

    def _setup_fuzzy_systems(self):
        """Configuración optimizada de sistemas difusos con correspondencia exacta a los estados de los nodos"""
        
        # Sismicidad (eventos/día) - 3 estados
        self.fuzzy_systems['sismicidad'] = {
            'ranges': {
                'baja': (0, 4),       # 0-4 eventos/día
                'media': (3, 10),     # 3-10 eventos/día
                'alta': (8, 20)       # 8+ eventos/día
            },
            'universe': np.arange(0, 21, 1)  # Rango 0-20
        }

        # Emisión de gases (ppm SO₂) - 2 estados
        self.fuzzy_systems['gases'] = {
            'ranges': {
                'normal': (0, 1200),   # 0-1200 ppm
                'elevada': (800, 5000) # 800+ ppm
            },
            'universe': np.arange(0, 5001, 100)  # Rango 0-5000
        }

        # Deformación del terreno (mm) - 3 estados
        self.fuzzy_systems['deformacion'] = {
            'ranges': {
                'nula': (0, 4),        # 0-4 mm
                'leve': (3, 20),       # 3-20 mm
                'significativa': (15, 50) # 15+ mm
            },
            'universe': np.arange(0, 51, 1)  # Rango 0-50
        }

        # Actividad histórica (0-10) - 3 estados
        self.fuzzy_systems['historia'] = {
            'ranges': {
                'baja': (0, 3),        # 0-3
                'media': (2, 8),       # 2-8
                'alta': (6, 10)        # 6-10
            },
            'universe': np.arange(0, 11, 1)  # Rango 0-10
        }

        # Densidad poblacional (personas/km²) - 3 estados
        self.fuzzy_systems['densidad'] = {
            'ranges': {
                'baja': (0, 6000),     # 0-6000
                'media': (4000, 12000), # 4000-12000
                'alta': (8000, 30000)   # 8000+
            },
            'universe': np.arange(0, 30001, 500)  # Rango 0-30000
        }

        # Preparación comunitaria (0-5) - 5 estados
        self.fuzzy_systems['preparacion'] = {
            'ranges': {
                'muy bajo': (0, 1),    # 0-1
                'bajo': (0.5, 2),      # 0.5-2
                'medio': (1.5, 3.5),   # 1.5-3.5
                'alto': (3, 4.5),      # 3-4.5
                'muy alto': (4, 5)     # 4-5
            },
            'universe': np.arange(0, 5.1, 0.1)  # Rango 0-5
        }

        # Proximidad al cráter (km) - 3 estados
        self.fuzzy_systems['proximidad'] = {
            'ranges': {
                'cercana': (0, 6),     # 0-6 km
                'media': (4, 14),      # 4-14 km
                'lejana': (10, 20)     # 10-20 km
            },
            'universe': np.arange(0, 21, 1)  # Rango 0-20
        }

        # Planes de evacuación (0-10) - 3 estados
        self.fuzzy_systems['evacuacion'] = {
            'ranges': {
                'inexistente': (0, 3), # 0-3
                'parcial': (2, 7),      # 2-7
                'completo': (5, 10)     # 5-10
            },
            'universe': np.arange(0, 11, 1)  # Rango 0-10
        }
        
    def crisp_to_fuzzy_state(self, variable, crisp_value, verbose=False):
        """Convierte un valor crisp a estado lingüístico difuso con mejor manejo de bordes"""
        if variable not in self.fuzzy_systems:
            return 'medio'  # Estado por defecto
        
        ranges = self.fuzzy_systems[variable]['ranges']
        max_membership = -1  # Inicializar con valor negativo
        best_state = list(ranges.keys())[0]
        
        for state, (low, high) in ranges.items():
            if verbose:
                print("Evaluando estado '%s' con rango (%s, %s) para valor crisp %.2f" % (state, low, high, crisp_value))
            membership = 0.0
            
            # Si el valor está dentro del rango
            if low <= crisp_value <= high:
                midpoint = (low + high) / 2
                
                # Calcular membresía triangular
                if crisp_value <= midpoint:
                    if midpoint != low:  # Evitar división por cero
                        membership = (crisp_value - low) / (midpoint - low)
                    else:
                        membership = 1.0
                else:
                    if high != midpoint:  # Evitar división por cero
                        membership = (high - crisp_value) / (high - midpoint)
                    else:
                        membership = 1.0
                        
                # Asegurar que la membresía esté en [0, 1]
                membership = max(0.0, min(1.0, membership))
                
                # Manejo especial para valores en los extremos
                if crisp_value == high and high == self.fuzzy_systems[variable]['universe'][-1]:
                    membership = 1.0  # Máxima membresía si es el valor máximo posible
                elif crisp_value == low and low == self.fuzzy_systems[variable]['universe'][0]:
                    membership = 1.0  # Máxima membresía si es el valor mínimo posible
            
            if verbose:
                print(f"   Membresía para '{state}': {membership:.3f}")
            # Actualizar el mejor estado si encontramos mayor membresía
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
        # Validación de entrada
        if not evidence_crisp or not isinstance(evidence_crisp, dict):
            raise ValueError("evidence_crisp debe ser un diccionario no vacío")
        
        if target_variable not in self.nodes:
            raise ValueError(f"Variable objetivo '{target_variable}' no existe en la red")
        
        # Validar que los valores sean numéricos
        for var, value in evidence_crisp.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"El valor para '{var}' debe ser numérico, recibido: {type(value)}")
                
        if verbose:
            print("🌋 INICIANDO INFERENCIA DIFUSA BAYESIANA")
            print("=" * 50)
        
        # Paso 1: Convertir evidencia crisp a estados lingüísticos
        evidence_linguistic = {}
        for var, value in evidence_crisp.items():
            if var in self.fuzzy_systems:
                state = self.crisp_to_fuzzy_state(var, value, verbose)
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
            
            if verbose:
                print(f"🔍 Evaluando amenaza con padres: {parent_states}")
                
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
            
            if verbose:
                print(f"🔍 Evaluando vulnerabilidad con padres: {parent_states}")
                
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
            
            if verbose:
                print(f"🔍 Evaluando riesgo con padres: {parent_states}")
                
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
        
        # Si no podemos inferir riesgo, usar distribución por defecto normalizada
        return {'riesgo': {
            'bajo': TriangularFuzzyProbability(0.4, 0.5, 0.6),
            'medio': TriangularFuzzyProbability(0.25, 0.35, 0.45),
            'alto': TriangularFuzzyProbability(0.15, 0.25, 0.35)
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
    
    def _normalize_fuzzy_distribution(self, fuzzy_distribution):
        """Normalizar una distribución difusa para que sume aproximadamente 1.0"""
        if not fuzzy_distribution:
            return fuzzy_distribution
            
        # Calcular la suma total de los centroides
        total_centroid = sum(fuzzy_num.defuzzify_centroid() 
                           for fuzzy_num in fuzzy_distribution.values())
        
        if total_centroid == 0:
            return fuzzy_distribution
            
        # Normalizar cada número difuso
        normalized = {}
        for state, fuzzy_num in fuzzy_distribution.items():
            factor = fuzzy_num.defuzzify_centroid() / total_centroid
            normalized[state] = TriangularFuzzyProbability(
                fuzzy_num.a * factor,
                fuzzy_num.m * factor, 
                fuzzy_num.b * factor
            )
        
        return normalized
    
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
    
    def diagnose_network(self):
        """Método para diagnosticar problemas en la red bayesiana"""
        issues = []
        
        # Verificar nodos
        for node_name, node in self.nodes.items():
            if not node.states:
                issues.append(f"Nodo '{node_name}' no tiene estados definidos")
            
            # Verificar CPDs para nodos con padres
            if node.parents and not node.fuzzy_cpd:
                issues.append(f"Nodo '{node_name}' tiene padres pero no CPD definida")
            
            # Verificar distribución a priori para nodos sin padres
            if not node.parents and not node.fuzzy_prior:
                issues.append(f"Nodo raíz '{node_name}' no tiene distribución a priori")
        
        # Verificar sistemas difusos
        for var_name in self.nodes.keys():
            if var_name not in self.fuzzy_systems:
                issues.append(f"Variable '{var_name}' no tiene sistema difuso definido")
        
        return issues
    
    def get_network_info(self):
        """Obtener información resumida de la red"""
        info = {
            'total_nodes': len(self.nodes),
            'root_nodes': [],
            'intermediate_nodes': [],
            'leaf_nodes': [],
            'total_cpd_rules': 0
        }
        
        for node_name, node in self.nodes.items():
            if not node.parents:
                info['root_nodes'].append(node_name)
            elif node_name in ['amenaza', 'vulnerabilidad']:
                info['intermediate_nodes'].append(node_name)
            else:
                info['leaf_nodes'].append(node_name)
                
            if node.fuzzy_cpd:
                info['total_cpd_rules'] += len(node.fuzzy_cpd)
        
        return info