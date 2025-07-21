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