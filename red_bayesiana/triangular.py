# Representación de una probabilidad difusa triangular
import numpy as np

class TriangularFuzzyProbability:
    def __init__(self, a, m, b):
        """
        Número difuso triangular (a, m, b)
        a: límite inferior
        m: modal (más probable)
        b: límite superior
        """
        self.a = a
        self.m = m
        self.b = b

    def __repr__(self):
        return f"TFP({self.a:.3f}, {self.m:.3f}, {self.b:.3f})"
    
    def __str__(self):
        return f"({self.a:.3f}, {self.m:.3f}, {self.b:.3f})"

    def __add__(self, other):
        """Suma de números difusos triangulares"""
        if isinstance(other, TriangularFuzzyProbability):
            return TriangularFuzzyProbability(
                self.a + other.a,
                self.m + other.m,
                self.b + other.b
            )
        else:  # Suma con escalar
            return TriangularFuzzyProbability(
                self.a + other,
                self.m + other,
                self.b + other
            )

    def __mul__(self, other):
        """Multiplicación de números difusos triangulares"""
        if isinstance(other, TriangularFuzzyProbability):
            # Multiplicación aproximada para números positivos
            combinations = [
                self.a * other.a, self.a * other.b,
                self.b * other.a, self.b * other.b,
                self.m * other.m
            ]
            return TriangularFuzzyProbability(
                min(combinations),
                self.m * other.m,
                max(combinations)
            )
        else:  # Multiplicación con escalar
            if other >= 0:
                return TriangularFuzzyProbability(
                    self.a * other,
                    self.m * other,
                    self.b * other
                )
            else:
                return TriangularFuzzyProbability(
                    self.b * other,
                    self.m * other,
                    self.a * other
                )

    def __truediv__(self, other):
        """División de números difusos triangulares"""
        if isinstance(other, (int, float)):
            return self * (1/other)
        # División más compleja para números difusos no implementada
        raise NotImplementedError("División entre números difusos no implementada")

    def centroid(self):
        """Calcula el centroide del número difuso triangular"""
        return (self.a + self.m + self.b) / 3

    def normalize(self):
        """Normaliza el número difuso para que el modal sea máximo 1"""
        if self.m == 0:
            return TriangularFuzzyProbability(0, 0, 0)
        factor = 1.0 / self.m
        return TriangularFuzzyProbability(
            self.a * factor,
            1.0,
            self.b * factor
        )

    def membership(self, x):
        """Calcula el grado de membresía para un valor x"""
        if x <= self.a or x >= self.b:
            return 0.0
        elif x == self.m:
            return 1.0
        elif x < self.m:
            return (x - self.a) / (self.m - self.a)
        else:
            return (self.b - x) / (self.b - self.m)

    def alpha_cut(self, alpha):
        """Calcula el corte alfa del número difuso"""
        if alpha == 0:
            return (self.a, self.b)
        elif alpha == 1:
            return (self.m, self.m)
        else:
            left = self.a + alpha * (self.m - self.a)
            right = self.b - alpha * (self.b - self.m)
            return (left, right)

    def defuzzify_centroid(self):
        """Defuzzificación por método del centroide"""
        return self.centroid()

    def defuzzify_mean_of_max(self):
        """Defuzzificación por media de máximos"""
        return self.m

    @staticmethod
    def linguistic_to_fuzzy(linguistic_value, domain_range=(0, 10)):
        """Convierte valores lingüísticos a números difusos"""
        low, high = domain_range
        span = high - low
        
        if linguistic_value in ['muy bajo', 'muy baja']:
            return TriangularFuzzyProbability(low, low, low + span*0.2)
        elif linguistic_value in ['bajo', 'baja']:
            return TriangularFuzzyProbability(low, low + span*0.1, low + span*0.3)
        elif linguistic_value in ['medio', 'media']:
            return TriangularFuzzyProbability(low + span*0.2, low + span*0.5, low + span*0.8)
        elif linguistic_value in ['alto', 'alta']:
            return TriangularFuzzyProbability(low + span*0.7, low + span*0.9, high)
        elif linguistic_value in ['muy alto', 'muy alta']:
            return TriangularFuzzyProbability(low + span*0.8, high, high)
        else:
            # Valor por defecto
            return TriangularFuzzyProbability(low + span*0.3, low + span*0.5, low + span*0.7)
