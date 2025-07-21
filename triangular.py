# Representación de una probabilidad difusa triangular
class TriangularFuzzyProbability:
    def __init__(self, a, m, b):
        self.a = a
        self.m = m
        self.b = b

    def __repr__(self):
        return f"({self.a}, {self.m}, {self.b})"
