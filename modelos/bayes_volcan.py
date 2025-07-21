from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def crear_red_bayesiana():
    # 1. Definir estructura
    modelo = DiscreteBayesianNetwork([
        ('sismicidad', 'amenaza'),
        ('gases', 'amenaza'),
        ('deformacion', 'amenaza'),
        ('historia', 'amenaza'),
        ('densidad', 'vulnerabilidad'),
        ('preparacion', 'vulnerabilidad')
    ])

    # 2. CPDs para nodos fuente
    cpd_sismicidad = TabularCPD(variable='sismicidad', variable_card=3, values=[[0.4], [0.4], [0.2]])
    cpd_gases = TabularCPD(variable='gases', variable_card=3, values=[[0.3], [0.5], [0.2]])
    cpd_deformacion = TabularCPD(variable='deformacion', variable_card=3, values=[[0.5], [0.3], [0.2]])
    cpd_historia = TabularCPD(variable='historia', variable_card=2, values=[[0.6], [0.4]])
    cpd_densidad = TabularCPD(variable='densidad', variable_card=3, values=[[0.3], [0.5], [0.2]])
    cpd_preparacion = TabularCPD(variable='preparacion', variable_card=3, values=[[0.2], [0.5], [0.3]])

    # 3. CPD de amenaza (simplificada)
    cpd_amenaza = TabularCPD(
        variable='amenaza', variable_card=3,
        evidence=['sismicidad', 'gases', 'deformacion', 'historia'],
        evidence_card=[3, 3, 3, 2],
        values=[
            [0.7]*54,  # baja
            [0.2]*54,  # media
            [0.1]*54   # alta
        ]
    )

    # 4. CPD de vulnerabilidad (también simplificada)
    cpd_vulnerabilidad = TabularCPD(
        variable='vulnerabilidad', variable_card=3,
        evidence=['densidad', 'preparacion'],
        evidence_card=[3, 3],
        values=[
            [0.6]*9,  # baja
            [0.3]*9,  # media
            [0.1]*9   # alta
        ]
    )

    # 5. Agregar CPDs al modelo
    modelo.add_cpds(
        cpd_sismicidad, cpd_gases, cpd_deformacion, cpd_historia,
        cpd_densidad, cpd_preparacion,
        cpd_amenaza, cpd_vulnerabilidad
    )

    modelo.check_model()
    return modelo

def probabilidad_a_valor_difuso(distribucion, etiquetas=['baja', 'media', 'alta'], dominio=[0, 5, 10]):
    """
    Convierte una distribución discreta bayesiana a un valor difuso (esperado).
    Por ejemplo: {'baja': 0.2, 'media': 0.5, 'alta': 0.3} → valor difuso ≈ 5.5
    """
    mapeo = dict(zip(etiquetas, dominio))  # {'baja': 0, 'media': 5, 'alta': 10}
    valor_esperado = sum(mapeo[estado] * prob for estado, prob in distribucion.items())
    return valor_esperado
