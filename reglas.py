
# Reglas para Amenaza Volcánica

# Entradas:
# Sismicidad: baja, media, alta
# Gases: bajos, medios, altos
# Deformación: baja, media, alta
# Histórico: baja, media, alta

# Salida:
# Amenaza volcánica: baja, media, alta

reglas_amenaza = [
    ("alta", "alta", "alta", "alta", "alta"),
    ("media", "media", "media", "media", "media"),
    ("baja", "baja", "baja", "baja", "baja"),
    ("alta", "media", "media", "alta", "alta"),
    ("media", "baja", "baja", "media", "baja"),
]


# Reglas para Vulnerabilidad Social

# Entradas:
# Densidad poblacional: baja, media, alta
# Nivel de preparación: 0–5 → muy bajo, bajo, medio, alto, muy alto
# Proximidad: lejana, media, cercana
# Planes evacuación: inexistente, parcial, completo

# Salida:
# Vulnerabilidad social: baja, media, alta

reglas_vulnerabilidad = [
    ("alta", "muy bajo", "cercana", "inexistente", "alta"),
    ("media", "bajo", "media", "parcial", "media"),
    ("baja", "alto", "lejana", "completo", "baja"),
    ("alta", "medio", "media", "parcial", "media"),
    ("media", "muy bajo", "cercana", "inexistente", "alta")
]

# 4.3 Reglas para Riesgo Volcánico
# Entradas:
# Amenaza volcánica: baja, media, alta
# Vulnerabilidad social: baja, media, alta

# Salida:
# Riesgo volcánico: bajo, medio, alto

reglas_riesgo = [
    ("alta", "alta", "alto"),
    ("media", "alta", "alto"),
    ("alta", "media", "alto"),
    ("media", "media", "medio"),
    ("baja", "alta", "medio"),
    ("baja", "baja", "bajo"),
    ("media", "baja", "medio"),
    ("alta", "baja", "medio"),
]