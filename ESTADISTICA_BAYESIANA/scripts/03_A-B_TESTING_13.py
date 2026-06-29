"""
Nombre del módulo 03_A-B_TESTING_13.py
Descripción: Script de desarrollo para el punto 13 de la PARTE III.
            
    Descripción del Experimento

    Objetivo: Determinar si la implementación de "ejercicios interactivos" mejora la tasa de 
        finalización de un curso de Python frente al método tradicional de "lectura de PDFs".
    Hipótesis: Los estudiantes que utilizan ejercicios interactivos tendrán una 
        tasa de finalización significativamente mayor que aquellos que solo leen material teórico.
    Métricas:
        Métrica Principal: Tasa de finalización (binaria: 1 = completó, 0 = no completó).
        Métrica Secundaria: Tiempo promedio en la plataforma.
    Grupos:
        Grupo A (Control): Acceso a material teórico (PDFs).
        Grupo B (Variante): Acceso a ejercicios interactivos.
    Método Estadístico: Prueba t de Student para comparar medias o prueba Z de 
        proporciones para tasas de conversión.

Autor: Hernández González Genesis
Fecha: 28/06/2026
Versión: 1.0

Dependencias: 
    - numpy: Operaciones numéricas y generación de números aleatorios
    - matplotlib.pyplot: Visualización de distribuciones
    - scipy.stats: Funciones estadísticas y distribuciones de probabilidad
    - statsmodels: Modelos estadísticos

Contenido:
    1. Simulación de datos
    2. Cálculo de tasas
    3. Análisis estadístico con prueba Z
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

# 1. Simulación de datos
np.random.seed(42)
n = 500  # Estudiantes por grupo

# Grupo A: Tasa de conversión del 30%
grupo_a = np.random.binomial(1, 0.30, n)
# Grupo B: Tasa de conversión del 38%
grupo_b = np.random.binomial(1, 0.38, n)

# 2. Cálculo de tasas
tasa_a = np.mean(grupo_a)
tasa_b = np.mean(grupo_b)

print(f"Tasa de finalización Grupo A (Control): {tasa_a:.2%}")
print(f"Tasa de finalización Grupo B (Variante): {tasa_b:.2%}")

# 3. Análisis estadístico con prueba Z
count = np.array([sum(grupo_a), sum(grupo_b)])
nobs = np.array([n, n])

z_stat, p_value = proportions_ztest(count, nobs)

print(f"\nValor p: {p_value:.4f}")

if p_value < 0.05:
    print("Resultado: La diferencia es estadísticamente significativa.")
else:
    print("Resultado: No hay evidencia suficiente para afirmar que el cambio mejora los resultados.")

grupos = ['Grupo A (Control)', 'Grupo B (Variante)']
tasas = [tasa_a * 100, tasa_b * 100]  # Convertimos a porcentaje
colores = ['red', 'blue']

# GRÁFICO
plt.figure(figsize=(8, 6))
bars = plt.bar(grupos, tasas, color=colores, alpha=0.7)
plt.ylabel('Tasa de Finalización (%)')
plt.title('Comparación de Resultados: Metodología de Enseñanza')
plt.ylim(0, 50)  # Ajustar según sea necesario
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.show()