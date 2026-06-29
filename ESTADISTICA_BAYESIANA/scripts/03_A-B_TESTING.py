"""
Nombre del módulo: 03_A-B_TESTING.py
Descripción: Script de desarrollo para ejercicios de Estadística Bayesiana.
            Implementa análisis de inferencia bayesiana para experimentos
            binomiales usando distribuciones Beta como prior y posterior.

Autor: Hernández González Genesis
Fecha: 28/06/2026
Versión: 1.0

Dependencias: 
    - numpy: Operaciones numéricas y generación de números aleatorios
    - matplotlib.pyplot: Visualización de distribuciones
    - scipy.stats: Funciones estadísticas y distribuciones de probabilidad

Contenido:
    11. A: 100/1000, B: 120/1000. Calculo de P(A > B)
    12. Estimación P(B - A > 0.02)
    14. Analisis de la sensibilidad de la prior.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

""" DESARROLLO DE LAS ACTIVIDADES """
# 11. DATOS
n_a, k_a = 1000, 100
n_b, k_b = 1000, 120 
alpha_a, beta_a = 1+k_a, 1+(n_a-k_a)
alpha_b, beta_b = 1+k_b, 1+(n_b-k_b)

# 12. Simulaciones
n_sim = 50000
p_a = np.random.beta(alpha_a, beta_a, n_sim)
p_b = np.random.beta(alpha_b, beta_b, n_sim)
dif = p_b - p_a

print(f"P(B > A) = {np.mean(p_b > p_a):.3f}") # 11.
print(f"P(B-A > 0.02) = {np.mean(dif > 0.02):.3f}") # 12.
print(f"Diferencia media: {np.mean(dif):.4f}")

# 14.
priors = [(1,1), (0.5,0.5), (5,5), (10,20), (20,10)]
for ap, bp in priors:
    a_a = ap+k_a; b_a = bp+(n_a-k_a)
    a_b = ap+k_b; b_b = bp+(n_b-k_b)
    p_ba = np.mean(np.random.beta(a_b, b_b, 30000) > np.random.beta(a_a, b_a, 30000))
    print(f"Prior Beta({ap},{bp}): P(B>A)={p_ba:.3f}")

""" GRÁFICOS """
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
theta = np.linspace(0.05, 0.20, 200)

# 11.
axes[0].plot(theta, stats.beta.pdf(theta, alpha_a, beta_a), 'b-', label='A (control)')
axes[0].plot(theta, stats.beta.pdf(theta, alpha_b, beta_b), 'r-', label='B (variante)')
axes[0].set_title('Posterior A vs B')
axes[0].legend()

# 12. 
axes[1].hist(dif, bins=40, alpha=0.7, color='purple')
axes[1].axvline(0, color='black', linestyle='--')
axes[1].axvline(0.02, color='red', linestyle='--', label='Umbral 0.02')
axes[1].set_title(f'Diferencia B-A\nP(B>A)={np.mean(p_b>p_a):.3f}')
axes[1].legend()

# 14. 
p_vals = []
nombres = []
for ap, bp in priors:
    a_a = ap+k_a; b_a = bp+(n_a-k_a)
    a_b = ap+k_b; b_b = bp+(n_b-k_b)
    p_vals.append(np.mean(np.random.beta(a_b, b_b, 30000) > np.random.beta(a_a, b_a, 30000)))
    nombres.append(f'Beta({ap},{bp})')

axes[2].bar(nombres, p_vals, color=plt.cm.viridis(np.linspace(0,1,5)))
axes[2].set_ylabel('P(B > A)')
axes[2].set_title('Sensibilidad al prior')
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


