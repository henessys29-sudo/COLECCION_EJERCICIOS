"""
Nombre del módulo: 01_FUNDAMENTOS.py
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
    1. Análisis de una moneda con prior Beta(1,1)
    2. Comparación de diferentes distribuciones prior Beta
    3. Comparación bayesiana entre dos monedas
    4. Cálculo de intervalos creíbles al 95%
    5. Efecto del tamaño muestral en la precisión de estimaciones
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

""" ACTIVIDADES """

#1
print("1. POSTERIOR CON PRIOR BETA(1,1)")
n, k = 50, 32
alpha_prior, beta_prior = 1, 1
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"Media: {alpha_post/(alpha_post+beta_post):.4f}")

#2
print("\n2. COMPARACIÓN DE PRIORS")
priors = [(1,1), (5,5), (20,20)]
for alpha_p, beta_p in priors:
    a_post = alpha_p + k
    b_post = beta_p + (n - k)
    media = a_post/(a_post+b_post)
    print(f"Beta({alpha_p},{beta_p}) -> Beta({a_post},{b_post}), media={media:.4f}")

#3
print("\n3. MONEDA A vs MONEDA B")
a_a, b_a = 1+32, 1+18
a_b, b_b = 1+25, 1+25
p_a = np.random.beta(a_a, b_a, 50000)
p_b = np.random.beta(a_b, b_b, 50000)
prob = np.mean(p_a > p_b)
print(f"P(p_a > p_b) = {prob:.4f}")

#4
print("\n4. INTERVALO CREÍBLE 95%")
ic_colas = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"Colas iguales: [{ic_colas[0]:.4f}, {ic_colas[1]:.4f}]")

# HPD
theta = np.linspace(0, 1, 5000)
densidad = stats.beta.pdf(theta, alpha_post, beta_post)
orden = np.argsort(densidad)[::-1]
acum = np.cumsum(densidad[orden]) / np.sum(densidad)
umbral = densidad[orden][np.searchsorted(acum, 0.95)]
hpd = theta[orden][densidad[orden] >= umbral]
print(f"HPD: [{hpd.min():.4f}, {hpd.max():.4f}]")

#5
print("\n5. EFECTO DEL TAMAÑO MUESTRAL")
for n_i in [10, 25, 50, 100, 200, 500]:
    k_i = int(0.64 * n_i)
    a_i, b_i = 1 + k_i, 1 + (n_i - k_i)
    ic = stats.beta.ppf([0.025, 0.975], a_i, b_i)
    print(f"n={n_i:3d}: IC=[{ic[0]:.4f}, {ic[1]:.4f}], long={ic[1]-ic[0]:.4f}")

""" GRÁFICOS """

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

# 1. 
theta = np.linspace(0, 1, 300)
axes[0,0].plot(theta, stats.beta.pdf(theta, 1, 1), 'b--', label='Prior')
axes[0,0].plot(theta, stats.beta.pdf(theta, alpha_post, beta_post), 'r-', label='Posterior')
axes[0,0].set_title('1. Posterior Beta(1,1)')
axes[0,0].legend(fontsize=7)

# 2. 
for (ap, bp), c in zip(priors, ['blue','green','orange']):
    axes[0,1].plot(theta, stats.beta.pdf(theta, ap+k, bp+(n-k)), 
                   color=c, label=f'Beta({ap},{bp})')
axes[0,1].set_title('2. Priors comparados')
axes[0,1].legend(fontsize=7)

# 3. 
axes[0,2].hist(p_a - p_b, bins=40, density=True, alpha=0.7, color='purple')
axes[0,2].axvline(0, color='black', linestyle='--')
axes[0,2].set_title(f'3. P(p_a>p_b)={prob:.3f}')

# 4. 
post = stats.beta.pdf(theta, alpha_post, beta_post)
axes[1,0].plot(theta, post, 'r-')
axes[1,0].fill_between(theta, post, where=(theta>=ic_colas[0]) & (theta<=ic_colas[1]), alpha=0.3, label='Colas')
axes[1,0].fill_between(theta, post, where=(theta>=hpd.min()) & (theta<=hpd.max()), alpha=0.3, label='HPD')
axes[1,0].set_title('4. IC 95% y HPD')
axes[1,0].legend(fontsize=7)

# 5. 
ns = [10,25,50,100,200,500]
longs = [stats.beta.ppf(0.975, 1+int(0.64*n), 1+n-int(0.64*n)) - 
         stats.beta.ppf(0.025, 1+int(0.64*n), 1+n-int(0.64*n)) for n in ns]
axes[1,1].plot(ns, longs, 'bo-')
axes[1,1].set_xlabel('n')
axes[1,1].set_ylabel('Longitud IC')
axes[1,1].set_title('5. Precisión vs n')
axes[1,2].axis('off')

plt.tight_layout()
plt.show()