"""
Nombre del módulo: 02_ENCUESTAS_Y_ELECCIONES.py
Descripción: Script de desarrollo para ejercicios de Estadística Bayesiana.
            Implementa simulaciones para experimentos con la posterior
            y se realiza un enfoque frecuentista contra el bayesiano
            para hacer estimaciones probabilisticas.

Autor: Hernández González Genesis
Fecha: 28/06/2026
Versión: 1.0

Dependencias: 
    - numpy: Operaciones numéricas y generación de números aleatorios
    - matplotlib.pyplot: Visualización de distribuciones
    - scipy.stats: Funciones estadísticas y distribuciones de probabilidad

Contenido:
    6. Encuesta: 540 de 1000 apoyan a un candidato.
    7. Simmulación de 100,000 elecciones usando la posterior.
    8. Combinación de 3 encuestas distintas.
    9. Comparación de enfoque frecuentista y bayesiano.
    10. Estimación de la pobabilidad de victoria.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

""" DESARROLLO DE LAS ACTIVIDADES """

# 6. 
n, k = 1000, 540
alpha_prior, beta_prior = 1, 1
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)
p_victoria = 1 - stats.beta.cdf(0.5, alpha_post, beta_post)

# 7. 
sims = np.random.beta(alpha_post, beta_post, 100000)

# 8.
k_comb = 540 + 270 + 120
n_comb = 1000 + 500 + 200
alpha_comb = 1 + k_comb
beta_comb = 1 + (n_comb - k_comb)

# 9.
p_hat = k/n
ic_f = p_hat + np.array([-1.96, 1.96]) * np.sqrt(p_hat*(1-p_hat)/n)
ic_b = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
media_b = alpha_post/(alpha_post+beta_post)
x = [0, 1]
alturas = [p_hat, media_b]
errores = [[alturas[0]-ic_f[0], alturas[1]-ic_b[0]],
           [ic_f[1]-alturas[0], ic_b[1]-alturas[1]]]

# 10.
umbrales = np.linspace(0.45, 0.6, 50)
p_vic = [1-stats.beta.cdf(u, alpha_post, beta_post) for u in umbrales]
p_vic_comb = [1-stats.beta.cdf(u, alpha_comb, beta_comb) for u in umbrales]

""" GRÁFICOS """

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
theta = np.linspace(0.4, 0.7, 300)

# 6. 
axes[0,0].plot(theta, stats.beta.pdf(theta, alpha_post, beta_post), 'b-', label='Posterior')
axes[0,0].fill_between(theta, stats.beta.pdf(theta, alpha_post, beta_post),
                        where=(theta>0.5), alpha=0.3, color='blue', label='P(victoria)')
axes[0,0].axvline(0.5, color='red', linestyle='--', label='Umbral 50%')
axes[0,0].set_title('1. Posterior (540/1000)')
axes[0,0].set_xlabel('Proporción de apoyo')
axes[0,0].set_ylabel('Densidad')
axes[0,0].legend(fontsize=8)

# 7. 
axes[0,1].hist(sims, bins=40, alpha=0.7, color='green', edgecolor='black', label='Simulaciones')
axes[0,1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Umbral 50%')
axes[0,1].set_title(f'2. 100,000 Simulaciones\nP(victoria)={np.mean(sims>0.5):.1%}')
axes[0,1].set_xlabel('Proporción simulada')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].legend(fontsize=8)

# 8. 
for n_e, k_e, c, lab in [(1000,540,'blue','Grande (1000)'),
                          (500,270,'green','Mediana (500)'),
                          (200,120,'orange','Pequeña (200)')]:
    a, b = 1+k_e, 1+(n_e-k_e)
    axes[0,2].plot(theta, stats.beta.pdf(theta, a, b), color=c, alpha=0.5, label=lab)
axes[0,2].plot(theta, stats.beta.pdf(theta, alpha_comb, beta_comb), 'r-', linewidth=2.5, label='Combinada')
axes[0,2].set_title('3. Combinación de 3 encuestas')
axes[0,2].set_xlabel('Proporción de apoyo')
axes[0,2].set_ylabel('Densidad')
axes[0,2].legend(fontsize=7)

# 9.
bar1 = axes[1,0].bar([0], [alturas[0]], color='blue', alpha=0.7, label='Frecuentista')
bar2 = axes[1,0].bar([1], [alturas[1]], color='red', alpha=0.7, label='Bayesiano')
axes[1,0].errorbar(x, alturas, yerr=errores, fmt='none', capsize=10, color='black', label='IC 95%')
axes[1,0].set_xticks([0, 1])
axes[1,0].set_xticklabels(['Frecuentista', 'Bayesiano'])
axes[1,0].set_ylabel('Proporción')
axes[1,0].set_title('4. IC 95% comparados')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(True, alpha=0.3, axis='y')

# 10. 
axes[1,1].plot(umbrales, p_vic, 'b-', linewidth=2, label='1 encuesta (540/1000)')
axes[1,1].plot(umbrales, p_vic_comb, 'r-', linewidth=2, label='3 encuestas combinadas')
axes[1,1].axvline(0.5, color='black', linestyle='--', label='Umbral 50%')
axes[1,1].axhline(0.5, color='black', linestyle='--', alpha=0.5)
axes[1,1].set_xlabel('Umbral de victoria')
axes[1,1].set_ylabel('P(proporción > umbral)')
axes[1,1].set_title('5. Probabilidad de victoria')
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3)

# Resumen
axes[1,2].axis('off')
axes[1,2].text(0.1, 0.5, 
    f"RESUMEN\n\n"
    f"1 Encuesta:\n"
    f"  Posterior: Beta({alpha_post},{beta_post})\n"
    f"  Media: {media_b:.3f}\n"
    f"  P(victoria): {p_victoria:.1%}\n\n"
    f"3 Encuestas:\n"
    f"  Posterior: Beta({alpha_comb},{beta_comb})\n"
    f"  Media: {alpha_comb/(alpha_comb+beta_comb):.3f}\n"
    f"  P(victoria): {1-stats.beta.cdf(0.5,alpha_comb,beta_comb):.1%}",
    fontsize=9, fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.show()