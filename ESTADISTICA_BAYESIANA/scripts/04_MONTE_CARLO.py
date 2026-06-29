"""
Nombre del módulo: 04_MONTE_CARLO.py
Descripción: Script de desarrollo para ejercicios de Estadística Bayesiana.
            Aplicación del método de Monte Carlo en cáluclo numérico
            para demostrar precisión y velocidad de cálculo.

Autor: Hernández González Genesis
Fecha: 28/06/2026
Versión: 1.0

Dependencias: 
    - numpy: Operaciones numéricas y generación de números aleatorios
    - matplotlib.pyplot: Visualización de distribuciones
    - scipy.stats: Funciones estadísticas y distribuciones de probabilidad

Contenido:
    15. Apriximacion de pi mediante simulación.
    16. Axpliximacion de función Gaussiana en el intervalo [0, 1]
    17. Estimación VaR mediante Monte Carlo.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate

np.random.seed(42)

""" ACTIVIDADES """
# 15.
print("APROXIMACIÓN DE π")
n = 50000
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
dentro = x**2 + y**2 <= 1
pi_aprox = 4 * np.sum(dentro) / n
print(f"π ≈ {pi_aprox:.6f} (error: {abs(pi_aprox - np.pi):.6f})")

# 16.
print("\nINTEGRAL GAUSSIANA")
x_int = np.random.uniform(0, 1, n)
est = np.exp(-x_int**2)
integral = np.mean(est)
valor_real = integrate.quad(lambda x: np.exp(-x**2), 0, 1)[0]
ic = integral + np.array([-1.96, 1.96]) * np.std(est) / np.sqrt(n)
print(f"Monte Carlo: {integral:.6f} ± {np.std(est)/np.sqrt(n):.6f}")
print(f"Valor real:  {valor_real:.6f}")
print(f"IC 95%: [{ic[0]:.6f}, {ic[1]:.6f}]")

# 17.
print("\nVALUE AT RISK (VaR)")
inversion = 1_000_000
retornos = np.random.normal(0.0005, 0.015, (50000, 5))
perdidas = inversion * (1 - np.prod(1 + retornos, axis=1))
var_95 = np.percentile(perdidas, 95)
cvar_95 = np.mean(perdidas[perdidas >= var_95])
print(f"VaR 95% (5 días): ${var_95:,.2f}")
print(f"CVaR 95%: ${cvar_95:,.2f}")

""" GRÁFICOS """
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 15.
axes[0].scatter(x[::20], y[::20], c=dentro[::20], s=1, alpha=0.5, cmap='coolwarm')
axes[0].add_patch(plt.Circle((0,0), 1, fill=False, color='black'))
axes[0].set_aspect('equal')
axes[0].set_title(f'π ≈ {pi_aprox:.4f}')

# 16.
axes[1].hist(est, bins=30, density=True, alpha=0.7, color='green')
axes[1].axvline(valor_real, color='red', linestyle='--', label=f'Real: {valor_real:.4f}')
axes[1].axvline(integral, color='blue', linestyle='--', label=f'MC: {integral:.4f}')
axes[1].set_title('∫₀¹ e^(-x²) dx')
axes[1].legend()

# 17.
axes[2].hist(perdidas, bins=50, density=True, alpha=0.7, color='red')
axes[2].axvline(var_95, color='black', linewidth=2, label=f'VaR: ${var_95:,.0f}')
axes[2].axvline(cvar_95, color='darkred', linestyle='--', label=f'CVaR: ${cvar_95:,.0f}')
axes[2].set_title('VaR 95% - Pérdidas')
axes[2].legend()

plt.tight_layout()
plt.show()