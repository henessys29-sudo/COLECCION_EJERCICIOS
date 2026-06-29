"""
Nombre del módulo: 05_MODELOS_AVANZADOS.py
Descripción: Script de desarrollo para ejercicios de Estadística Bayesiana.
            Implementación de modelos avanzados como regresión lineal
            bayesiana y Metropolis-Hastings.

Autor: Hernández González Genesis
Fecha: 28/06/2026
Versión: 1.0

Dependencias: 
    - numpy: Operaciones numéricas y generación de números aleatorios
    - matplotlib.pyplot: Visualización de distribuciones
    - scipy.stats: Funciones estadísticas y distribuciones de probabilidad
    - pandas : Manipulación de Data Frames requeridos para la regreción bayesiana

Contenido:
    18. Regresión lineal bayesiana.
    19. Implementación de Metropolis-Hastings.
    20. Implementación de Gibbs Sampling para una normal con parámetros desconocidos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
import pandas as pd

np.random.seed(42)

""" ACTIVIDADES """

# 18.
print("REGRESIÓN BAYESIANA LINEAL")

# Generar datos sintéticos realistas
n_datos = 100
X = np.random.normal(12, 3, n_datos)  
beta0_real, beta1_real = -10000, 5000  
sigma_real = 8000
y = beta0_real + beta1_real * X + np.random.normal(0, sigma_real, n_datos)

# Priors
prior_beta0 = stats.norm(0, 10000)
prior_beta1 = stats.norm(0, 5000)     
prior_sigma2 = stats.invgamma(3, scale=100000)  

# Diseño matricial
X_diseno = np.column_stack([np.ones(n_datos), X])

# Posterior analítica para regresión conjugada
# Posterior de sigma^2: Inv-Gamma
n = len(y)
k = X_diseno.shape[1]
beta_hat = np.linalg.inv(X_diseno.T @ X_diseno) @ X_diseno.T @ y
residuos = y - X_diseno @ beta_hat
SSR = np.sum(residuos**2)

# Parámetros posteriores
a_post = 3 + n/2
b_post = 100000 + SSR/2

# Muestrear de la posterior
n_muestras = 5000
sigma2_muestras = 1 / np.random.gamma(a_post, 1/b_post, n_muestras)
beta_muestras = np.zeros((n_muestras, 2))

for i in range(n_muestras):
    cov_beta = np.linalg.inv(X_diseno.T @ X_diseno) * sigma2_muestras[i]
    beta_muestras[i] = np.random.multivariate_normal(beta_hat, cov_beta)

print(f"Parámetros reales: β₀={beta0_real}, β₁={beta1_real}")
print(f"\nPosterior:")
print(f"β₀: media={np.mean(beta_muestras[:,0]):.2f}, IC 95%=[{np.percentile(beta_muestras[:,0],2.5):.2f}, {np.percentile(beta_muestras[:,0],97.5):.2f}]")
print(f"β₁: media={np.mean(beta_muestras[:,1]):.2f}, IC 95%=[{np.percentile(beta_muestras[:,1],2.5):.2f}, {np.percentile(beta_muestras[:,1],97.5):.2f}]")
print(f"σ²: media={np.mean(sigma2_muestras):.2f}")

# 19.
print("\nMETROPOLIS-HASTINGS")
print("Estimando distribución posterior de una mezcla de normales")

def target_distribution(x):
    """Distribución objetivo: mezcla de dos normales"""
    return 0.3 * stats.norm.pdf(x, -2, 0.5) + 0.7 * stats.norm.pdf(x, 2, 0.8)

def metropolis_hastings(target, n_iter, proposal_std=1.0, inicial=0):
    """Algoritmo de Metropolis-Hastings"""
    muestras = np.zeros(n_iter)
    muestras[0] = inicial
    aceptados = 0
    
    for t in range(1, n_iter):
        # Propuesta: random walk
        propuesta = muestras[t-1] + np.random.normal(0, proposal_std)
        
        # Ratio de aceptación
        ratio = target(propuesta) / target(muestras[t-1])
        
        # Aceptar o rechazar
        if np.random.random() < ratio:
            muestras[t] = propuesta
            aceptados += 1
        else:
            muestras[t] = muestras[t-1]
    
    tasa_aceptacion = aceptados / n_iter
    return muestras, tasa_aceptacion

# Ejecusión MH
n_iter = 10000
muestras_mh, tasa_acept = metropolis_hastings(target_distribution, n_iter, proposal_std=1.5)

print(f"Iteraciones: {n_iter}")
print(f"Tasa de aceptación: {tasa_acept:.3f}")
print(f"Media estimada: {np.mean(muestras_mh[1000:]):.3f}")
print(f"Desviación estándar: {np.std(muestras_mh[1000:]):.3f}")

# 20.
print("\nGIBBS SAMPLING")
print("Estimando μ y σ² de una distribución normal")

# Datos sintéticos
datos_gibbs = np.random.normal(5, 2, 100)

# Priors
mu0, tau0 = 0, 10      # prior para μ: Normal(mu0, tau0²)
a0, b0 = 2, 2          # prior para σ²: Inv-Gamma(a0, b0)

def gibbs_sampler(datos, n_iter):
    """Gibbs sampling para Normal(μ, σ²)"""
    n = len(datos)
    x_barra = np.mean(datos)
    
    # Almacenar muestras
    mu_muestras = np.zeros(n_iter)
    sigma2_muestras = np.zeros(n_iter)
    
    # Valores iniciales
    sigma2_muestras[0] = np.var(datos)
    
    for t in range(1, n_iter):
        # 1. Samplear μ | σ², datos
        var_mu = 1 / (1/tau0**2 + n/sigma2_muestras[t-1])
        media_mu = var_mu * (mu0/tau0**2 + n*x_barra/sigma2_muestras[t-1])
        mu_muestras[t] = np.random.normal(media_mu, np.sqrt(var_mu))
        
        # 2. Samplear σ² | μ, datos
        a_post = a0 + n/2
        b_post = b0 + 0.5 * np.sum((datos - mu_muestras[t])**2)
        sigma2_muestras[t] = 1 / np.random.gamma(a_post, 1/b_post)
    
    return mu_muestras, sigma2_muestras

# Ejecutar Gibbs
n_iter_gibbs = 5000
mu_gibbs, sigma2_gibbs = gibbs_sampler(datos_gibbs, n_iter_gibbs)

# Quemado (burn-in)
burn_in = 1000
mu_final = mu_gibbs[burn_in:]
sigma2_final = sigma2_gibbs[burn_in:]

print(f"Media real: μ=5, σ²=4")
print(f"\nPosterior después de {burn_in} burn-in:")
print(f"μ: media={np.mean(mu_final):.3f}, IC 95%=[{np.percentile(mu_final,2.5):.3f}, {np.percentile(mu_final,97.5):.3f}]")
print(f"σ²: media={np.mean(sigma2_final):.3f}, IC 95%=[{np.percentile(sigma2_final,2.5):.3f}, {np.percentile(sigma2_final,97.5):.3f}]")

""" GRÁFICCOS """
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 18.
axes[0,0].scatter(X, y, alpha=0.5, s=30, label='Datos')
x_plot = np.linspace(min(X), max(X), 100)
for i in range(100):
    beta0_sample = beta_muestras[i*50, 0]
    beta1_sample = beta_muestras[i*50, 1]
    axes[0,0].plot(x_plot, beta0_sample + beta1_sample*x_plot, 
                   'b-', alpha=0.05)
axes[0,0].plot(x_plot, beta0_real + beta1_real*x_plot, 'r-', linewidth=2, label='Real')
axes[0,0].set_title('1. Regresión Bayesiana')
axes[0,0].set_xlabel('Años de educación')
axes[0,0].set_ylabel('Salario')
axes[0,0].legend()

# 19.
axes[0,1].hist(beta_muestras[:,1], bins=40, alpha=0.7, color='green', density=True)
axes[0,1].axvline(beta1_real, color='red', linestyle='--', linewidth=2, label='Real')
axes[0,1].set_title('Posterior de β₁')
axes[0,1].set_xlabel('β₁')
axes[0,1].legend()

# 20.
x_plot = np.linspace(-5, 5, 200)
axes[0,2].hist(muestras_mh[1000:], bins=50, alpha=0.7, density=True, color='purple', label='MH muestras')
axes[0,2].plot(x_plot, target_distribution(x_plot), 'r-', linewidth=2, label='Distribución real')
axes[0,2].set_title(f'2. Metropolis-Hastings\nAceptación: {tasa_acept:.1%}')
axes[0,2].set_xlabel('x')
axes[0,2].legend()

# 19.1 Traza de MH
axes[1,0].plot(muestras_mh[:500], 'b-', alpha=0.7)
axes[1,0].set_title('Traza MH (primeras 500 iter)')
axes[1,0].set_xlabel('Iteración')
axes[1,0].set_ylabel('x')

# 20.1 Gibbs Sampling - μ y σ²
axes[1,1].scatter(mu_gibbs[burn_in::5], sigma2_gibbs[burn_in::5], 
                  alpha=0.3, s=10, color='orange')
axes[1,1].axvline(5, color='red', linestyle='--', label='μ real=5')
axes[1,1].axhline(4, color='green', linestyle='--', label='σ² real=4')
axes[1,1].set_title('3. Gibbs Sampling\nMuestras (μ, σ²)')
axes[1,1].set_xlabel('μ')
axes[1,1].set_ylabel('σ²')
axes[1,1].legend()

# 20.2 Trazas de Gibbs
axes[1,2].plot(mu_gibbs[:500], 'b-', alpha=0.6, label='μ')
axes[1,2].plot(sigma2_gibbs[:500], 'r-', alpha=0.6, label='σ²')
axes[1,2].set_title('Trazas Gibbs (primeras 500)')
axes[1,2].set_xlabel('Iteración')
axes[1,2].legend()

plt.tight_layout()
plt.show()