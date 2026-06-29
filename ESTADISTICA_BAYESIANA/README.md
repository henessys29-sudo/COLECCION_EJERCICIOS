# Análisis Estadístico e Inferencia Bayesiana

Este repositorio contiene una colección de scripts en Python que implementan diversos métodos de inferencia estadística, desde conceptos básicos bayesianos hasta técnicas de simulación avanzada y modelos MCMC.

## Estructura del Proyecto

El proyecto está organizado en 5 módulos principales, cada uno diseñado para abordar una parte específica del análisis estadístico:

- `01_FUNDAMENTOS.py`: Análisis de distribuciones Beta, comportamiento de *priors* y su impacto en la posterior.
- `02_ENCUESTAS_Y_ELECCIONES.py`: Modelado de encuestas, combinación de datos y estimación de probabilidad de victoria.
- `03__A-B_TESTING.py`: Aplicación de pruebas A/B bayesianas, cálculo de diferencias entre grupos y análisis de sensibilidad.
- `03__A-B_TESTING_13.py`: Resolución del punto 13.
- `04_MONTE_CARLO.py`: Implementación de simulaciones de Monte Carlo para cálculo de $\pi$, integración numérica y estimación de VaR (Value at Risk) financiero.
- `05_MODELOS_AVANZADOS.py`: Implementación de regresión lineal bayesiana, algoritmo de Metropolis-Hastings y Gibbs Sampling.

## Requisitos

Para ejecutar los scripts, asegúrate de tener instalado Python 3.11+ y las siguientes librerías:

```bash
pip install numpy matplotlib scipy pandas