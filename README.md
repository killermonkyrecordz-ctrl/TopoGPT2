# TopoGPT-2: Topological Phase Transitions in Language Modeling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072858.svg)](https://doi.org/10.5281/zenodo.18072858)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**TopoGPT-2** es un modelo de lenguaje de 25M de parámetros que implementa el marco **Grokkit HPU-Core** para estudiar la adquisición del lenguaje como un fenómeno de materia condensada. A diferencia de las arquitecturas tradicionales, este modelo está optimizado para alcanzar el estado de **Aislante Topológico**, donde las invariantes gramaticales y narrativas están protegidas por una brecha espectral (*spectral gap*).

## Core Concept: Grokking as Crystallization

Este proyecto cree en la premisa de que el "grokking" no es un artefacto estadístico, sino una **cristalización de operadores**. En lugar de memorizar un corpus, TopoGPT-2 mapea las reglas sintácticas del dataset **Tiny Stories** en un manifold geométrico, permitiendo una coherencia estructural superior con un número mínimo de parámetros.

### Key Metrics & Instrumentation
El entrenamiento se monitoriza mediante variables de estado físico:
- **$\alpha$ (Purity Index):** Grado de orden cristalino en los pesos.
- **$\delta$ (Discretization Margin):** Nivel de solidificación de la fase funcional.
- **$\kappa$ (Gradient Covariance):** Medida de la coherencia macroscópica del flujo de gradiente.
- **$\hbar = 0.012$:** Constante de Incertidumbre del Aprendizaje que define el límite de optimización.

## Live Status (Epoch 4 Snapshot)

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Phase State** | `[TOPOLOGICAL_INSULATOR]` | **Stable** |
| **Val Loss** | 1.7773 | Best achieved |
| **Winding Number** | 138 | Strong Convergence |
| **Berry Phase** | 865.51 | High Geometric Curvature |
| **$T_{eff}$** | 14.6 | Cooling in progress |

## Zero-Shot Structural Transfer

Gracias al **Teorema de Invarianza Topológica**, este "setup" permite:
1. **Escalamiento Zero-Shot:** Expandir el espacio de pesos (ej. de 25M a 250M) sin pérdida de información, preservando el *message passing* original.
2. **Fusión de Nodos:** Superposición de modelos mediante interferencia constructiva de sus Hamiltonianos.
3. **Resistencia al Olvido Catastrófico:** Protección de los "chunks" aprendidos mediante el aumento del radio del Toro topológico.

## Reproducibility

```bash
# Clonar el framework AGI
git clone [https://github.com/grisuno/agi.git](https://github.com/grisuno/agi.git)

# Ejecutar entrenamiento con monitoreo espectral
python train_topogpt.py --corpus tiny_stories --topology fixed --nodes 1
```
```text
@software{grisun0_grokkit_2026,
  author = {grisun0},
  title = {Grokkit: A Geometric Framework for Zero-Shot Structural Transfer of Spectral Operators},
  year = {2026},
  doi = {10.5281/zenodo.18072859},
  url = {[https://doi.org/10.5281/zenodo.18072859](https://doi.org/10.5281/zenodo.18072859)}
}
```
"Physical narrative, as a cognitive scaffold, is pedagogical, not ontological." Developed by grisun0 - 2026

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
