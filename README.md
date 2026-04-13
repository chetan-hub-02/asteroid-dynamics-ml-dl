# asteroid-dynamics-ml-dl

Machine learning and deep learning framework for classifying the dynamical fate of near-Earth asteroids (NEAs) from 1 Myr integrations.

---

## Overview

This repository contains code and trained models for classifying NEAs into two outcomes:

- **Ejected**
- **Non-ejected / bound**

The project includes both:

- **traditional machine learning (ML)** models trained on tabulated orbital features
- **deep learning (DL)** models trained on recurrence-plot images generated from orbital time series

The dataset and trained CNN weights are also available in Zenodo for reproducibility.

---

## Dataset

The dataset is available on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.19553922  
**Zenodo title:** *Dataset for Machine Learning and Deep Learning Classification of Near-Earth Asteroid Ejection*

### Dataset contents

The Zenodo record contains two complementary data formats:

#### 1. ML_Data
Tabulated features extracted at the initial epoch for traditional ML models.

**Features include:**
- Semi-major axis
- Eccentricity
- Inclination
- Argument of perihelion
- Longitude of ascending node
- Mean anomaly
- Perihelion distance
- Aphelion distance
- Focal distance
- Asteroid type

**Classes:**
- `class_1`: ejected asteroids
- `class_2`: non-ejected asteroids

The ML dataset is imbalanced, so the classes should be balanced before training, with approximately **15,000 objects per class**.

#### 2. DL_Data
Recurrence-plot images generated from **0.2 Myr** orbital time-series data for CNN-based classification.

**Directory structure:**
- `Train/class_1/`
- `Train/class_2/`
- `Test/class_1/`
- `Test/class_2/`

Here:
- `class_1` = ejected asteroids
- `class_2` = non-ejected asteroids

For CNN training, the training set should be further split into **train / validation / test** subsets in an approximate **80:10:10** ratio.

---

## Trained Models

- The trained CNN weights (`best_model.h5`) are provided in Zenodo because of GitHub size constraints.
- The weights for the other ML models are included in this GitHub repository.

---

## Repository Contents

- `ml_models/` — scripts for classical ML models
- `train_CNN.py` — CNN training script
- `weights/` — saved weights for ML models
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `LICENSE` — license file

### ML training scripts
- `train_adaboost.py`
- `train_decision_tree.py`
- `train_extra_tree.py`
- `train_gboost.py`
- `train_k_nearest_neighbor.py`
- `train_mlp.py`
- `train_random_forest.py`

---

## Notes

- The data are derived from **MERCURY N-body integrations** of NEA trajectories over 1 Myr.
- Use the test set only for final evaluation.
- Validation data should be used for early stopping and model selection.

---

## Installation

```bash
pip install -r requirements.txt
