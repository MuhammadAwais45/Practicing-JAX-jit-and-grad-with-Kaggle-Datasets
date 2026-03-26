# Practicing-JAX-jit-and-grad-with-Kaggle-Datasets
This repository demonstrates how to use JAX jit and grad with Kaggle datasets for regression and classification. It highlights faster computation, automatic gradient calculation, and efficient model training using real-world datasets like House Prices and Titanic Survival.
# JAX vs NumPy: Kaggle Case Studies

This repository contains experiments comparing **JAX** and **NumPy** for machine learning tasks on Kaggle datasets. The focus is on using **Just-In-Time compilation (`jit`)** and **Automatic Differentiation (`grad`)** to accelerate computations and simplify derivative calculations.  

Two case studies are included:  

- **House Prices Prediction** – Regression  
- **Titanic Survival Prediction** – Classification  

---

## Kaggle Achievements

| Competition | Goal | Features Used | Rank / Teams |
|------------|------|---------------|--------------|
| [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Predict sales prices | Small set of numerical features | 3615 / 4046 |
| [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Predict survival | Age, class, sex, family size, fare | 11725 / 12243 |

> Only a small subset of features from the datasets was used, following the Lab Working Manual: Practicing JAX `jit` and `grad` workflow.

---

## Objectives

- Apply **JIT compilation (`jit`)** and **automatic differentiation (`grad`)** in real-world machine learning scenarios.  
- Understand how JAX transformations **accelerate computation** and simplify optimization.  
- Practice regression and classification tasks with Kaggle datasets.  
- Reflect on the reasoning behind optimization choices.

---

## Prerequisites

- Python 3.8+  
- Install JAX:  
```bash
pip install jax jaxlib
```
