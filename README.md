# Simple Linear Regression from Scratch

A theory-driven implementation of **Simple Linear Regression using Ordinary Least Squares (OLS)** from first principles, followed by validation against `scikit-learn`, controlled loss-function experiments, and regression residual analysis.

The goal of this project is not simply to train a regression model, but to understand **what happens underneath a machine-learning library**: how the coefficients are derived, why squared error is used by OLS, how predictions are evaluated, and how residuals can be used to diagnose a regression model.

---

## Project Overview

Linear Regression is one of the fundamental algorithms in machine learning and statistics.

Although libraries such as `scikit-learn` allow us to fit a linear regression model with a single function call, this project focuses on understanding and implementing the underlying mathematics and methodology.

The project uses the **Advertising dataset**, which contains advertising expenditure across three media channels and the resulting sales.

For the Simple Linear Regression project, the relationship investigated is:

$$
TV \rightarrow Sales
$$

The model is:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

where:

- $x$ = TV advertising expenditure
- $y$ = Sales
- $\beta_0$ = intercept
- $\beta_1$ = slope
- $\hat{y}$ = predicted sales

---

# Objectives

The main objectives of this project are:

1. Understand the mathematical foundation of Simple Linear Regression.
2. Derive the Ordinary Least Squares solution.
3. Implement Simple Linear Regression from scratch using NumPy.
4. Implement regression evaluation metrics manually.
5. Validate the custom implementation against `scikit-learn`.
6. Understand why squared error is used in classical OLS.
7. Experiment with different loss functions.
8. Investigate the effect of outliers and noise on OLS.
9. Analyze regression residuals.
10. Build a reproducible, modular, and testable machine-learning project.

---

# Dataset

The project uses the **Advertising dataset**.

The dataset contains approximately 200 observations with the following variables:

| Feature | Description |
|---|---|
| `TV` | Advertising expenditure through TV |
| `Radio` | Advertising expenditure through radio |
| `Newspaper` | Advertising expenditure through newspapers |
| `Sales` | Product sales |

For this project, only:

```text
TV → Sales
```

is used for Simple Linear Regression.

The dataset is used for demonstrating regression methodology and implementation. The project does **not** make a causal claim that increasing TV advertising necessarily causes an increase in sales.

---

# Project Structure

```text
simple-linear-regression-from-scratch/
│
├── data/
│   ├── raw/
│   │   └── advertising.csv
│   └── processed/
│       └── advertising_clean.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_linear_regression_math.ipynb
│   ├── 03_slr_from_scratch.ipynb
│   ├── 04_sklearn_validation.ipynb
│   ├── 05_loss_function_experiments.ipynb
│   └── 06_residual_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── linear_regression.py
│   ├── metrics.py
│   └── visualization.py
│
├── tests/
│   ├── __init__.py
│   ├── test_linear_regression.py
│   └── test_metrics.py
│
├── reports/
│   ├── figures/
│   └── model_report.md
│
├── models/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Methodology

The project follows the workflow:

```text
Data
  ↓
Exploration
  ↓
Mathematical Understanding
  ↓
OLS Implementation
  ↓
Model Evaluation
  ↓
Library Validation
  ↓
Loss Function Experiments
  ↓
Residual Analysis
```

Each stage has a specific purpose rather than simply adding another model.

---

# Notebook 01 — Data Exploration

`01_data_exploration.ipynb`

The first notebook investigates the structure and quality of the dataset.

### Analysis performed

- Load the dataset
- Inspect dataset dimensions
- Inspect data types
- Check missing values
- Check duplicate observations
- Generate descriptive statistics
- Examine distributions
- Analyze the relationship between `TV` and `Sales`
- Calculate correlation
- Visualize the predictor-target relationship

### Main question

> Is a linear relationship between TV advertising expenditure and sales reasonable as a starting modeling assumption?

The exploratory analysis provides the motivation for applying Simple Linear Regression.

---

# Notebook 02 — Linear Regression Mathematics

`02_linear_regression_math.ipynb`

This notebook focuses on the mathematical foundation of Ordinary Least Squares.

The main model is:

$$
y_i = \beta_0 + \beta_1 x_i + \epsilon_i
$$

The prediction is:

$$
\hat{y}_i = \beta_0 + \beta_1 x_i
$$

and the residual is:

$$
e_i = y_i - \hat{y}_i
$$

### Topics covered

- Linear regression formulation
- Predictions
- Residuals
- Signed error
- Error cancellation
- MAE
- MSE
- SSE
- Why OLS minimizes squared error
- Derivation of the slope
- Derivation of the intercept
- Relationship between OLS and MSE
- Residual properties
- Connection between OLS and Gaussian errors
- Classical linear regression assumptions
- Limitations of linear regression

The closed-form OLS estimates are:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}
$$

and:

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

---

# Notebook 03 — Simple Linear Regression from Scratch

`03_slr_from_scratch.ipynb`

This notebook implements the regression algorithm without using `sklearn` for model fitting.

The implementation is contained in:

```text
src/linear_regression.py
```

The model uses the closed-form OLS solution.

### Core implementation

```python
class SimpleLinearRegression:
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

The implementation calculates:

1. Mean of `X`
2. Mean of `y`
3. OLS numerator
4. OLS denominator
5. Slope
6. Intercept
7. Predictions

### Evaluation metrics

The following metrics are also implemented from scratch:

- MAE
- MSE
- RMSE
- $R^2$

These implementations are located in:

```text
src/metrics.py
```

### Additional checks

The notebook also verifies important OLS properties, including:

$$
\sum_{i=1}^{n} e_i \approx 0
$$

and:

$$
\sum_{i=1}^{n} x_i e_i \approx 0
$$

when an intercept is included.

---

# Notebook 04 — Validation Against Scikit-Learn

`04_sklearn_validation.ipynb`

The custom implementation is validated against:

```python
sklearn.linear_model.LinearRegression
```

The same:

- dataset
- train/test split
- feature
- target

are used for both implementations.

### Validation includes

- Intercept comparison
- Slope comparison
- Prediction comparison
- MAE comparison
- MSE comparison
- RMSE comparison
- $R^2$ comparison
- Regression line comparison
- Residual comparison

Numerical comparisons use tolerance-based checks such as:

```python
np.isclose()
```

and:

```python
np.allclose()
```

rather than exact floating-point equality.

### Purpose

This provides an independent validation that the from-scratch implementation reproduces the expected OLS solution.

---

# Notebook 05 — Loss Function Experiments

`05_loss_function_experiments.ipynb`

This notebook investigates the behavior of different error functions experimentally.

Rather than only stating that OLS minimizes squared error, the notebook demonstrates the consequences of different loss functions.

### Experiments include

#### 1. Signed error cancellation

Demonstrates why:

$$
\sum_{i=1}^{n} (y_i - \hat{y}_i)
$$

is not an appropriate general-purpose measure of model error.

Positive and negative errors can cancel each other.

---

#### 2. MAE vs MSE

Comparison of:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

and:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

The experiment demonstrates that MSE assigns substantially greater penalty to large errors.

---

#### 3. Loss-function visualization

The behavior of absolute loss and squared loss is visualized as the magnitude of the prediction error changes.

This provides an intuitive explanation for why the two objectives respond differently to large residuals.

---

#### 4. Outlier sensitivity

Synthetic data is generated from a linear relationship.

An extreme observation is then introduced and the OLS model is refitted.

The experiment investigates how a single influential observation can change:

- estimated slope
- estimated intercept
- fitted regression line

This demonstrates an important property of squared-error-based regression:

> Large residuals can have a disproportionately large effect on the objective.

This is a sensitivity property, not an indication that MSE is inherently incorrect.

---

#### 5. Effect of noise

The experiment varies the amount of random noise in a synthetic linear dataset.

This demonstrates how increasing noise can affect the observed relationship and regression fit.

---

# Notebook 06 — Residual Analysis

`06_residual_analysis.ipynb`

Residual analysis investigates the errors produced by the fitted regression model.

For each observation:

$$
e_i = y_i - \hat{y}_i
$$

Residuals are important because aggregate metrics such as MAE, MSE, RMSE, and $R^2$ do not reveal the complete structure of model errors.

### Analysis currently completed

#### Residual calculation

The notebook calculates:

```python
residuals = y_actual - y_predicted
```

and examines their basic statistical properties.

---

### Residual summary

The residuals are summarized using statistics such as:

- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

For OLS with an intercept:

$$
\sum_{i=1}^{n} e_i \approx 0
$$

up to numerical precision.

---

### Residuals vs predicted values

The residuals are plotted against the model's predicted values.

The purpose is to look for systematic structures in the prediction errors.

Ideally, residuals should be scattered around zero without an obvious systematic pattern.

Patterns such as curves or changing spread can indicate that the linear model may not adequately describe the relationship.

---

### Residuals vs feature

Residuals are also plotted against the original predictor:

```text
TV
```

This helps investigate whether the magnitude or direction of prediction errors changes systematically with the predictor.

---

### Distribution of regression residuals

The residual distribution is examined using a histogram and density estimate.

This helps investigate:

- Centering around zero
- Spread of residuals
- Symmetry
- Potential extreme errors
- Approximate distributional shape

A visual distribution that appears approximately bell-shaped can be consistent with the normal-error assumption used in classical regression inference.

However:

> Visual inspection alone does not prove that residuals are normally distributed.

---

# Current Project Status

The project currently covers:

```text
Data Exploration                         ✓
        ↓
Regression Mathematics                   ✓ / notebook prepared
        ↓
SLR from Scratch                         ✓
        ↓
Manual Evaluation Metrics                ✓
        ↓
Scikit-Learn Validation                  ✓
        ↓
Loss Function Experiments                ✓
        ↓
Residual Analysis                        ✓
```

The residual-analysis notebook currently ends after examining the **distribution of regression residuals**.

---

# Implementation Details

## Simple Linear Regression

The model is implemented using NumPy rather than calling a pre-built regression estimator.

The central calculation is:

```python
x_mean = np.mean(X)
y_mean = np.mean(y)

numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)

slope = numerator / denominator
intercept = y_mean - slope * x_mean
```

Predictions are then generated using:

```python
y_pred = intercept + slope * X
```

---

# Evaluation Metrics

## Mean Absolute Error

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Measures the average absolute prediction error.

---

## Mean Squared Error

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Penalizes larger errors more heavily because the residual is squared.

---

## Root Mean Squared Error

$$
RMSE = \sqrt{MSE}
$$

RMSE has the same units as the target variable.

---

## Coefficient of Determination

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

$R^2$ compares the model's residual variation against the total variation in the target.

It should not be interpreted as proof of causality or as a complete measure of model quality.

---

# Engineering Practices

This project is structured to separate experimentation from reusable implementation.

### Reusable source code

```text
src/
```

contains the actual model, metrics, preprocessing, and visualization utilities.

### Experiments

```text
notebooks/
```

contains exploratory analysis, mathematical demonstrations, experiments, and validation.

### Testing

```text
tests/
```

contains automated tests for the implementation.

### Reproducibility

The project uses a fixed train/test split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This ensures that different implementations can be compared on the same observations.

---

# Why Implement Linear Regression from Scratch?

Using:

```python
LinearRegression()
```

is straightforward.

However, using a library without understanding the underlying optimization process can hide important concepts.

This project focuses on understanding:

- What OLS actually minimizes
- Why squared errors are used
- How coefficients are calculated
- How residuals behave
- How model assumptions affect interpretation
- How evaluation metrics are constructed
- How a library implementation can be independently validated

The purpose is therefore **understanding and verification**, not replacing production-grade libraries.

---

# Limitations

This project intentionally focuses on Simple Linear Regression.

Important limitations include:

- Only one predictor is used.
- The relationship is assumed to be adequately represented by a linear model.
- OLS is sensitive to influential observations.
- Regression performance does not establish causality.
- Extrapolation outside the observed range can be unreliable.
- $R^2$ alone is insufficient for evaluating a regression model.
- Statistical assumptions require diagnostic investigation rather than being automatically guaranteed.

---

# Future Extensions

The project can later be extended with additional statistical diagnostics and models.

Potential extensions include:

### Statistical diagnostics

- Q-Q plots
- Formal normality assessment
- Heteroscedasticity tests
- Leverage analysis
- Cook's distance
- Influence analysis
- Coefficient standard errors
- Confidence intervals
- Hypothesis testing

### Multiple Linear Regression

Extend the project from:

$$
y = \beta_0 + \beta_1 x
$$

to:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

using the matrix OLS solution:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

This will form the basis of a separate Multiple Linear Regression project.

### Regularization

Future projects can investigate:

- Ridge Regression
- Lasso Regression
- Elastic Net

and compare their behavior with ordinary least squares.

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Statsmodels
- Jupyter Notebook
- Pytest

---

# Learning Outcomes

By completing this project, the following concepts are demonstrated:

- Simple Linear Regression
- Ordinary Least Squares
- Closed-form parameter estimation
- Residuals
- SSE, MSE, MAE and RMSE
- $R^2$
- Error cancellation
- Loss-function behavior
- Outlier sensitivity
- Noise and model fitting
- Regression diagnostics
- Train/test evaluation
- Numerical validation
- Modular Python code
- Unit testing
- Reproducible experimentation

---

# Disclaimer

The relationship between advertising expenditure and sales in this project is analyzed as a statistical association for modeling purposes.

The regression model should not be interpreted as establishing a causal effect of TV advertising on sales.