# AutoSpectraEngine

<p align="center">
  <img src="ASNlogo.png" alt="Drag Racing" width="200"/>
</p>

AutoSpectraEngine is a Python-based platform developed by the Laboratory of Food Innovation (LINA) at the University of Campinas and the Machine Learning Lab at the University of Trieste. It is designed for the analysis of Near-Infrared (NIR) and Raman spectral data. The platform addresses critical challenges in spectral analysis, including preprocessing, spectral window optimization, and model hyperparameter tuning, all while automating the entire pipeline.

## Features

### Core Preprocessing Functions
1. **Mean Centering (MC)** - Center spectral data by subtracting the mean of each variable.
2. **Autoscaling** - Normalize data using z-score normalization.
3. **Smoothing (SMO)** - Apply Savitzky-Golay filters for noise reduction.
4. **First Derivative (D1)** - Compute the first derivative of spectral data for baseline correction.
5. **Second Derivative (D2)** - Compute the second derivative for resolving overlapping peaks.
6. **Multiplicative Scatter Correction (MSC)** - Reduce scattering effects in spectral data.
7. **Standard Normal Variate (SNV)** - Normalize data to eliminate scatter effects.

### Outlier Detection
- **Isolation Forest**: Detect and remove outliers from spectral datasets.

### Automated Pipeline Testing
- Perform experiments with over 70 unique preprocessing pipeline combinations.
- Flexibility to test pipelines tailored for NIR or Raman data.

### Modeling and Analysis
- **PCA Visualization**: Visualize spectral data distribution in reduced dimensionality.
- **PLSR (Partial Least Squares Regression)**: Evaluate model performance using metrics such as RMSE, RPD, and \( R^2 \).
- **PLS-DA (Partial Least Squares Discriminant Analysis)**: Assess classification accuracy and optimal latent variables.
- **Random Forest (RF)**: Evaluate classification performance using sensitivity and specificity.
- **One-Class PLS**: For one-class classification problems.
- **DDSIMCA**: A robust classification method for discriminant analysis.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sbarbonjr/AutoSpectraEngine.git
cd AutoSpectraEngine
pip install -r requirements.txt
```

## Installation
### Running Experiments

Use the run_all_experiments function to automate pipeline testing and model evaluation:

```bash
from auto_spectra_engine import run_all_experiments

run_all_experiments(
    file="path/to/data.csv",
    modelo="PLSDA", 
    coluna_predicao="Class",
    test_contamination=True,
    pipeline_family='all'
)
```