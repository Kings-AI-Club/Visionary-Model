# Project Lighthouse - Visionary Model

A machine learning system for predicting homelessness risk using synthetic data generation and neural network-based classification. This project combines a sophisticated synthetic data generator with a deep learning model to identify individuals at risk of homelessness based on demographic and risk factor data.

## Overview

The Visionary Model consists of two main components:

1. **Synthetic Data Generator**: Creates realistic synthetic datasets based on Australian demographic distributions and domain-informed risk factors
2. **Predictive Neural Network**: A feedforward neural network built with TensorFlow Keras that classifies individuals by homelessness risk

## Features

- Synthetic data generation using weighted probability modeling
- Logistic regression-based probability calculation with domain-informed weights
- Deep learning classification using feedforward neural networks
- Support for multiple risk factors (substance abuse, mental health, domestic violence, indigenous status)
- Demographic modeling based on Australian population statistics
- Comprehensive model evaluation with accuracy, AUC, and loss metrics

## Project Structure

```
Visionary-Model/
├── synthetic-data/          # Data generation components
│   ├── generator.py         # Command-line data generator
│   ├── generator.ipynb      # Interactive notebook for data generation
│   ├── maths.py            # Mathematical utilities (logistic, calibration)
│   └── test.py             # Testing utilities
├── model/                   # Machine learning model
│   ├── model.ipynb         # Model training notebook (v0.1)
│   ├── ipf_model.py        # Model training script
│   ├── data/               # Generated datasets
│   └── figures/            # Training visualizations
├── homelessness_risk_model.h5  # Trained model weights
├── training_history.png    # Training metrics visualization
└── LICENSE                 # Apache 2.0 License
```

## Synthetic Data Generator

### How It Works

The data generator creates synthetic populations using a three-stage process:

#### 1. Covariate Sampling
Samples demographic and risk factors from real-world Australian distributions:

- **Demographics**: Gender, Age (7 bands from 0-17 to 65+), Location (8 Australian states/territories)
- **Risk Factors**: Substance abuse, Mental health issues, Indigenous status, Domestic violence exposure

Data distributions are based on Australian Bureau of Statistics (ABS) 2021 Census data.

#### 2. Logistic Probability Model
Computes homelessness probability using a weighted logistic model:

```
P(Homeless) = σ(Σ(w_attribute × Attribute))
```

Where:
- σ(x) = 1/(1 + e^(-x)) is the sigmoid/logistic function
- Weights are domain-informed and tunable
- Higher weights for high-impact factors (DV: 1.3, Drug: 1.1, Mental: 0.7)

**Key Risk Factor Weights:**
- Domestic Violence (DV): 1.3 (strongest predictor)
- Drug abuse: 1.1
- Mental health: 0.7
- Indigenous status: 0.25
- Gender: 0.08

**Age Band Effects:**
- Peak risk: 35-44 years (+0.35)
- Low risk: 0-17 years (-0.60), 65+ (-0.25)

**Location Effects:**
- VIC: +0.15, NT: +0.10 (higher risk)
- ACT: -0.05, TAS: -0.02 (lower risk)

#### 3. Intercept Calibration
Automatically calibrates the intercept term to match target homelessness prevalence rates using binary search.

### Usage

**Command Line:**
```bash
python synthetic-data/generator.py
```

**With Custom Parameters:**
```bash
SEED=42 TOTAL=100000 TARGET_RATE=0.5 python synthetic-data/generator.py
```

**Interactive Notebook:**
```bash
jupyter notebook synthetic-data/generator.ipynb
```

### Configuration

Environment variables for tuning:
- `SEED`: Random seed for reproducibility (default: 42)
- `TOTAL`: Number of samples to generate (default: 95,359)
- `TARGET_RATE`: Target homelessness prevalence (default: 0.533)
- `W_GENDER`, `W_DRUG`, `W_MENTAL`, `W_INDIG`, `W_DV`: Risk factor weights

## Predictive Model (v0.1)

### Architecture

Feedforward Neural Network with the following structure:

```
Input Layer (14 features)
    ↓
Dense Layer (128 neurons, sigmoid activation)
    ↓
Dropout (15%)
    ↓
Dense Layer (128 neurons, sigmoid activation)
    ↓
Dropout (15%)
    ↓
Dense Layer (64 neurons, sigmoid activation)
    ↓
Dropout (15%)
    ↓
Dense Layer (32 neurons, sigmoid activation)
    ↓
Output Layer (1 neuron, sigmoid activation)
```

### Training Configuration

- **Loss Function**: Binary Crossentropy
- **Alternative Loss**: Tversky Loss (α=0.4, β=0.6) for handling class imbalance
- **Optimizer**: Adam
- **Metrics**: Accuracy, AUC (Area Under Curve)
- **Training Split**: 80% train, 20% test
- **Batch Size**: 2048
- **Epochs**: 100

### Data Features

The model uses 14 input features:

**Binary Features (6):**
- Gender (0=Female, 1=Male)
- Drug (substance abuse indicator)
- Mental (mental health issue indicator)
- Indigenous (indigenous status)
- DV (domestic violence exposure)

**Continuous Features (1):**
- Age (normalized using Z-score standardization)

**One-Hot Encoded Location (8):**
- ACT, NSW, NT, QLD, SA, TAS, VIC, WA

**Target Variable:**
- Homeless (binary: 0=not homeless, 1=homeless)

### Data Preprocessing

1. **Boolean to Binary Conversion**: All boolean features converted to 0/1
2. **Age Normalization**: Z-score normalization applied to age
   - Formula: Z = (X - μ) / σ
   - Results in mean=0, std=1
3. **Stratified Split**: Train/test split maintains class balance

### Usage

**Interactive Training:**
```bash
jupyter notebook model/model.ipynb
```

**Script-Based Training:**
```bash
python model/ipf_model.py
```

### Model Outputs

- `homelessness_risk_model.h5`: Trained model weights
- `training_history.png`: Visualization of accuracy, loss, and AUC over epochs
- Classification reports and confusion matrices

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib jupyter
```

**Core Libraries:**
- TensorFlow/Keras: Deep learning framework
- NumPy: Numerical computing
- Pandas: Data manipulation
- scikit-learn: Data preprocessing and metrics
- Matplotlib: Visualization
- Jupyter: Interactive notebooks

## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Visionary-Model.git
cd Visionary-Model
```

2. **Generate synthetic data:**
```bash
python synthetic-data/generator.py
```

3. **Train the model:**
```bash
jupyter notebook model/model.ipynb
```

4. **View results:**
Check `training_history.png` for training metrics and model performance.

## Model Performance

The model is evaluated on:
- **Accuracy**: Overall classification accuracy
- **AUC (Area Under Curve)**: Measures model's ability to distinguish between classes
- **Loss**: Binary crossentropy loss tracking convergence

Training visualizations show:
- Training vs validation accuracy over epochs
- Training vs validation loss over epochs
- Training vs validation AUC over epochs

## Technical Details

### Tversky Loss

The model optionally uses Tversky loss to handle class imbalance:

```
L_Tversky(y, ŷ) = 1 - (Σ y_i ŷ_i) / (Σ y_i ŷ_i + α Σ y_i(1-ŷ_i) + β Σ (1-y_i)ŷ_i)
```

Where:
- α and β control penalties for false negatives and false positives
- Common in medical ML where false negatives are costly
- Parameters: α=0.4 (FN penalty), β=0.6 (FP penalty)

### Activation Functions

**ReLU (Hidden Layers):**
```
ReLU(x) = max(0, x)
```

**Sigmoid (Output Layer):**
```
σ(x) = 1 / (1 + e^(-x))
```

Sigmoid output provides probability scores between 0 and 1.

## Data Ethics and Limitations

This model uses **synthetic data** generated from statistical distributions and domain knowledge. Key considerations:

- Data is artificially generated, not real individual records
- Risk factor weights are informed estimates based on research literature
- Australian demographic distributions used where available
- Model should not be used for real-world decision-making without validation
- Designed for research, education, and proof-of-concept purposes

## Contributing

Contributions are welcome! Areas for improvement:

- Enhanced risk factor modeling
- Additional demographic variables
- Model architecture experiments
- Real-world data validation studies
- Performance optimization

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Project Context

Project Lighthouse - Visionary Model is a proof-of-concept system demonstrating the application of machine learning to social service challenges. The goal is to explore how predictive models could potentially assist in early intervention and resource allocation for homelessness prevention programs.

## Acknowledgments

- Australian Bureau of Statistics for demographic data sources
- Research literature on homelessness risk factors
- TensorFlow/Keras community for machine learning frameworks
