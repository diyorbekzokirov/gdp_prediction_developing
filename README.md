# World Bank Development Prediction Project

This project uses World Bank economic and social indicators to predict the time it takes for developing countries to reach developed status. The analysis employs machine learning models to understand key factors that influence economic development trajectories.

## Project Overview

The project analyzes World Bank data across multiple countries and regions to:
- Identify key indicators that predict development timelines
- Build machine learning models to forecast years-to-development
- Understand feature importance and development patterns
- Compare different modeling approaches (Linear Regression, Random Forest, XGBoost)

## Project Structure

```
world_bank_data/
├── data/
│   ├── raw_data/           # Original World Bank data
│   ├── processed_data/     # Cleaned datasets
│   └── ML_data/            # Machine learning ready datasets
├── notebooks/
│   ├── data_prep.ipynb     # Data preparation and feature engineering
│   ├── analysis.ipynb      # Exploratory data analysis
│   ├── model_prediction.ipynb  # Model training and evaluation
│   └── models/             # Saved trained models
└── requirements.txt        # Python dependencies
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/diyorbekzokirov/gdp_prediction_developing.git
cd world_bank_data
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dependencies

- **numpy** (1.26.4) - Numerical computing
- **pandas** (2.1.4) - Data manipulation and analysis
- **scikit-learn** (1.4.2) - Machine learning models and preprocessing
- **xgboost** (2.0.3) - Gradient boosting framework
- **shap** (0.45.1) - Model interpretability and feature importance
- **matplotlib** (3.8.4) - Data visualization
- **seaborn** (0.13.2) - Statistical data visualization
- **joblib** (1.4.2) - Model serialization

## Notebooks

### 1. `data_prep.ipynb` - Data Preparation
**Purpose**: Prepares World Bank data for machine learning analysis

**Key Steps**:
- Load raw World Bank data from multiple regions
- Explore data quality and missing values
- Engineer features relevant to development prediction
- Handle missing data through imputation strategies
- Create both cross-sectional and panel data formats
- Generate train/test splits for model validation
- Export ML-ready datasets

**Outputs**:
- `train_years_to_developed_final.csv` - Training data
- `train_panel_years_to_developed.csv` - Panel format
- `predict_countries_features_final.csv` - Prediction features
- `predict_panel_features.csv` - Panel predictions

### 2. `analysis.ipynb` - Exploratory Data Analysis
**Purpose**: Understand patterns and relationships in the World Bank data

**Key Steps**:
- Load processed datasets
- Examine distributions of key economic indicators
- Analyze correlations between features
- Visualize development trajectories
- Identify outliers and data quality issues
- Explore regional differences

### 3. `model_prediction.ipynb` - Machine Learning Models
**Purpose**: Build and evaluate predictive models for development timelines

**Key Steps**:
- Load ML-ready datasets
- Implement preprocessing pipelines (imputation, scaling, encoding)
- Train multiple models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
- Evaluate model performance using cross-validation
- Compare models using metrics (RMSE, MAE, R²)
- Analyze feature importance using SHAP values
- Generate predictions for developing countries
- Save trained models for future use

**Model Outputs**:
- Saved models in `notebooks/models/`
- Performance metrics and comparisons
- Feature importance visualizations
- Development timeline predictions

## Key Features

### Economic Indicators
The project analyzes various World Bank indicators including:
- GDP growth rates
- Population demographics
- Education metrics
- Healthcare indicators
- Infrastructure development
- Trade and investment flows
- Government effectiveness
- And many more...

### Machine Learning Approach
- **Cross-sectional models**: Predict development time based on country characteristics
- **Panel data models**: Leverage time-series patterns across countries
- **Feature engineering**: Create meaningful indicators from raw data
- **Model interpretability**: Use SHAP values to understand predictions

## Usage

### Running the Analysis

1. **Data Preparation**:
```bash
jupyter notebook notebooks/data_prep.ipynb
```
Run all cells to process raw data and create ML-ready datasets.

2. **Exploratory Analysis**:
```bash
jupyter notebook notebooks/analysis.ipynb
```
Explore visualizations and patterns in the data.

3. **Model Training**:
```bash
jupyter notebook notebooks/model_prediction.ipynb
```
Train models and generate predictions.

### Using Pre-trained Models

Load a saved model:
```python
import joblib
model = joblib.load('notebooks/models/random_forest.joblib')
```

Make predictions:
```python
predictions = model.predict(new_data)
```

## Results

The project provides:
- **Predictive models** that estimate years-to-development for countries
- **Feature importance rankings** showing which indicators matter most
- **Model comparisons** across different algorithms
- **Interpretable insights** through SHAP analysis
- **Country-specific predictions** for development timelines

## License

This project is part of academic research analyzing World Bank public data.

## Data Source

Data sourced from the [World Bank Open Data](https://data.worldbank.org/) platform, which provides free access to global development indicators.

---

**Note**: This project is for research and educational purposes. Development predictions should be interpreted with caution and in context with other economic analyses.
