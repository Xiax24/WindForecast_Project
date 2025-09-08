# Wind Farm Power Forecasting: Near-Surface Wind Speed Advantage

This repository contains the complete code implementation for the research paper: "The Unexpected Superiority of Near-Surface Wind Speed in Wind Farm Power Forecasting".

## Overview

This study challenges the conventional assumption that hub-height wind speed is most critical for wind power forecasting, revealing unexpected advantages of 10 m wind speed for farm-level power prediction through a physically consistent framework integrating EOF analysis, SHAP interpretability, and EEMD decomposition.

## Key Findings

- 10 m wind speed outperforms hub-height measurements for wind farm power prediction
- Wind direction-based validation confirms wake effects suppress hub-height predictive value
- Multi-source fusion strategies achieve 7.13% RMSE reduction and 11.86% correlation improvement
- Dual-perspective framework considers both variable contribution and predictability

## Repository Structure

```
WindForecast_Project/
├── data_preprocessing/
│   ├── data_cleaning.py          # Data quality control and preprocessing
│   ├── wind_profile_analysis.py  # Multi-height wind speed analysis
│   └── meteorological_utils.py   # Utility functions for met data
├── analysis/
│   ├── eof_analysis.py           # Empirical Orthogonal Function analysis
│   ├── correlation_analysis.py   # Statistical correlation analysis
│   ├── shap_interpretability.py # SHAP-based variable importance
│   └── eemd_decomposition.py     # Ensemble Empirical Mode Decomposition
├── forecasting/
│   ├── lightgbm_models.py        # LightGBM implementation
│   ├── prediction_strategies.py  # Multi-source fusion strategies
│   └── performance_evaluation.py # Model evaluation metrics
├── validation/
│   ├── wind_direction_analysis.py # Wake effect validation
│   ├── seasonal_evaluation.py    # Seasonal performance assessment
│   └── shear_classification.py   # Wind shear analysis
├── visualization/
│   ├── plotting_utils.py         # General plotting functions
│   ├── shap_plots.py            # SHAP visualization
│   └── performance_plots.py     # Performance comparison plots
├── config/
│   ├── model_config.py          # Model hyperparameters
│   └── data_paths.py            # Data file paths
├── examples/
│   ├── full_analysis_pipeline.py # Complete analysis workflow
│   └── quick_start_demo.py       # Quick demonstration
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Xiax24/WindForecast_Project.git
cd WindForecast_Project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the demonstration script to see the main analysis pipeline:
```python
python examples/quick_start_demo.py
```

### Full Analysis Pipeline

To reproduce the complete analysis from the paper:
```python
python examples/full_analysis_pipeline.py
```

### Individual Components

#### 1. Data Preprocessing
```python
from data_preprocessing.data_cleaning import preprocess_wind_data
from data_preprocessing.wind_profile_analysis import analyze_vertical_profile

# Load and preprocess data
clean_data = preprocess_wind_data(raw_data_path)
profile_analysis = analyze_vertical_profile(clean_data)
```

#### 2. Variable Importance Analysis
```python
from analysis.shap_interpretability import calculate_shap_importance
from analysis.correlation_analysis import compute_correlations

# SHAP analysis
shap_values = calculate_shap_importance(features, target)

# Correlation analysis
correlations = compute_correlations(wind_speeds, power_output)
```

#### 3. Forecasting Strategies
```python
from forecasting.prediction_strategies import MultiSourceFusion
from forecasting.performance_evaluation import evaluate_predictions

# Multi-source fusion
fusion_model = MultiSourceFusion()
predictions = fusion_model.predict(test_data)

# Performance evaluation
metrics = evaluate_predictions(predictions, actual_values)
```

#### 4. Wake Effect Validation
```python
from validation.wind_direction_analysis import stratify_by_wind_direction
from validation.wind_direction_analysis import compare_variable_importance

# Wind direction stratification
east_wind_data, west_wind_data = stratify_by_wind_direction(data)

# Compare importance under different conditions
importance_comparison = compare_variable_importance(east_wind_data, west_wind_data)
```

## Data Requirements

The code expects meteorological data with the following structure:
- Multi-height wind speeds (10m, 30m, 50m, 70m)
- Wind directions at multiple heights
- Temperature measurements
- Wind farm power output
- Timestamps for all measurements

Sample data format:
```
timestamp, wind_speed_10m, wind_speed_30m, wind_speed_50m, wind_speed_70m, 
wind_dir_10m, wind_dir_30m, wind_dir_50m, wind_dir_70m, temperature_10m, power_output
```

## Key Dependencies

- Python 3.8+
- numpy
- pandas
- scikit-learn
- lightgbm
- shap
- matplotlib
- seaborn
- scipy
- PyEMD (for EEMD analysis)

## Results Reproduction

To reproduce the main results from the paper:

1. **Figure 1 (EOF Analysis)**: Run `analysis/eof_analysis.py`
2. **Figure 2 (Correlation Matrix)**: Run `analysis/correlation_analysis.py`
3. **Figure 3 (SHAP Analysis)**: Run `analysis/shap_interpretability.py`
4. **Performance Tables**: Run `forecasting/performance_evaluation.py`
5. **Wake Validation**: Run `validation/wind_direction_analysis.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{xia2024wind,
  title={The Unexpected Superiority of Near-Surface Wind Speed in Wind Farm Power Forecasting},
  author={Xia, Xin and Li, Peidu and Luo, Yong},
  journal={[Journal Name]},
  year={2024},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Data Availability

The datasets supporting this study are publicly available at Science Data Bank with DOI: https://doi.org/10.57760/sciencedb.28855

## Contact

For questions or collaboration inquiries, please contact:
- Xin Xia: xiax24@mails.tsinghua.edu.cn

## Acknowledgments

We thank the wind farm operators for providing the observational data and the meteorological services for the numerical weather prediction outputs used in this study.
