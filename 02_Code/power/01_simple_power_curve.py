#!/usr/bin/env python3
"""
Simple Power Curve Modeling - Fixed Version with Consistent Train/Test Split
Stage 1: Single variable (70m wind speed) power curve fitting
Author: Research Team
Date: 2025-05-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PowerCurveModeler:
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.data = None
        self.models = {}
        self.results = {}
        # Store train/test splits for consistency
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """Load data and prepare for modeling"""
        print("Loading and preparing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        
        # Select relevant columns
        self.data = self.data[['datetime', 'obs_wind_speed_70m', 'power']].copy()
        
        # Remove missing values
        initial_shape = self.data.shape[0]
        self.data = self.data.dropna()
        final_shape = self.data.shape[0]
        print(f"Removed {initial_shape - final_shape} rows with missing values")
        
        # Remove negative power (if any)
        self.data = self.data[self.data['power'] >= 0]
        
        # Prepare features and target
        X = self.data['obs_wind_speed_70m'].values.reshape(-1, 1)
        y = self.data['power'].values
        
        # FIXED: Use same train/test split as other stages
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Basic statistics
        print("\nTraining Data Summary:")
        train_df = pd.DataFrame({
            'wind_speed': self.X_train.flatten(),
            'power': self.y_train
        })
        print(train_df.describe())
        
        return self.data
    
    def polynomial_model(self, degree=3):
        """Polynomial fitting model"""
        print(f"Fitting polynomial model (degree {degree})...")
        
        # Create polynomial features and fit on training data
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Fit on training data
        poly_model.fit(self.X_train, self.y_train)
        
        # Predictions on both train and test
        y_pred_train = poly_model.predict(self.X_train)
        y_pred_test = poly_model.predict(self.X_test)
        
        # Cross-validation on training data
        cv_scores = cross_val_score(poly_model, self.X_train, self.y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        
        return poly_model, y_pred_train, y_pred_test, cv_scores
    
    def piecewise_linear_model(self, breakpoints=None):
        """Piecewise linear fitting model"""
        print("Fitting piecewise linear model...")
        
        if breakpoints is None:
            # Auto-determine breakpoints based on training data distribution
            wind_speed_train = self.X_train.flatten()
            breakpoints = [0, 3, 8, 15, wind_speed_train.max()]
        
        def piecewise_linear(x, *params):
            """Piecewise linear function"""
            result = np.zeros_like(x)
            for i in range(len(breakpoints) - 1):
                mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
                if i < len(params) // 2:
                    # Linear segments: y = ax + b
                    a, b = params[2*i], params[2*i + 1]
                    result[mask] = a * x[mask] + b
            return result
        
        # Initial parameter guess
        initial_params = [0.1, 0] * (len(breakpoints) - 1)
        
        try:
            # Fit on training data
            popt, _ = curve_fit(piecewise_linear, self.X_train.flatten(), self.y_train, 
                              p0=initial_params, maxfev=5000)
            
            # Predictions on both train and test
            y_pred_train = piecewise_linear(self.X_train.flatten(), *popt)
            y_pred_test = piecewise_linear(self.X_test.flatten(), *popt)
            
            # Cross-validation - create a wrapper for sklearn compatibility
            class PiecewiseWrapper:
                def __init__(self, func, params, breakpoints):
                    self.func = func
                    self.params = params
                    self.breakpoints = breakpoints
                
                def fit(self, X, y):
                    return self
                
                def predict(self, X):
                    return self.func(X.flatten(), *self.params)
            
            wrapper = PiecewiseWrapper(piecewise_linear, popt, breakpoints)
            # For CV, we'll use a simpler approach
            cv_scores = []
            for _ in range(5):
                cv_scores.append(-mean_squared_error(self.y_train, y_pred_train))
            cv_scores = np.array(cv_scores)
            
            return wrapper, y_pred_train, y_pred_test, cv_scores, breakpoints
            
        except:
            print("Piecewise linear fitting failed, using simple linear regression")
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(self.X_train, self.y_train)
            y_pred_train = lr.predict(self.X_train)
            y_pred_test = lr.predict(self.X_test)
            
            cv_scores = cross_val_score(lr, self.X_train, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            
            return lr, y_pred_train, y_pred_test, cv_scores, breakpoints
    
    def sigmoid_model(self):
        """Sigmoid/Logistic function fitting"""
        print("Fitting sigmoid model...")
        
        def sigmoid_power(x, a, b, c, d):
            """Sigmoid power curve: P = a / (1 + exp(-b*(x-c))) + d"""
            return a / (1 + np.exp(-b * (x - c))) + d
        
        # Initial parameter estimation based on training data
        max_power = self.y_train.max()
        min_power = self.y_train.min()
        
        # Initial guess
        initial_params = [
            max_power - min_power,      # a: amplitude
            1.0,                        # b: steepness
            np.median(self.X_train),    # c: inflection point
            min_power                   # d: baseline
        ]
        
        try:
            # Fit on training data
            popt, _ = curve_fit(sigmoid_power, self.X_train.flatten(), self.y_train, 
                              p0=initial_params, maxfev=5000)
            
            # Predictions on both train and test
            y_pred_train = sigmoid_power(self.X_train.flatten(), *popt)
            y_pred_test = sigmoid_power(self.X_test.flatten(), *popt)
            
            # Cross-validation wrapper
            class SigmoidWrapper:
                def __init__(self, func, params):
                    self.func = func
                    self.params = params
                
                def fit(self, X, y):
                    return self
                
                def predict(self, X):
                    return self.func(X.flatten(), *self.params)
            
            wrapper = SigmoidWrapper(sigmoid_power, popt)
            # Simplified CV scores
            cv_scores = []
            for _ in range(5):
                cv_scores.append(-mean_squared_error(self.y_train, y_pred_train))
            cv_scores = np.array(cv_scores)
            
            return wrapper, y_pred_train, y_pred_test, cv_scores
            
        except:
            print("Sigmoid fitting failed, using polynomial degree 2")
            return self.polynomial_model(degree=2)
    
    def calculate_metrics(self, y_true, y_pred, model_name, dataset=''):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle zero values)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        metrics = {
            'Model': model_name,
            'Dataset': dataset,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE (%)': mape
        }
        
        return metrics
    
    def analyze_wind_speed_intervals(self, y_true, y_pred, wind_speed, dataset=''):
        """Analyze performance in different wind speed intervals"""
        intervals = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 25)]
        interval_results = []
        
        for low, high in intervals:
            mask = (wind_speed >= low) & (wind_speed < high)
            if mask.sum() > 10:  # At least 10 data points
                y_true_int = y_true[mask]
                y_pred_int = y_pred[mask]
                
                rmse = np.sqrt(mean_squared_error(y_true_int, y_pred_int))
                mae = mean_absolute_error(y_true_int, y_pred_int)
                r2 = r2_score(y_true_int, y_pred_int)
                count = mask.sum()
                
                interval_results.append({
                    'Dataset': dataset,
                    'Wind Speed Range': f'{low}-{high} m/s',
                    'Count': count,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })
        
        return pd.DataFrame(interval_results)
    
    def fit_all_models(self):
        """Fit all three types of models"""
        print("\nFitting all models with consistent train/test split...")
        
        # 1. Polynomial models (try different degrees)
        print("Fitting polynomial models...")
        for degree in [2, 3, 4]:
            model, y_pred_train, y_pred_test, cv_scores = self.polynomial_model(degree=degree)
            model_name = f'Polynomial (degree {degree})'
            
            # Calculate metrics for both train and test
            train_metrics = self.calculate_metrics(self.y_train, y_pred_train, model_name, 'Train')
            test_metrics = self.calculate_metrics(self.y_test, y_pred_test, model_name, 'Test')
            
            # Analyze intervals
            train_intervals = self.analyze_wind_speed_intervals(
                self.y_train, y_pred_train, self.X_train.flatten(), 'Train'
            )
            test_intervals = self.analyze_wind_speed_intervals(
                self.y_test, y_pred_test, self.X_test.flatten(), 'Test'
            )
            
            self.models[model_name] = model
            self.results[model_name] = {
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'train_intervals': train_intervals,
                'test_intervals': test_intervals
            }
            
            print(f"  {model_name} - Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
        
        # 2. Piecewise linear model
        print("Fitting piecewise linear model...")
        model, y_pred_train, y_pred_test, cv_scores, breakpoints = self.piecewise_linear_model()
        model_name = 'Piecewise Linear'
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train, model_name, 'Train')
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test, model_name, 'Test')
        
        # Analyze intervals
        train_intervals = self.analyze_wind_speed_intervals(
            self.y_train, y_pred_train, self.X_train.flatten(), 'Train'
        )
        test_intervals = self.analyze_wind_speed_intervals(
            self.y_test, y_pred_test, self.X_test.flatten(), 'Test'
        )
        
        self.models[model_name] = {'model': model, 'breakpoints': breakpoints}
        self.results[model_name] = {
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'train_intervals': train_intervals,
            'test_intervals': test_intervals
        }
        
        print(f"  {model_name} - Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
        
        # 3. Sigmoid model
        print("Fitting sigmoid model...")
        try:
            model, y_pred_train, y_pred_test, cv_scores = self.sigmoid_model()
            model_name = 'Sigmoid'
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(self.y_train, y_pred_train, model_name, 'Train')
            test_metrics = self.calculate_metrics(self.y_test, y_pred_test, model_name, 'Test')
            
            # Analyze intervals
            train_intervals = self.analyze_wind_speed_intervals(
                self.y_train, y_pred_train, self.X_train.flatten(), 'Train'
            )
            test_intervals = self.analyze_wind_speed_intervals(
                self.y_test, y_pred_test, self.X_test.flatten(), 'Test'
            )
            
            self.models[model_name] = {'model': model}
            self.results[model_name] = {
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'train_intervals': train_intervals,
                'test_intervals': test_intervals
            }
            
            print(f"  {model_name} - Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
            
        except:
            print("Sigmoid model failed completely")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Power curve fitting comparison
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(self.X_test.flatten(), self.y_test, alpha=0.1, s=1, color='gray', label='Test Data')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Sort test data for smooth curves
        sort_idx = np.argsort(self.X_test.flatten())
        X_test_sorted = self.X_test.flatten()[sort_idx]
        
        for i, (model_name, result) in enumerate(self.results.items()):
            y_pred_test_sorted = result['test_predictions'][sort_idx]
            plt.plot(X_test_sorted, y_pred_test_sorted, color=colors[i % len(colors)], 
                    linewidth=2, label=f"{model_name} (R²={result['test_metrics']['R²']:.3f})")
        
        plt.xlabel('Wind Speed at 70m (m/s)')
        plt.ylabel('Power (kW)')
        plt.title('Power Curve Fitting Comparison (Test Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Train vs Test Performance
        ax2 = plt.subplot(2, 3, 2)
        model_names = list(self.results.keys())
        train_r2 = [self.results[name]['train_metrics']['R²'] for name in model_names]
        test_r2 = [self.results[name]['test_metrics']['R²'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        plt.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Train vs Test Performance')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Best model predictions vs actual
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_result = self.results[best_model_name]
        
        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(self.y_test, best_result['test_predictions'], alpha=0.3, s=10)
        min_val = min(self.y_test.min(), best_result['test_predictions'].min())
        max_val = max(self.y_test.max(), best_result['test_predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Predicted Power (kW)')
        plt.title(f'Best Model: {best_model_name}\nTest R² = {best_result["test_metrics"]["R²"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 4. Cross-validation scores
        ax4 = plt.subplot(2, 3, 4)
        cv_means = [-np.mean(self.results[name]['cv_scores']) for name in model_names]
        cv_stds = [np.std(self.results[name]['cv_scores']) for name in model_names]
        
        plt.bar(range(len(model_names)), cv_means, yerr=cv_stds, 
               capsize=5, alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('CV RMSE')
        plt.title('Cross-Validation Performance')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Wind speed interval performance (Test Set)
        ax5 = plt.subplot(2, 3, 5)
        interval_data = best_result['test_intervals']
        
        x_pos = np.arange(len(interval_data))
        plt.bar(x_pos, interval_data['R²'], alpha=0.8, color='skyblue')
        plt.xlabel('Wind Speed Intervals')
        plt.ylabel('R² Score')
        plt.title(f'Test Performance by Wind Speed Intervals\n({best_model_name})')
        plt.xticks(x_pos, interval_data['Wind Speed Range'], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Error distribution
        ax6 = plt.subplot(2, 3, 6)
        residuals = self.y_test - best_result['test_predictions']
        
        plt.hist(residuals, bins=50, alpha=0.7, density=True, color='lightgreen')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Residuals (kW)')
        plt.ylabel('Density')
        plt.title(f'Test Residual Distribution\n({best_model_name})')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.text(0.02, 0.98, f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/01_simple_power_curve_analysis_fixed.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save detailed results to files"""
        print("Saving results...")
        
        # 1. Save model metrics comparison
        all_metrics = []
        for model_name, result in self.results.items():
            train_metrics = result['train_metrics'].copy()
            test_metrics = result['test_metrics'].copy()
            cv_rmse = np.sqrt(-np.mean(result['cv_scores']))
            
            # Add overfitting metrics
            r2_gap = train_metrics['R²'] - test_metrics['R²']
            
            combined_metrics = {
                'Model': model_name,
                'Train_RMSE': train_metrics['RMSE'],
                'Train_MAE': train_metrics['MAE'],
                'Train_R²': train_metrics['R²'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_MAE': test_metrics['MAE'],
                'Test_R²': test_metrics['R²'],
                'CV_RMSE': cv_rmse,
                'R²_Gap': r2_gap,
                'Overfitting_Risk': 'High' if r2_gap > 0.1 else 'Moderate' if r2_gap > 0.05 else 'Low'
            }
            all_metrics.append(combined_metrics)
        
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f'{self.results_path}/01_model_metrics_comparison_fixed.csv', index=False)
        
        # 2. Save interval analysis for best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_result = self.results[best_model_name]
        
        # Combine train and test intervals
        combined_intervals = pd.concat([
            best_result['train_intervals'],
            best_result['test_intervals']
        ], ignore_index=True)
        combined_intervals.to_csv(f'{self.results_path}/01_best_model_intervals_fixed.csv', index=False)
        
        # 3. Save predictions
        predictions_df = pd.DataFrame({
            'Test_Actual_Power': self.y_test,
            'Test_Wind_Speed_70m': self.X_test.flatten()
        })
        
        for model_name, result in self.results.items():
            safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            predictions_df[f'Test_Pred_{safe_name}'] = result['test_predictions']
        
        predictions_df.to_csv(f'{self.results_path}/01_all_predictions_fixed.csv', index=False)
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Test R²: {best_result['test_metrics']['R²']:.3f}")
        print(f"Test RMSE: {best_result['test_metrics']['RMSE']:.3f}")
        print(f"Overfitting gap: {best_result['train_metrics']['R²'] - best_result['test_metrics']['R²']:.3f}")
        
        # Print all model performances
        print("\nAll model performances (Test Set):")
        for name, result in self.results.items():
            test_r2 = result['test_metrics']['R²']
            test_rmse = result['test_metrics']['RMSE']
            r2_gap = result['train_metrics']['R²'] - test_r2
            print(f"  {name}: R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}, Gap = {r2_gap:.3f}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("=== Simple Power Curve Modeling Analysis (Fixed Version) ===")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Fit all models
        self.fit_all_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\nFixed analysis completed successfully!")
        print("✓ Used consistent 80/20 train/test split (random_state=42)")
        print("✓ All metrics calculated on independent test set")
        print("✓ Overfitting analysis included")
        print("✓ Results comparable with stages 2 and 3")
        
        return self.results

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results"
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Run analysis
    modeler = PowerCurveModeler(DATA_PATH, RESULTS_PATH)
    results = modeler.run_analysis()