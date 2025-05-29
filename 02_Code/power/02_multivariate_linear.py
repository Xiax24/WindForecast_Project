
#!/usr/bin/env python3
"""
Multivariate Linear Power Modeling - Simplified Version
Stage 2: Multiple variables linear regression with basic feature processing
Author: Research Team
Date: 2025-05-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MultivariateLinearModeler:
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.data = None
        self.features = None
        self.target = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load data and basic preprocessing"""
        print("Loading and preparing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"Original data shape: {self.data.shape}")
        
        # Select relevant columns (only observed data)
        obs_columns = [col for col in self.data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power']
        
        # Remove density and humidity as specified
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        self.data = self.data[obs_columns].copy()
        print(f"Selected columns after removing density and humidity: {len(obs_columns)}")
        print("Variables used:", [col for col in obs_columns if col not in ['datetime', 'power']])
        
        # Remove missing values
        initial_shape = self.data.shape[0]
        self.data = self.data.dropna()
        final_shape = self.data.shape[0]
        print(f"Removed {initial_shape - final_shape} rows with missing values")
        
        # Remove negative power
        self.data = self.data[self.data['power'] >= 0]
        
        print(f"Final data shape: {self.data.shape}")
        return self.data
    
    def process_wind_direction(self):
        """Process wind direction variables to sin/cos components (consistent with correlation analysis)"""
        print("Processing wind direction variables...")
        
        # Find wind direction columns
        wind_dir_cols = [col for col in self.data.columns if 'wind_direction' in col]
        print(f"Found {len(wind_dir_cols)} wind direction columns: {wind_dir_cols}")
        
        # Process each wind direction column
        wind_dir_processed = {}
        for col in wind_dir_cols:
            print(f"Processing {col}...")
            
            # Direct conversion method (consistent with correlation analysis)
            wind_dir_rad = np.deg2rad(self.data[col])
            
            # Create sin/cos components
            sin_col = col.replace('wind_direction', 'wind_dir_sin')
            cos_col = col.replace('wind_direction', 'wind_dir_cos')
            
            self.data[sin_col] = np.sin(wind_dir_rad)
            self.data[cos_col] = np.cos(wind_dir_rad)
            
            wind_dir_processed[col] = {'sin': sin_col, 'cos': cos_col}
            print(f"  Created: {sin_col}, {cos_col}")
        
        # Remove original wind direction columns
        self.data = self.data.drop(columns=wind_dir_cols)
        print(f"Removed original wind direction columns: {wind_dir_cols}")
        
        return wind_dir_processed
    
    def create_features(self):
        """Create feature matrix from processed data"""
        print("Creating feature matrix...")
        
        # Process wind direction first
        wind_dir_processed = self.process_wind_direction()
        
        # Select all observation variables (excluding datetime and power)
        feature_cols = [col for col in self.data.columns 
                       if col not in ['datetime', 'power']]
        
        print(f"Using {len(feature_cols)} features:")
        
        # Group features by type for display
        wind_speed_cols = [col for col in feature_cols if 'wind_speed' in col]
        wind_dir_cols = [col for col in feature_cols if 'wind_dir' in col]
        temp_cols = [col for col in feature_cols if 'temperature' in col]
        
        print(f"  Wind speed variables ({len(wind_speed_cols)}): {wind_speed_cols}")
        print(f"  Wind direction variables ({len(wind_dir_cols)}): {wind_dir_cols}")
        print(f"  Temperature variables ({len(temp_cols)}): {temp_cols}")
        
        # Create feature matrix
        self.features = self.data[feature_cols].values
        self.target = self.data['power'].values
        self.feature_names = feature_cols
        
        print(f"Feature matrix shape: {self.features.shape}")
        print(f"Target vector shape: {self.target.shape}")
        
        return feature_cols
    
    def fit_linear_models(self):
        """Fit various linear regression models"""
        print("Fitting linear regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Models to try
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net': ElasticNet()
        }
        
        # Hyperparameter grids for regularized models
        param_grids = {
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso Regression': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'Elastic Net': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        }
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name in param_grids:
                # Grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=5, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                print(f"  Best parameters: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = best_model.predict(X_train_scaled)
            y_pred_test = best_model.predict(X_test_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            
            # Metrics
            train_metrics = self.calculate_metrics(y_train, y_pred_train, 'Train')
            test_metrics = self.calculate_metrics(y_test, y_pred_test, 'Test')
            
            # Store results
            self.models[name] = best_model
            self.results[name] = {
                'model': best_model,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test
            }
            
            print(f"  Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
    
    def calculate_metrics(self, y_true, y_pred, dataset=''):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        return {
            'Dataset': dataset,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE (%)': mape
        }
    
    def analyze_coefficients(self):
        """Analyze model coefficients and feature importance"""
        print("Analyzing model coefficients...")
        
        # Get the best model (highest test R²)
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_model = self.results[best_model_name]['model']
        
        print(f"Best model: {best_model_name}")
        
        # Extract coefficients
        if hasattr(best_model, 'coef_'):
            coefficients = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': best_model.coef_,
                'Abs_Coefficient': np.abs(best_model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            print("\nTop 10 Feature Coefficients (sorted by absolute value):")
            print(coefficients.head(10))
            
            # Visualize coefficients
            plt.figure(figsize=(12, 8))
            top_15 = coefficients.head(15)
            colors = ['red' if x < 0 else 'blue' for x in top_15['Coefficient']]
            
            plt.barh(range(len(top_15)), top_15['Coefficient'], color=colors, alpha=0.7)
            plt.yticks(range(len(top_15)), [name.replace('obs_', '').replace('_', ' ') for name in top_15['Feature']])
            plt.xlabel('Coefficient Value')
            plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.results_path}/02_feature_coefficients.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            return coefficients
        
        return None
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model performance comparison
        ax1 = plt.subplot(2, 3, 1)
        model_names = list(self.results.keys())
        train_r2 = [self.results[name]['train_metrics']['R²'] for name in model_names]
        test_r2 = [self.results[name]['test_metrics']['R²'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        plt.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Best model predictions vs actual
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_result = self.results[best_model_name]
        
        ax2 = plt.subplot(2, 3, 2)
        plt.scatter(best_result['y_test'], best_result['test_predictions'], 
                   alpha=0.5, s=10)
        min_val = min(best_result['y_test'].min(), best_result['test_predictions'].min())
        max_val = max(best_result['y_test'].max(), best_result['test_predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Predicted Power (kW)')
        plt.title(f'Predictions vs Actual - {best_model_name}\nTest R² = {best_result["test_metrics"]["R²"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 3. Residual analysis
        ax3 = plt.subplot(2, 3, 3)
        residuals = best_result['y_test'] - best_result['test_predictions']
        plt.scatter(best_result['test_predictions'], residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Power (kW)')
        plt.ylabel('Residuals (kW)')
        plt.title('Residual Analysis')
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
        
        # 5. Feature importance (if available)
        ax5 = plt.subplot(2, 3, 5)
        best_model = best_result['model']
        if hasattr(best_model, 'coef_'):
            feature_importance = np.abs(best_model.coef_)
            sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
            
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            feature_labels = [self.feature_names[i].replace('obs_', '').replace('_', ' ') 
                             for i in sorted_idx]
            plt.yticks(range(len(sorted_idx)), feature_labels)
            plt.xlabel('|Coefficient|')
            plt.title('Top 10 Feature Importance')
            plt.grid(True, alpha=0.3)
        
        # 6. Error distribution
        ax6 = plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=50, alpha=0.7, density=True)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Residuals (kW)')
        plt.ylabel('Density')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.text(0.02, 0.98, f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/02_multivariate_linear_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save detailed results"""
        print("Saving results...")
        
        # 1. Model performance comparison
        performance_data = []
        for name, result in self.results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            cv_rmse = np.sqrt(-np.mean(result['cv_scores']))
            
            performance_data.append({
                'Model': name,
                'Train_RMSE': train_metrics['RMSE'],
                'Train_MAE': train_metrics['MAE'],
                'Train_R²': train_metrics['R²'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_MAE': test_metrics['MAE'],
                'Test_R²': test_metrics['R²'],
                'CV_RMSE': cv_rmse
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(f'{self.results_path}/02_model_performance.csv', index=False)
        
        # 2. Feature list
        feature_info = pd.DataFrame({
            'Feature_Index': range(len(self.feature_names)),
            'Feature_Name': self.feature_names,
            'Feature_Type': [self.get_feature_type(name) for name in self.feature_names]
        })
        feature_info.to_csv(f'{self.results_path}/02_feature_list.csv', index=False)
        
        # 3. Coefficients of best model
        coefficients = self.analyze_coefficients()
        if coefficients is not None:
            coefficients.to_csv(f'{self.results_path}/02_best_model_coefficients.csv', index=False)
        
        # 4. Predictions
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_result = self.results[best_model_name]
        
        # Save test predictions
        test_predictions_df = pd.DataFrame({
            'Actual_Power': best_result['y_test'],
            'Predicted_Power': best_result['test_predictions'],
            'Residuals': best_result['y_test'] - best_result['test_predictions']
        })
        test_predictions_df.to_csv(f'{self.results_path}/02_test_predictions.csv', index=False)
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Test R²: {best_result['test_metrics']['R²']:.3f}")
        print(f"Test RMSE: {best_result['test_metrics']['RMSE']:.3f}")
        print(f"Total features used: {len(self.feature_names)}")
    
    def get_feature_type(self, feature_name):
        """Categorize feature type"""
        if 'wind_speed' in feature_name:
            return 'Wind Speed'
        elif 'wind_dir' in feature_name:
            return 'Wind Direction'
        elif 'temperature' in feature_name:
            return 'Temperature'
        else:
            return 'Other'
    
    def run_analysis(self):
        """Run complete multivariate linear analysis"""
        print("=== Multivariate Linear Power Modeling Analysis (Simplified) ===")
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Create features (including wind direction processing)
        self.create_features()
        
        # 3. Fit linear models
        self.fit_linear_models()
        
        # 4. Create visualizations
        self.create_visualizations()
        
        # 5. Save results
        self.save_results()
        
        print("\nMultivariate linear analysis completed successfully!")
        return self.results

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results"
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Run analysis
    modeler = MultivariateLinearModeler(DATA_PATH, RESULTS_PATH)
    results = modeler.run_analysis()