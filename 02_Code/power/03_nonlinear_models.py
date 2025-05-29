#!/usr/bin/env python3
"""
Nonlinear Power Modeling - Enhanced Version with LightGBM and SHAP
Stage 3: Multiple nonlinear models comparison with advanced analysis
Author: Research Team
Date: 2025-05-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedNonlinearModeler:
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
        self.shap_results = {}
        
    def load_and_prepare_data(self):
        """Load data and basic preprocessing (same as linear models)"""
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
        """Process wind direction variables to sin/cos components (consistent with previous analysis)"""
        print("Processing wind direction variables...")
        
        # Find wind direction columns
        wind_dir_cols = [col for col in self.data.columns if 'wind_direction' in col]
        print(f"Found {len(wind_dir_cols)} wind direction columns: {wind_dir_cols}")
        
        # Process each wind direction column
        wind_dir_processed = {}
        for col in wind_dir_cols:
            print(f"Processing {col}...")
            
            # Direct conversion method (consistent with previous analysis)
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
        """Create feature matrix from processed data (same as linear models)"""
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
    
    def fit_polynomial_model(self):
        """Fit polynomial regression with interaction terms"""
        print("Fitting polynomial regression model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try different polynomial degrees
        poly_results = {}
        for degree in [2, 3]:
            print(f"  Trying polynomial degree {degree}...")
            
            # Create polynomial pipeline
            poly_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            
            try:
                # Fit model
                poly_pipeline.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_train = poly_pipeline.predict(X_train_scaled)
                y_pred_test = poly_pipeline.predict(X_test_scaled)
                
                # Cross-validation (on smaller sample due to computational cost)
                sample_size = min(5000, len(X_train_scaled))
                indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
                cv_scores = cross_val_score(poly_pipeline, X_train_scaled[indices], y_train[indices], 
                                          cv=3, scoring='neg_mean_squared_error')
                
                # Metrics
                train_metrics = self.calculate_metrics(y_train, y_pred_train, 'Train')
                test_metrics = self.calculate_metrics(y_test, y_pred_test, 'Test')
                
                poly_results[f'Polynomial (degree {degree})'] = {
                    'model': poly_pipeline,
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
                
                print(f"    Degree {degree} - Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
                
            except Exception as e:
                print(f"    Degree {degree} failed: {str(e)}")
        
        return poly_results
    
    def fit_random_forest(self):
        """Fit Random Forest regression with improved parameters to reduce overfitting"""
        print("Fitting Random Forest model with anti-overfitting parameters...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Random Forest doesn't require scaling, but we'll scale for consistency
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Improved parameter grid to reduce overfitting
        improved_param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [8, 12, 15],          # Reduced from 20
            'min_samples_split': [5, 10],       # Increased from 2
            'min_samples_leaf': [2, 4],         # Increased from 1
            'max_features': ['sqrt', 0.8]       # Limited feature selection
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, improved_param_grid, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_rf = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_pred_train = best_rf.predict(X_train_scaled)
        y_pred_test = best_rf.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, 
                                  cv=3, scoring='neg_mean_squared_error')
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train, 'Train')
        test_metrics = self.calculate_metrics(y_test, y_pred_test, 'Test')
        
        print(f"  Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
        print(f"  Train-Test R² gap: {train_metrics['R²'] - test_metrics['R²']:.3f} (smaller is better)")
        
        return {
            'Random Forest (Improved)': {
                'model': best_rf,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'best_params': grid_search.best_params_
            }
        }
    
    def fit_lightgbm(self):
        """Fit LightGBM model with regularization"""
        print("Fitting LightGBM model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # LightGBM doesn't require scaling, but we'll use original features
        X_train_lgb = X_train
        X_test_lgb = X_test
        
        # Parameter grid for LightGBM
        lgb_param_grid = {
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.05, 0.1, 0.15],
            'feature_fraction': [0.8, 0.9],
            'bagging_fraction': [0.8, 0.9],
            'reg_alpha': [0.1, 0.5, 1.0],      # L1 regularization
            'reg_lambda': [0.1, 0.5, 1.0],     # L2 regularization
            'min_child_samples': [20, 30]       # Minimum samples in leaf
        }
        
        # Use smaller grid for efficiency
        simple_lgb_params = {
            'num_leaves': [31, 63],
            'learning_rate': [0.05, 0.1],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5]
        }
        
        # Create LightGBM regressor
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            n_estimators=200,
            random_state=42,
            verbose=-1
        )
        
        # Grid search
        grid_search = GridSearchCV(
            lgb_model, simple_lgb_params, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_lgb, y_train)
        best_lgb = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_pred_train = best_lgb.predict(X_train_lgb)
        y_pred_test = best_lgb.predict(X_test_lgb)
        
        # Cross-validation
        cv_scores = cross_val_score(best_lgb, X_train_lgb, y_train,
                                  cv=3, scoring='neg_mean_squared_error')
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train, 'Train')
        test_metrics = self.calculate_metrics(y_test, y_pred_test, 'Test')
        
        print(f"  Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
        print(f"  Train-Test R² gap: {train_metrics['R²'] - test_metrics['R²']:.3f} (smaller is better)")
        
        return {
            'LightGBM': {
                'model': best_lgb,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'X_train': X_train_lgb,
                'X_test': X_test_lgb,
                'y_train': y_train,
                'y_test': y_test,
                'best_params': grid_search.best_params_
            }
        }
    
    def fit_neural_network(self):
        """Fit Multi-layer Perceptron (Neural Network)"""
        print("Fitting Neural Network (MLP) model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Scale features (important for neural networks)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use smaller grid for computational efficiency
        simple_param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.01, 0.1]
        }
        
        mlp = MLPRegressor(random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.1)
        
        grid_search = GridSearchCV(
            mlp, simple_param_grid, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_mlp = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_pred_train = best_mlp.predict(X_train_scaled)
        y_pred_test = best_mlp.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(best_mlp, X_train_scaled, y_train, 
                                  cv=3, scoring='neg_mean_squared_error')
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train, 'Train')
        test_metrics = self.calculate_metrics(y_test, y_pred_test, 'Test')
        
        print(f"  Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
        
        return {
            'Neural Network (MLP)': {
                'model': best_mlp,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'best_params': grid_search.best_params_
            }
        }
    
    def fit_support_vector_regression(self):
        """Fit Support Vector Regression"""
        print("Fitting Support Vector Regression model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Scale features (important for SVR)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use smaller dataset for SVR due to computational cost
        if len(X_train_scaled) > 10000:
            print("  Using subset of data for SVR due to computational constraints...")
            indices = np.random.choice(len(X_train_scaled), 10000, replace=False)
            X_train_subset = X_train_scaled[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train_scaled
            y_train_subset = y_train
        
        # Use smaller grid for computational efficiency
        simple_param_grid = {
            'kernel': ['rbf'],
            'C': [1, 10],
            'gamma': ['scale']
        }
        
        svr = SVR()
        
        grid_search = GridSearchCV(
            svr, simple_param_grid, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_subset, y_train_subset)
        best_svr = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_pred_train = best_svr.predict(X_train_scaled)
        y_pred_test = best_svr.predict(X_test_scaled)
        
        # Cross-validation on subset
        cv_scores = cross_val_score(best_svr, X_train_subset, y_train_subset, 
                                  cv=3, scoring='neg_mean_squared_error')
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train, 'Train')
        test_metrics = self.calculate_metrics(y_test, y_pred_test, 'Test')
        
        print(f"  Test R²: {test_metrics['R²']:.3f}, Test RMSE: {test_metrics['RMSE']:.3f}")
        
        return {
            'Support Vector Regression': {
                'model': best_svr,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'best_params': grid_search.best_params_
            }
        }
    
    def fit_all_models(self):
        """Fit all nonlinear models including LightGBM"""
        print("Fitting all nonlinear models...")
        
        # 1. Polynomial regression
        poly_results = self.fit_polynomial_model()
        self.results.update(poly_results)
        
        # 2. Improved Random Forest
        rf_results = self.fit_random_forest()
        self.results.update(rf_results)
        
        # 3. LightGBM (New!)
        lgb_results = self.fit_lightgbm()
        self.results.update(lgb_results)
        
        # 4. Neural Network
        nn_results = self.fit_neural_network()
        self.results.update(nn_results)
        
        # 5. Support Vector Regression
        svr_results = self.fit_support_vector_regression()
        self.results.update(svr_results)
        
        print(f"\nTotal models fitted: {len(self.results)}")
    
    def perform_shap_analysis(self):
        """Perform SHAP analysis for tree-based models"""
        print("Performing SHAP analysis...")
        
        shap_models = {}
        
        # Find tree-based models for SHAP analysis
        for model_name, result in self.results.items():
            model = result['model']
            
            # SHAP analysis for Random Forest and LightGBM
            if ('Random Forest' in model_name or 'LightGBM' in model_name):
                print(f"  Analyzing {model_name} with SHAP...")
                
                try:
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    
                    # Use a sample of test data for SHAP (for efficiency)
                    X_test = result['X_test']
                    sample_size = min(1000, len(X_test))
                    indices = np.random.choice(len(X_test), sample_size, replace=False)
                    X_sample = X_test[indices]
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Store results
                    shap_models[model_name] = {
                        'explainer': explainer,
                        'shap_values': shap_values,
                        'X_sample': X_sample,
                        'feature_names': self.feature_names
                    }
                    
                    print(f"    SHAP analysis completed for {model_name}")
                    
                except Exception as e:
                    print(f"    SHAP analysis failed for {model_name}: {str(e)}")
        
        self.shap_results = shap_models
        return shap_models
    
    def create_shap_visualizations(self):
        """Create SHAP visualizations"""
        print("Creating SHAP visualizations...")
        
        if not self.shap_results:
            print("No SHAP results available")
            return
        
        # Create SHAP plots for each model
        for model_name, shap_data in self.shap_results.items():
            print(f"  Creating SHAP plots for {model_name}...")
            
            shap_values = shap_data['shap_values']
            X_sample = shap_data['X_sample']
            feature_names = [name.replace('obs_', '').replace('_', ' ') for name in shap_data['feature_names']]
            
            # Create figure with multiple SHAP plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'SHAP Analysis - {model_name}', fontsize=16, fontweight='bold')
            
            # 1. Summary plot (bar)
            plt.subplot(2, 2, 1)
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            plot_type="bar", show=False, max_display=10)
            plt.title('Feature Importance (SHAP)')
            
            # 2. Summary plot (beeswarm)
            plt.subplot(2, 2, 2)
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            show=False, max_display=10)
            plt.title('Feature Impact Distribution')
            
            # 3. Feature importance comparison
            ax3 = plt.subplot(2, 2, 3)
            feature_importance = np.abs(shap_values).mean(0)
            sorted_idx = np.argsort(feature_importance)[-10:]
            
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top 10 Features (SHAP Importance)')
            plt.grid(True, alpha=0.3)
            
            # 4. Waterfall plot for a single prediction
            ax4 = plt.subplot(2, 2, 4)
            if len(X_sample) > 0:
                try:
                    # Create waterfall plot for first sample
                    shap.waterfall_plot(
                        shap.Explanation(values=shap_values[0], 
                                       base_values=shap_data['explainer'].expected_value,
                                       data=X_sample[0],
                                       feature_names=feature_names),
                        show=False
                    )
                    plt.title('Single Prediction Explanation')
                except:
                    plt.text(0.5, 0.5, 'Waterfall plot\nnot available', 
                            ha='center', va='center', transform=ax4.transAxes)
                    plt.title('Single Prediction Explanation')
            
            plt.tight_layout()
            
            # Save SHAP plot
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            plt.savefig(f'{self.results_path}/03_shap_analysis_{safe_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models (traditional + SHAP)"""
        print("Analyzing feature importance...")
        
        importance_results = {}
        
        for model_name, result in self.results.items():
            model = result['model']
            
            # Traditional feature importance
            if hasattr(model, 'feature_importances_'):
                importance_results[model_name] = {
                    'features': self.feature_names,
                    'importance': model.feature_importances_,
                    'importance_type': 'Traditional'
                }
            
            # SHAP feature importance
            if model_name in self.shap_results:
                shap_data = self.shap_results[model_name]
                shap_importance = np.abs(shap_data['shap_values']).mean(0)
                
                importance_results[f"{model_name}_SHAP"] = {
                    'features': self.feature_names,
                    'importance': shap_importance,
                    'importance_type': 'SHAP'
                }
        
        return importance_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations including overfitting analysis"""
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
        plt.xticks(x, [name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=9)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Overfitting analysis (Train-Test R² gap)
        ax2 = plt.subplot(2, 3, 2)
        r2_gaps = [self.results[name]['train_metrics']['R²'] - self.results[name]['test_metrics']['R²'] 
                   for name in model_names]
        
        colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in r2_gaps]
        plt.bar(range(len(model_names)), r2_gaps, color=colors, alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Train R² - Test R²')
        plt.title('Overfitting Analysis\n(Lower is Better)')
        plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=9)
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High Overfitting')
        plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderate Overfitting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Best model predictions vs actual
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_result = self.results[best_model_name]
        
        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(best_result['y_test'], best_result['test_predictions'], 
                   alpha=0.5, s=10)
        min_val = min(best_result['y_test'].min(), best_result['test_predictions'].min())
        max_val = max(best_result['y_test'].max(), best_result['test_predictions'].max())
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
        plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 5. Feature importance comparison (Traditional vs SHAP if available)
        ax5 = plt.subplot(2, 3, 5)
        
        # Try to show both traditional and SHAP importance for tree models
        tree_models = [name for name in model_names if 'Random Forest' in name or 'LightGBM' in name]
        
        if tree_models and self.shap_results:
            # Show SHAP importance for the best tree model
            best_tree_model = tree_models[0]
            
            if best_tree_model in self.shap_results:
                shap_importance = np.abs(self.shap_results[best_tree_model]['shap_values']).mean(0)
                sorted_idx = np.argsort(shap_importance)[-10:]
                
                plt.barh(range(len(sorted_idx)), shap_importance[sorted_idx])
                feature_labels = [self.feature_names[i].replace('obs_', '').replace('_', ' ') 
                                 for i in sorted_idx]
                plt.yticks(range(len(sorted_idx)), feature_labels)
                plt.xlabel('Mean |SHAP Value|')
                plt.title(f'Top 10 Features (SHAP)\n{best_tree_model}')
            else:
                # Fallback to traditional importance
                if hasattr(self.results[best_tree_model]['model'], 'feature_importances_'):
                    feature_importance = self.results[best_tree_model]['model'].feature_importances_
                    sorted_idx = np.argsort(feature_importance)[-10:]
                    
                    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                    feature_labels = [self.feature_names[i].replace('obs_', '').replace('_', ' ') 
                                     for i in sorted_idx]
                    plt.yticks(range(len(sorted_idx)), feature_labels)
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top 10 Features\n{best_tree_model}')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes)
            plt.title('Feature Importance')
        
        plt.grid(True, alpha=0.3)
        
        # 6. Model complexity vs performance
        ax6 = plt.subplot(2, 3, 6)
        
        # Define model complexity (subjective ranking)
        complexity_map = {
            'Polynomial (degree 2)': 2,
            'Polynomial (degree 3)': 3,
            'Random Forest (Improved)': 4,
            'LightGBM': 4,
            'Neural Network (MLP)': 5,
            'Support Vector Regression': 3
        }
        
        complexities = [complexity_map.get(name, 3) for name in model_names]
        test_r2_values = [self.results[name]['test_metrics']['R²'] for name in model_names]
        
        plt.scatter(complexities, test_r2_values, s=100, alpha=0.7)
        
        # Add model names as labels
        for i, name in enumerate(model_names):
            plt.annotate(name.replace(' ', '\n'), (complexities[i], test_r2_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Complexity (Subjective Scale)')
        plt.ylabel('Test R²')
        plt.title('Model Complexity vs Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/03_enhanced_nonlinear_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save detailed results including SHAP analysis"""
        print("Saving results...")
        
        # 1. Model performance comparison with overfitting metrics
        performance_data = []
        for name, result in self.results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            cv_rmse = np.sqrt(-np.mean(result['cv_scores']))
            
            # Calculate overfitting metrics
            r2_gap = train_metrics['R²'] - test_metrics['R²']
            rmse_ratio = test_metrics['RMSE'] / train_metrics['RMSE']
            
            # Get best parameters if available
            best_params = result.get('best_params', {})
            
            performance_data.append({
                'Model': name,
                'Train_RMSE': train_metrics['RMSE'],
                'Train_MAE': train_metrics['MAE'],
                'Train_R²': train_metrics['R²'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_MAE': test_metrics['MAE'],
                'Test_R²': test_metrics['R²'],
                'CV_RMSE': cv_rmse,
                'R²_Gap': r2_gap,
                'RMSE_Ratio': rmse_ratio,
                'Overfitting_Risk': 'High' if r2_gap > 0.1 else 'Moderate' if r2_gap > 0.05 else 'Low',
                'Best_Params': str(best_params)
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(f'{self.results_path}/03_enhanced_model_performance.csv', index=False)
        
        # 2. Feature importance (traditional + SHAP)
        importance_results = self.analyze_feature_importance()
        
        for model_name, importance_data in importance_results.items():
            importance_df = pd.DataFrame({
                'Feature': importance_data['features'],
                'Importance': importance_data['importance'],
                'Importance_Type': importance_data['importance_type']
            }).sort_values('Importance', ascending=False)
            
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            importance_df.to_csv(f'{self.results_path}/03_{safe_name}_feature_importance.csv', index=False)
        
        # 3. SHAP results summary
        if self.shap_results:
            shap_summary = []
            for model_name, shap_data in self.shap_results.items():
                shap_values = shap_data['shap_values']
                feature_names = shap_data['feature_names']
                
                # Calculate mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(0)
                
                for i, feature in enumerate(feature_names):
                    shap_summary.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Mean_Abs_SHAP': mean_shap[i],
                        'SHAP_Std': np.abs(shap_values[:, i]).std()
                    })
            
            shap_df = pd.DataFrame(shap_summary)
            shap_df.to_csv(f'{self.results_path}/03_shap_summary.csv', index=False)
        
        # 4. Predictions from best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_metrics']['R²'])
        best_result = self.results[best_model_name]
        
        # Save test predictions
        test_predictions_df = pd.DataFrame({
            'Actual_Power': best_result['y_test'],
            'Predicted_Power': best_result['test_predictions'],
            'Residuals': best_result['y_test'] - best_result['test_predictions']
        })
        test_predictions_df.to_csv(f'{self.results_path}/03_enhanced_test_predictions.csv', index=False)
        
        # 5. Model ranking summary
        ranking_data = []
        for name, result in self.results.items():
            test_r2 = result['test_metrics']['R²']
            test_rmse = result['test_metrics']['RMSE']
            r2_gap = result['train_metrics']['R²'] - test_r2
            
            ranking_data.append({
                'Model': name,
                'Test_R²': test_r2,
                'Test_RMSE': test_rmse,
                'Overfitting_Gap': r2_gap,
                'Overall_Score': test_r2 - 0.1 * r2_gap  # Penalize overfitting
            })
        
        ranking_df = pd.DataFrame(ranking_data).sort_values('Overall_Score', ascending=False)
        ranking_df.to_csv(f'{self.results_path}/03_model_ranking.csv', index=False)
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Test R²: {best_result['test_metrics']['R²']:.3f}")
        print(f"Test RMSE: {best_result['test_metrics']['RMSE']:.3f}")
        print(f"Overfitting gap: {best_result['train_metrics']['R²'] - best_result['test_metrics']['R²']:.3f}")
        print(f"Total features used: {len(self.feature_names)}")
        
        # Print all model performances with overfitting analysis
        print("\nAll model performances:")
        for name, result in self.results.items():
            test_r2 = result['test_metrics']['R²']
            test_rmse = result['test_metrics']['RMSE']
            r2_gap = result['train_metrics']['R²'] - test_r2
            overfitting_status = 'High' if r2_gap > 0.1 else 'Moderate' if r2_gap > 0.05 else 'Low'
            print(f"  {name}: R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}, Overfitting = {overfitting_status} ({r2_gap:.3f})")
    
    def run_analysis(self):
        """Run complete enhanced nonlinear modeling analysis"""
        print("=== Enhanced Nonlinear Power Modeling Analysis ===")
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Create features (including wind direction processing)
        self.create_features()
        
        # 3. Fit all nonlinear models (including LightGBM)
        self.fit_all_models()
        
        # 4. Perform SHAP analysis
        self.perform_shap_analysis()
        
        # 5. Create SHAP visualizations
        self.create_shap_visualizations()
        
        # 6. Create comprehensive visualizations
        self.create_visualizations()
        
        # 7. Save results
        self.save_results()
        
        print("\nEnhanced nonlinear modeling analysis completed successfully!")
        print("\nKey improvements in this version:")
        print("✓ Added LightGBM model with regularization")
        print("✓ Improved Random Forest parameters to reduce overfitting")
        print("✓ SHAP analysis for interpretable feature importance")
        print("✓ Overfitting analysis and metrics")
        print("✓ Enhanced visualizations and model comparisons")
        
        return self.results

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/03_nonlinear_models"
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Run enhanced analysis
    modeler = EnhancedNonlinearModeler(DATA_PATH, RESULTS_PATH)
    results = modeler.run_analysis()