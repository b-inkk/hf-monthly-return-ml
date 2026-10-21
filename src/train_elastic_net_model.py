"""
Monthly Stock Return Prediction - Elastic Net Model Training
===========================================================

This script trains an Elastic Net regression model for monthly stock return prediction
using features from 1998-2017. Based on previous testing (2017-2019), Elastic Net
was identified as the most effective model for this task.

Key Features:
- Elastic Net regression with L1 and L2 regularization
- Comprehensive hyperparameter tuning (alpha, l1_ratio)
- Cross-validation for robust model selection
- Feature importance analysis
- Complete documentation and model persistence

Training Data: 1998-2017 (26,523 records, 126 stocks, 13 features)
Target: Monthly stock returns (y)

Date: October 2025
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime
from typing import Dict, Tuple, Any

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ElasticNetTrainer:
    def __init__(self, output_dir: str = 'monthly_model_training'):
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'models')
        self.results_dir = os.path.join(output_dir, 'results')
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        # Create directories
        for dir_path in [self.output_dir, self.model_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"🚀 ELASTIC NET MODEL TRAINING")
        print("="*60)
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Focus: Elastic Net regression (most effective from 2017-2019 testing)")
        
        # Model components
        self.scaler = StandardScaler()
        self.model = None
        self.pipeline = None
        self.best_params = None
        self.cv_results = None
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.training_stats = {}
        
    def load_training_data(self) -> bool:
        """Load and prepare training data from 1998-2017"""
        
        print(f"\\n📊 LOADING TRAINING DATA")
        print("-"*40)
        
        # Load training features
        training_file = os.path.join('data', 'monthly_features_1998_2017.csv')
        if not os.path.exists(training_file):
            training_file = 'monthly_features_1998_2017_FIXED/combined/combined_monthly_features_FIXED.csv'
        
        if not os.path.exists(training_file):
            print(f"❌ Training file not found: {training_file}")
            return False
        
        try:
            df = pd.read_csv(training_file)
            print(f"✅ Loaded training data: {len(df):,} records")
            
            # Verify data completeness
            print(f"📈 Data overview:")
            print(f"  Date range: {df['YearMonth'].min()} to {df['YearMonth'].max()}")
            print(f"  Unique stocks: {df['ticker'].nunique()}")
            print(f"  Total columns: {len(df.columns)}")
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col.startswith('x')]
            self.feature_names = sorted(feature_cols)
            
            print(f"  Feature columns: {len(self.feature_names)}")
            print(f"  Features: {self.feature_names}")
            
            # Extract features and target
            self.X_train = df[self.feature_names].values
            self.y_train = df['y'].values
            
            # Check for missing values
            missing_features = pd.DataFrame(self.X_train, columns=self.feature_names).isnull().sum()
            missing_target = pd.isna(self.y_train).sum()
            
            print(f"\\n🔍 Data quality check:")
            if missing_features.sum() == 0:
                print(f"  ✅ Features: No missing values")
            else:
                print(f"  ⚠️ Features: {missing_features.sum()} missing values")
                print(missing_features[missing_features > 0])
            
            if missing_target == 0:
                print(f"  ✅ Target: No missing values")
            else:
                print(f"  ⚠️ Target: {missing_target} missing values")
            
            # Store training statistics
            self.training_stats = {
                'n_samples': len(df),
                'n_features': len(self.feature_names),
                'n_stocks': df['ticker'].nunique(),
                'date_range': f"{df['YearMonth'].min()} to {df['YearMonth'].max()}",
                'target_mean': float(np.mean(self.y_train)),
                'target_std': float(np.std(self.y_train)),
                'feature_names': self.feature_names
            }
            
            print(f"\\n📊 Target variable statistics:")
            print(f"  Mean return: {self.training_stats['target_mean']:.6f}")
            print(f"  Std deviation: {self.training_stats['target_std']:.6f}")
            print(f"  Min return: {np.min(self.y_train):.6f}")
            print(f"  Max return: {np.max(self.y_train):.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    
    def tune_hyperparameters(self) -> bool:
        """Perform comprehensive hyperparameter tuning for Elastic Net"""
        
        print(f"\\n🔧 HYPERPARAMETER TUNING")
        print("-"*35)
        
        # Define parameter grid based on previous successful configurations
        param_grid = {
            'elasticnet__alpha': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0],
            'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        print(f"📊 Parameter grid:")
        print(f"  Alpha values: {param_grid['elasticnet__alpha']}")
        print(f"  L1 ratio values: {param_grid['elasticnet__l1_ratio']}")
        print(f"  Total combinations: {len(param_grid['elasticnet__alpha']) * len(param_grid['elasticnet__l1_ratio'])}")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(random_state=42, max_iter=2000))
        ])
        
        # Use TimeSeriesSplit for cross-validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        print(f"\\n🔄 Running grid search with time series cross-validation...")
        print(f"  CV method: TimeSeriesSplit (5 splits)")
        print(f"  Scoring: Negative mean squared error")
        
        try:
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store results
            self.best_params = grid_search.best_params_
            self.cv_results = grid_search.cv_results_
            self.pipeline = grid_search.best_estimator_
            
            print(f"\\n✅ Hyperparameter tuning completed!")
            print(f"📊 Best parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
            
            print(f"\\n📈 Best cross-validation score:")
            print(f"  Neg MSE: {grid_search.best_score_:.6f}")
            print(f"  RMSE: {np.sqrt(-grid_search.best_score_):.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during hyperparameter tuning: {e}")
            return False
    
    def train_final_model(self) -> bool:
        """Train the final model with best parameters"""
        
        print(f"\\n🎯 TRAINING FINAL MODEL")
        print("-"*30)
        
        if self.pipeline is None:
            print(f"❌ No tuned pipeline available")
            return False
        
        try:
            # The pipeline is already fitted during grid search, but let's refit for clarity
            print(f"🔄 Training final Elastic Net model...")
            print(f"  Alpha: {self.best_params['elasticnet__alpha']}")
            print(f"  L1 ratio: {self.best_params['elasticnet__l1_ratio']}")
            
            self.pipeline.fit(self.X_train, self.y_train)
            
            # Get the trained model
            self.model = self.pipeline.named_steps['elasticnet']
            
            # Calculate training metrics
            y_pred_train = self.pipeline.predict(self.X_train)
            
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            train_r2 = r2_score(self.y_train, y_pred_train)
            
            print(f"\\n📊 Training performance:")
            print(f"  MSE: {train_mse:.6f}")
            print(f"  RMSE: {train_rmse:.6f}")
            print(f"  MAE: {train_mae:.6f}")
            print(f"  R²: {train_r2:.6f}")
            
            # Analyze feature importance (coefficients)
            coefficients = self.model.coef_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            print(f"\\n🔍 Feature importance (top 10):")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['coefficient']:.6f}")
            
            # Count non-zero coefficients
            non_zero_coef = np.sum(coefficients != 0)
            print(f"\\n📈 Model complexity:")
            print(f"  Non-zero coefficients: {non_zero_coef}/{len(coefficients)}")
            print(f"  Sparsity: {(len(coefficients) - non_zero_coef)/len(coefficients)*100:.1f}%")
            
            # Store training results
            self.training_stats.update({
                'best_alpha': float(self.best_params['elasticnet__alpha']),
                'best_l1_ratio': float(self.best_params['elasticnet__l1_ratio']),
                'train_mse': float(train_mse),
                'train_rmse': float(train_rmse),
                'train_mae': float(train_mae),
                'train_r2': float(train_r2),
                'n_nonzero_coef': int(non_zero_coef),
                'feature_importance': feature_importance.to_dict('records'),
                'intercept': float(self.model.intercept_)
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Error training final model: {e}")
            return False
    
    def save_model_and_results(self) -> bool:
        """Save the trained model and all results"""
        
        print(f"\\n💾 SAVING MODEL AND RESULTS")
        print("-"*35)
        
        try:
            # Save the complete pipeline
            model_file = os.path.join(self.model_dir, 'elastic_net_pipeline.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(self.pipeline, f)
            print(f"✅ Saved pipeline: {model_file}")
            
            # Save training statistics
            stats_file = os.path.join(self.results_dir, 'training_results.json')
            with open(stats_file, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            print(f"✅ Saved training stats: {stats_file}")
            
            # Save hyperparameter tuning results
            cv_results_file = os.path.join(self.results_dir, 'cv_results.json')
            cv_results_clean = {}
            for key, value in self.cv_results.items():
                if isinstance(value, np.ndarray):
                    cv_results_clean[key] = value.tolist()
                else:
                    cv_results_clean[key] = value
            
            with open(cv_results_file, 'w') as f:
                json.dump(cv_results_clean, f, indent=2)
            print(f"✅ Saved CV results: {cv_results_file}")
            
            # Create feature importance plot
            self.create_feature_importance_plot()
            
            # Create hyperparameter heatmap
            self.create_hyperparameter_heatmap()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model and results: {e}")
            return False
    
    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        
        try:
            feature_importance = pd.DataFrame(self.training_stats['feature_importance'])
            
            plt.figure(figsize=(12, 8))
            
            # Plot coefficients
            colors = ['red' if x < 0 else 'blue' for x in feature_importance['coefficient']]
            bars = plt.barh(range(len(feature_importance)), feature_importance['coefficient'], color=colors, alpha=0.7)
            
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('Coefficient Value')
            plt.title('Elastic Net Feature Importance (Coefficients)\\nRed: Negative, Blue: Positive')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, coef) in enumerate(zip(bars, feature_importance['coefficient'])):
                plt.text(coef + (0.001 if coef >= 0 else -0.001), i, f'{coef:.4f}', 
                        ha='left' if coef >= 0 else 'right', va='center', fontsize=9)
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.plots_dir, 'feature_importance.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved feature importance plot: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")
    
    def create_hyperparameter_heatmap(self):
        """Create hyperparameter tuning heatmap"""
        
        try:
            # Extract results for heatmap
            alphas = sorted(set([params['elasticnet__alpha'] for params in 
                               [eval(p) for p in self.cv_results['params']]]))
            l1_ratios = sorted(set([params['elasticnet__l1_ratio'] for params in 
                                  [eval(p) for p in self.cv_results['params']]]))
            
            # Create score matrix
            score_matrix = np.zeros((len(l1_ratios), len(alphas)))
            
            for i, params in enumerate(self.cv_results['params']):
                params_dict = eval(params)
                alpha_idx = alphas.index(params_dict['elasticnet__alpha'])
                l1_idx = l1_ratios.index(params_dict['elasticnet__l1_ratio'])
                score_matrix[l1_idx, alpha_idx] = -self.cv_results['mean_test_score'][i]  # Convert back to positive RMSE
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(score_matrix, 
                       xticklabels=[f'{a:.0e}' for a in alphas],
                       yticklabels=[f'{r:.1f}' for r in l1_ratios],
                       annot=True, fmt='.4f', cmap='viridis_r')
            
            plt.xlabel('Alpha (Regularization Strength)')
            plt.ylabel('L1 Ratio')
            plt.title('Elastic Net Hyperparameter Tuning Results\\n(RMSE - Lower is Better)')
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.plots_dir, 'hyperparameter_heatmap.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved hyperparameter heatmap: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Could not create hyperparameter heatmap: {e}")
    
    def run_complete_training(self) -> bool:
        """Run the complete training pipeline"""
        
        print(f"🚀 STARTING COMPLETE ELASTIC NET TRAINING")
        print("="*70)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        if not self.load_training_data():
            print(f"❌ Failed to load training data")
            return False
        
        # Step 2: Hyperparameter tuning
        if not self.tune_hyperparameters():
            print(f"❌ Failed hyperparameter tuning")
            return False
        
        # Step 3: Train final model
        if not self.train_final_model():
            print(f"❌ Failed to train final model")
            return False
        
        # Step 4: Save everything
        if not self.save_model_and_results():
            print(f"❌ Failed to save model and results")
            return False
        
        print(f"\\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"🤖 Model ready for testing and predictions")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True

def main():
    """Main execution function"""
    
    # Create trainer
    trainer = ElasticNetTrainer('results')
    
    # Run complete training
    success = trainer.run_complete_training()
    
    if success:
        print(f"\\n✅ SUCCESS: Elastic Net model training completed!")
        print(f"📋 Next steps:")
        print(f"  1. Review training results in monthly_model_training/results/")
        print(f"  2. Check feature importance plots in monthly_model_training/plots/")
        print(f"  3. Confirm model performance before proceeding to testing")
        print(f"  4. Ready for out-of-sample predictions on 2017-2023 data")
    else:
        print(f"\\n❌ FAILED: Training encountered errors")

if __name__ == "__main__":
    main()
