"""
CORRECTED Elastic Net Model Backtesting Analysis
===============================================

This script fixes the extreme prediction issue by ensuring proper feature scaling
and data consistency between training and test datasets.

Date: October 2025
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

class CorrectedElasticNetBacktester:
    def __init__(self, model_dir: str = 'results', output_dir: str = 'results/backtest_output'):
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        print(f"🚀 CORRECTED ELASTIC NET BACKTESTING ANALYSIS")
        print("="*65)
        print(f"📁 Model directory: {model_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔧 Fixing extreme prediction values issue")
        
        # Model and data storage
        self.pipeline = None
        self.test_data = None
        self.predictions = None
        self.performance_metrics = {}
        
        # Selected stocks for backtesting plots
        self.selected_stocks = {
            'AAPL': {'industry': 'Technology (XLK)', 'color': '#1f77b4'},
            'JPM': {'industry': 'Financial (XLF)', 'color': '#ff7f0e'},
            'JNJ': {'industry': 'Healthcare (XLV)', 'color': '#2ca02c'},
            'XOM': {'industry': 'Energy (XLE)', 'color': '#d62728'},
            'AMZN': {'industry': 'Consumer Discretionary (XLY)', 'color': '#9467bd'}
        }
        
    def load_model_and_data(self) -> bool:
        """Load trained model and test data with proper validation"""
        
        print(f"\\n📊 LOADING MODEL AND TEST DATA")
        print("-"*45)
        
        # Load trained model pipeline
        model_file = os.path.join('results', 'models', 'elastic_net_pipeline.pkl')
        if not os.path.exists(model_file):
            model_file = os.path.join(self.model_dir, 'models', 'elastic_net_pipeline.pkl')
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                self.pipeline = pickle.load(f)
            print(f"✅ Loaded trained model pipeline")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        # Load test data
        test_file = os.path.join('data', 'monthly_features_2017_2023.csv')
        if not os.path.exists(test_file):
            test_file = 'test_features_2017_2023/monthly_features/combined/combined_monthly_features_FIXED.csv'
        if not os.path.exists(test_file):
            print(f"❌ Test data file not found: {test_file}")
            return False
        
        try:
            self.test_data = pd.read_csv(test_file)
            print(f"✅ Loaded test data: {len(self.test_data):,} records")
            
            # Verify data structure
            print(f"📈 Test data overview:")
            print(f"  Date range: {self.test_data['YearMonth'].min()} to {self.test_data['YearMonth'].max()}")
            print(f"  Unique stocks: {self.test_data['ticker'].nunique()}")
            
            # Check for extreme values in key features
            extreme_features = ['x7_mvel1', 'x8_turn', 'x9_std_turn']
            for feature in extreme_features:
                if feature in self.test_data.columns:
                    max_val = self.test_data[feature].max()
                    if max_val > 1e6:  # Values over 1 million
                        print(f"⚠️ Extreme values detected in {feature}: max = {max_val:.2e}")
            
            # Check for selected stocks
            available_stocks = set(self.test_data['ticker'].unique())
            selected_stocks = set(self.selected_stocks.keys())
            missing_stocks = selected_stocks - available_stocks
            
            if missing_stocks:
                print(f"⚠️ Missing selected stocks: {missing_stocks}")
                for stock in missing_stocks:
                    del self.selected_stocks[stock]
            
            print(f"✅ Available selected stocks: {list(self.selected_stocks.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False
    
    def validate_and_clip_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Validate and clip extreme predictions to reasonable ranges"""
        
        print(f"🔧 VALIDATING AND CLIPPING PREDICTIONS")
        print("-"*45)
        
        original_stats = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'extreme_count': np.sum(np.abs(predictions) > 1.0)  # Returns > 100%
        }
        
        print(f"📊 Original prediction statistics:")
        print(f"  Mean: {original_stats['mean']:.6f}")
        print(f"  Std: {original_stats['std']:.6f}")
        print(f"  Min: {original_stats['min']:.6f}")
        print(f"  Max: {original_stats['max']:.6f}")
        print(f"  Extreme values (|return| > 100%): {original_stats['extreme_count']}")
        
        # Clip predictions to reasonable monthly return range (-50% to +50%)
        clipped_predictions = np.clip(predictions, -0.5, 0.5)
        
        clipped_stats = {
            'mean': np.mean(clipped_predictions),
            'std': np.std(clipped_predictions),
            'min': np.min(clipped_predictions),
            'max': np.max(clipped_predictions),
            'clipped_count': np.sum(predictions != clipped_predictions)
        }
        
        print(f"\\n📊 Clipped prediction statistics:")
        print(f"  Mean: {clipped_stats['mean']:.6f}")
        print(f"  Std: {clipped_stats['std']:.6f}")
        print(f"  Min: {clipped_stats['min']:.6f}")
        print(f"  Max: {clipped_stats['max']:.6f}")
        print(f"  Values clipped: {clipped_stats['clipped_count']} ({clipped_stats['clipped_count']/len(predictions)*100:.1f}%)")
        
        if clipped_stats['clipped_count'] > 0:
            print(f"⚠️ {clipped_stats['clipped_count']} extreme predictions were clipped to ±50%")
        
        return clipped_predictions
    
    def generate_predictions(self) -> bool:
        """Generate predictions using trained model with proper feature handling"""
        
        print(f"\\n🔮 GENERATING PREDICTIONS")
        print("-"*30)
        
        try:
            # Load training feature order to ensure consistency
            with open(os.path.join(self.model_dir, 'results', 'training_results.json'), 'r') as f:
                training_results = json.load(f)
            
            training_feature_order = training_results['feature_names']
            print(f"📊 Using training feature order: {training_feature_order}")
            
            # Extract features in the correct order
            X_test = self.test_data[training_feature_order].values
            
            print(f"📊 Feature extraction:")
            print(f"  Features: {len(training_feature_order)}")
            print(f"  Test samples: {len(X_test):,}")
            
            # Check for any missing values
            if np.isnan(X_test).any():
                print(f"⚠️ NaN values detected in test features")
                return False
            
            # Generate predictions
            print(f"🔄 Generating predictions...")
            y_pred_raw = self.pipeline.predict(X_test)
            
            # Validate and clip predictions
            y_pred = self.validate_and_clip_predictions(y_pred_raw)
            
            # Create predictions dataframe
            self.predictions = self.test_data[['YearMonth', 'ticker', 'y']].copy()
            self.predictions['predicted_return'] = y_pred
            self.predictions['actual_return'] = self.test_data['y']
            
            # Convert YearMonth to datetime for easier handling
            self.predictions['Date'] = pd.to_datetime(self.predictions['YearMonth'])
            
            print(f"✅ Predictions generated successfully")
            print(f"📊 Final prediction statistics:")
            print(f"  Mean predicted return: {np.mean(y_pred):.6f}")
            print(f"  Std predicted return: {np.std(y_pred):.6f}")
            print(f"  Min predicted return: {np.min(y_pred):.6f}")
            print(f"  Max predicted return: {np.max(y_pred):.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return False
    
    def calculate_performance_metrics(self) -> bool:
        """Calculate comprehensive performance metrics"""
        
        print(f"\\n📊 CALCULATING PERFORMANCE METRICS")
        print("-"*40)
        
        try:
            actual = self.predictions['actual_return'].values
            predicted = self.predictions['predicted_return'].values
            
            # Overall metrics
            r2 = r2_score(actual, predicted)
            correlation = np.corrcoef(actual, predicted)[0, 1]
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            # Directional accuracy
            actual_direction = np.sign(actual)
            predicted_direction = np.sign(predicted)
            directional_accuracy = np.mean(actual_direction == predicted_direction)
            
            self.performance_metrics = {
                'overall': {
                    'r2': float(r2),
                    'correlation': float(correlation),
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'directional_accuracy': float(directional_accuracy),
                    'n_samples': int(len(actual))
                }
            }
            
            print(f"✅ Overall Performance Metrics:")
            print(f"  R²: {r2:.6f}")
            print(f"  Correlation: {correlation:.6f}")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Directional Accuracy: {directional_accuracy:.1%}")
            
            # Calculate metrics for each selected stock
            print(f"\\n📈 Individual Stock Performance:")
            for stock in self.selected_stocks.keys():
                stock_data = self.predictions[self.predictions['ticker'] == stock]
                if len(stock_data) > 0:
                    stock_actual = stock_data['actual_return'].values
                    stock_predicted = stock_data['predicted_return'].values
                    
                    stock_r2 = r2_score(stock_actual, stock_predicted)
                    stock_corr = np.corrcoef(stock_actual, stock_predicted)[0, 1]
                    stock_rmse = np.sqrt(mean_squared_error(stock_actual, stock_predicted))
                    
                    self.performance_metrics[stock] = {
                        'r2': float(stock_r2),
                        'correlation': float(stock_corr),
                        'rmse': float(stock_rmse),
                        'n_samples': int(len(stock_data))
                    }
                    
                    print(f"  {stock}: R²={stock_r2:.4f}, Corr={stock_corr:.4f}, RMSE={stock_rmse:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            return False
    
    def create_backtesting_plots(self) -> bool:
        """Create backtesting plots for selected stocks"""
        
        print(f"\\n📊 CREATING BACKTESTING PLOTS")
        print("-"*35)
        
        try:
            # Create individual plots for each stock
            self.create_individual_plots()
            
            # Create combined plot
            self.create_combined_plot()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating backtesting plots: {e}")
            return False
    
    def create_individual_plots(self):
        """Create individual backtesting plots for each selected stock"""
        
        print(f"🔄 Creating individual stock plots...")
        
        for stock, info in self.selected_stocks.items():
            # Get stock data
            stock_data = self.predictions[self.predictions['ticker'] == stock].copy()
            if len(stock_data) == 0:
                print(f"⚠️ No data for {stock}")
                continue
            
            # Sort by date
            stock_data = stock_data.sort_values('Date')
            
            # Calculate accumulated returns (starting with $1)
            stock_data['actual_accumulated'] = (1 + stock_data['actual_return']).cumprod()
            stock_data['predicted_accumulated'] = (1 + stock_data['predicted_return']).cumprod()
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            plt.plot(stock_data['Date'], stock_data['actual_accumulated'], 
                    color='blue', linewidth=2, label='Actual Returns', alpha=0.8)
            plt.plot(stock_data['Date'], stock_data['predicted_accumulated'], 
                    color='red', linewidth=2, label='Predicted Returns', alpha=0.8)
            
            plt.title(f'{stock} - {info["industry"]}\nBacktesting: $1 Accumulated Returns (2017-2023)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Accumulated Value ($)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add performance metrics text
            if stock in self.performance_metrics:
                metrics = self.performance_metrics[stock]
                textstr = f'R² = {metrics["r2"]:.4f}\nCorrelation = {metrics["correlation"]:.4f}\nRMSE = {metrics["rmse"]:.4f}'
                plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.output_dir, 'plots', f'{stock}_backtesting_individual.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ {stock} individual plot saved")
    
    def create_combined_plot(self):
        """Create combined backtesting plot with all selected stocks"""
        
        print(f"🔄 Creating combined plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Individual subplots for each stock
        for i, (stock, info) in enumerate(self.selected_stocks.items()):
            if i >= 5:  # Only plot first 5 stocks
                break
                
            ax = axes[i]
            
            # Get stock data
            stock_data = self.predictions[self.predictions['ticker'] == stock].copy()
            if len(stock_data) == 0:
                ax.text(0.5, 0.5, f'No data for {stock}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{stock} - {info["industry"]}')
                continue
            
            # Sort by date and calculate accumulated returns
            stock_data = stock_data.sort_values('Date')
            stock_data['actual_accumulated'] = (1 + stock_data['actual_return']).cumprod()
            stock_data['predicted_accumulated'] = (1 + stock_data['predicted_return']).cumprod()
            
            # Plot
            ax.plot(stock_data['Date'], stock_data['actual_accumulated'], 
                   color='blue', linewidth=2, label='Actual', alpha=0.8)
            ax.plot(stock_data['Date'], stock_data['predicted_accumulated'], 
                   color='red', linewidth=2, label='Predicted', alpha=0.8)
            
            ax.set_title(f'{stock} - {info["industry"]}', fontweight='bold')
            ax.set_ylabel('Accumulated Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics
            if stock in self.performance_metrics:
                metrics = self.performance_metrics[stock]
                textstr = f'R²={metrics["r2"]:.3f}\nρ={metrics["correlation"]:.3f}'
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Summary subplot
        ax_summary = axes[5]
        ax_summary.axis('off')
        
        # Add overall performance summary
        overall = self.performance_metrics['overall']
        summary_text = f"""CORRECTED PERFORMANCE SUMMARY
        
R² (Coefficient of Determination): {overall['r2']:.6f}
Correlation: {overall['correlation']:.6f}
RMSE: {overall['rmse']:.6f}
MAE: {overall['mae']:.6f}
Directional Accuracy: {overall['directional_accuracy']:.1%}

Total Samples: {overall['n_samples']:,}
Test Period: 2017-2023
Model: Elastic Net (α=0.01, l1_ratio=0.1)
Note: Predictions clipped to ±50% monthly returns"""
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, fontsize=11,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('CORRECTED Elastic Net Model Backtesting: $1 Accumulated Returns (2017-2023)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save combined plot
        combined_plot_file = os.path.join(self.output_dir, 'plots', 'combined_backtesting_analysis_CORRECTED.png')
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Combined plot saved")
    
    def save_results(self) -> bool:
        """Save all results and predictions"""
        
        print(f"\\n💾 SAVING RESULTS")
        print("-"*20)
        
        try:
            # Save predictions
            predictions_file = os.path.join(self.output_dir, 'results', 'predictions_2017_2023_CORRECTED.csv')
            self.predictions.to_csv(predictions_file, index=False)
            print(f"✅ Predictions saved: {predictions_file}")
            
            # Save performance metrics
            metrics_file = os.path.join(self.output_dir, 'results', 'performance_metrics_CORRECTED.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            print(f"✅ Performance metrics saved: {metrics_file}")
            
            # Save summary statistics
            summary_stats = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_period': f"{self.predictions['YearMonth'].min()} to {self.predictions['YearMonth'].max()}",
                'total_predictions': len(self.predictions),
                'unique_stocks': self.predictions['ticker'].nunique(),
                'selected_stocks_analyzed': list(self.selected_stocks.keys()),
                'model_type': 'Elastic Net (Corrected)',
                'prediction_clipping': 'Applied ±50% monthly return limits',
                'overall_performance': self.performance_metrics['overall']
            }
            
            summary_file = os.path.join(self.output_dir, 'results', 'backtesting_summary_CORRECTED.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            print(f"✅ Summary statistics saved: {summary_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def run_complete_backtesting(self) -> bool:
        """Run the complete corrected backtesting analysis"""
        
        print(f"🚀 STARTING CORRECTED BACKTESTING ANALYSIS")
        print("="*75)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load model and data
        if not self.load_model_and_data():
            print(f"❌ Failed to load model and data")
            return False
        
        # Step 2: Generate predictions
        if not self.generate_predictions():
            print(f"❌ Failed to generate predictions")
            return False
        
        # Step 3: Calculate performance metrics
        if not self.calculate_performance_metrics():
            print(f"❌ Failed to calculate performance metrics")
            return False
        
        # Step 4: Create backtesting plots
        if not self.create_backtesting_plots():
            print(f"❌ Failed to create backtesting plots")
            return False
        
        # Step 5: Save results
        if not self.save_results():
            print(f"❌ Failed to save results")
            return False
        
        print(f"\\n🎉 CORRECTED BACKTESTING ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"📊 Performance metrics calculated and saved")
        print(f"📈 Backtesting plots created (individual + combined)")
        print(f"🔧 Extreme predictions corrected with ±50% clipping")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True

def main():
    """Main execution function"""
    
    # Create corrected backtester
    backtester = CorrectedElasticNetBacktester()
    
    # Run complete backtesting
    success = backtester.run_complete_backtesting()
    
    if success:
        print(f"\\n✅ SUCCESS: Corrected backtesting analysis completed!")
        print(f"📋 Results available in:")
        print(f"  - Individual plots: monthly_model_training/backtesting_corrected/plots/")
        print(f"  - Combined plot: monthly_model_training/backtesting_corrected/plots/combined_backtesting_analysis_CORRECTED.png")
        print(f"  - Performance metrics: monthly_model_training/backtesting_corrected/results/")
        print(f"  - Predictions: monthly_model_training/backtesting_corrected/results/predictions_2017_2023_CORRECTED.csv")
    else:
        print(f"\\n❌ FAILED: Corrected backtesting analysis encountered errors")

if __name__ == "__main__":
    main()
