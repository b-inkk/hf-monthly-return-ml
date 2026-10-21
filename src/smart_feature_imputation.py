"""
Smart Imputation for Missing x5/x13 Features in Test Data
=========================================================

This script implements intelligent imputation for missing x5 (industry momentum) 
and x13 (industry premium) features in the test period (2017-2023).

Strategy:
1. Use available ETF data (2017-2020) to establish relationships
2. For 2021-2023, impute using market proxies and historical patterns
3. Maintain feature distributions similar to training period
4. Add imputation flags for transparency

Date: October 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartFeatureImputer:
    def __init__(self, test_features_file: str, output_file: str = None):
        self.test_features_file = test_features_file
        self.output_file = output_file or test_features_file.replace('.csv', '_IMPUTED.csv')
        
        # Load data
        self.df = pd.read_csv(test_features_file)
        self.df['YearMonth'] = pd.to_datetime(self.df['YearMonth'])
        
        print(f"🔧 SMART FEATURE IMPUTER")
        print(f"📊 Loaded: {len(self.df):,} records, {self.df['ticker'].nunique()} stocks")
        
        # Load industry mapping
        from feature_extraction.stock_industry_mapping_updated import STOCK_TO_INDUSTRY_ETF
        self.industry_mapping = STOCK_TO_INDUSTRY_ETF
        
    def calculate_market_proxy_relationships(self) -> Dict:
        """Calculate relationships between market and industry features for available data"""
        
        print(f"\n📈 CALCULATING MARKET-INDUSTRY RELATIONSHIPS")
        print("-"*55)
        
        # Filter to periods with good x5/x13 data (2017-2020)
        available_data = self.df[
            (self.df['YearMonth'] >= '2017-01') & 
            (self.df['YearMonth'] <= '2020-12') &
            self.df['x5_indmom'].notna() & 
            self.df['x13_indprem'].notna()
        ].copy()
        
        print(f"✅ Available data for relationship modeling: {len(available_data):,} records")
        
        relationships = {}
        
        # Calculate market momentum (using x2_mom6m as market proxy)
        market_momentum = available_data.groupby('YearMonth')['x2_mom6m'].mean()
        
        # For each industry, calculate relationship with market
        available_data['industry_etf'] = available_data['ticker'].map(self.industry_mapping)
        
        for industry in available_data['industry_etf'].unique():
            if pd.isna(industry):
                continue
                
            industry_data = available_data[available_data['industry_etf'] == industry]
            
            if len(industry_data) < 10:  # Need minimum data points
                continue
            
            # Calculate average industry features by month
            monthly_industry = industry_data.groupby('YearMonth').agg({
                'x5_indmom': 'mean',
                'x13_indprem': 'mean',
                'x2_mom6m': 'mean'  # Market proxy
            })
            
            # Calculate correlations and scaling factors
            if len(monthly_industry) >= 5:
                # x5 relationship with market momentum
                x5_corr = monthly_industry['x5_indmom'].corr(monthly_industry['x2_mom6m'])
                x5_scale = monthly_industry['x5_indmom'].std() / monthly_industry['x2_mom6m'].std()
                x5_mean = monthly_industry['x5_indmom'].mean()
                
                # x13 historical average and volatility
                x13_mean = monthly_industry['x13_indprem'].mean()
                x13_std = monthly_industry['x13_indprem'].std()
                
                relationships[industry] = {
                    'x5_corr': x5_corr if not pd.isna(x5_corr) else 0.3,
                    'x5_scale': x5_scale if not pd.isna(x5_scale) else 1.0,
                    'x5_mean': x5_mean if not pd.isna(x5_mean) else 0.0,
                    'x13_mean': x13_mean if not pd.isna(x13_mean) else 0.0,
                    'x13_std': x13_std if not pd.isna(x13_std) else 0.02,
                    'data_points': len(monthly_industry)
                }
                
                print(f"  {industry}: x5_corr={x5_corr:.3f}, x5_scale={x5_scale:.3f}, data_points={len(monthly_industry)}")
        
        print(f"✅ Calculated relationships for {len(relationships)} industries")
        return relationships
    
    def impute_missing_features(self, relationships: Dict) -> pd.DataFrame:
        """Impute missing x5 and x13 features using calculated relationships"""
        
        print(f"\n🔧 IMPUTING MISSING FEATURES")
        print("-"*40)
        
        df_imputed = self.df.copy()
        
        # Add imputation flags
        df_imputed['x5_imputed'] = df_imputed['x5_indmom'].isna()
        df_imputed['x13_imputed'] = df_imputed['x13_indprem'].isna()
        
        # Count missing values before imputation
        x5_missing_before = df_imputed['x5_indmom'].isna().sum()
        x13_missing_before = df_imputed['x13_indprem'].isna().sum()
        
        print(f"📊 Missing values before imputation:")
        print(f"  x5_indmom: {x5_missing_before:,} ({x5_missing_before/len(df_imputed)*100:.1f}%)")
        print(f"  x13_indprem: {x13_missing_before:,} ({x13_missing_before/len(df_imputed)*100:.1f}%)")
        
        # Add industry mapping
        df_imputed['industry_etf'] = df_imputed['ticker'].map(self.industry_mapping)
        
        # Impute by industry
        imputed_count = 0
        
        for industry, params in relationships.items():
            industry_mask = (df_imputed['industry_etf'] == industry)
            
            # Impute x5_indmom using market momentum relationship
            x5_missing_mask = industry_mask & df_imputed['x5_indmom'].isna()
            if x5_missing_mask.sum() > 0:
                # Use market momentum (x2_mom6m) as proxy
                market_proxy = df_imputed.loc[x5_missing_mask, 'x2_mom6m']
                
                # Apply relationship: x5 = correlation * scale * market_momentum + mean_adjustment
                imputed_x5 = (params['x5_corr'] * params['x5_scale'] * market_proxy + 
                              params['x5_mean'] * 0.3)  # Partial mean reversion
                
                df_imputed.loc[x5_missing_mask, 'x5_indmom'] = imputed_x5
                imputed_count += x5_missing_mask.sum()
            
            # Impute x13_indprem using historical average with noise
            x13_missing_mask = industry_mask & df_imputed['x13_indprem'].isna()
            if x13_missing_mask.sum() > 0:
                # Use historical mean with some random variation
                np.random.seed(42)  # For reproducibility
                noise = np.random.normal(0, params['x13_std'] * 0.5, x13_missing_mask.sum())
                imputed_x13 = params['x13_mean'] + noise
                
                df_imputed.loc[x13_missing_mask, 'x13_indprem'] = imputed_x13
                imputed_count += x13_missing_mask.sum()
        
        # Handle stocks without industry mapping (use overall market averages)
        unmapped_mask = df_imputed['industry_etf'].isna()
        if unmapped_mask.sum() > 0:
            print(f"⚠️ Handling {unmapped_mask.sum()} records without industry mapping")
            
            # Use overall market averages
            overall_x5_mean = df_imputed['x5_indmom'].mean()
            overall_x13_mean = df_imputed['x13_indprem'].mean()
            
            # Impute unmapped x5
            x5_unmapped_missing = unmapped_mask & df_imputed['x5_indmom'].isna()
            if x5_unmapped_missing.sum() > 0:
                df_imputed.loc[x5_unmapped_missing, 'x5_indmom'] = overall_x5_mean
            
            # Impute unmapped x13
            x13_unmapped_missing = unmapped_mask & df_imputed['x13_indprem'].isna()
            if x13_unmapped_missing.sum() > 0:
                df_imputed.loc[x13_unmapped_missing, 'x13_indprem'] = overall_x13_mean
        
        # Final check
        x5_missing_after = df_imputed['x5_indmom'].isna().sum()
        x13_missing_after = df_imputed['x13_indprem'].isna().sum()
        
        print(f"\n✅ IMPUTATION RESULTS:")
        print(f"  x5_indmom: {x5_missing_before:,} → {x5_missing_after:,} missing")
        print(f"  x13_indprem: {x13_missing_before:,} → {x13_missing_after:,} missing")
        print(f"  Total imputed: {x5_missing_before + x13_missing_before - x5_missing_after - x13_missing_after:,}")
        
        # Remove temporary columns
        df_imputed = df_imputed.drop(['industry_etf'], axis=1)
        
        return df_imputed
    
    def validate_imputation(self, df_original: pd.DataFrame, df_imputed: pd.DataFrame):
        """Validate the quality of imputation"""
        
        print(f"\n🔍 IMPUTATION QUALITY VALIDATION")
        print("-"*45)
        
        # Compare distributions
        original_x5 = df_original['x5_indmom'].dropna()
        original_x13 = df_original['x13_indprem'].dropna()
        
        imputed_x5 = df_imputed['x5_indmom']
        imputed_x13 = df_imputed['x13_indprem']
        
        print(f"📊 Feature distributions:")
        print(f"x5_indmom:")
        print(f"  Original: mean={original_x5.mean():.4f}, std={original_x5.std():.4f}")
        print(f"  Imputed:  mean={imputed_x5.mean():.4f}, std={imputed_x5.std():.4f}")
        
        print(f"x13_indprem:")
        print(f"  Original: mean={original_x13.mean():.4f}, std={original_x13.std():.4f}")
        print(f"  Imputed:  mean={imputed_x13.mean():.4f}, std={imputed_x13.std():.4f}")
        
        # Check for extreme values
        x5_extreme = (imputed_x5.abs() > 0.5).sum()
        x13_extreme = (imputed_x13.abs() > 0.3).sum()
        
        print(f"\n⚠️ Extreme values check:")
        print(f"  x5_indmom |value| > 0.5: {x5_extreme} records")
        print(f"  x13_indprem |value| > 0.3: {x13_extreme} records")
        
        # Imputation summary by year
        df_imputed['Year'] = df_imputed['YearMonth'].dt.year
        yearly_imputation = df_imputed.groupby('Year').agg({
            'x5_imputed': 'sum',
            'x13_imputed': 'sum',
            'ticker': 'count'
        })
        yearly_imputation['x5_imputed_pct'] = yearly_imputation['x5_imputed'] / yearly_imputation['ticker'] * 100
        yearly_imputation['x13_imputed_pct'] = yearly_imputation['x13_imputed'] / yearly_imputation['ticker'] * 100
        
        print(f"\n📅 Imputation by year:")
        print(yearly_imputation[['x5_imputed_pct', 'x13_imputed_pct']].round(1).to_string())
    
    def run_imputation(self) -> str:
        """Run the complete imputation process"""
        
        print(f"🚀 STARTING SMART FEATURE IMPUTATION")
        print("="*60)
        
        # Step 1: Calculate relationships
        relationships = self.calculate_market_proxy_relationships()
        
        if not relationships:
            print("❌ No relationships calculated - cannot proceed with imputation")
            return None
        
        # Step 2: Impute missing features
        df_imputed = self.impute_missing_features(relationships)
        
        # Step 3: Validate imputation
        self.validate_imputation(self.df, df_imputed)
        
        # Step 4: Save results
        df_imputed.to_csv(self.output_file, index=False)
        
        print(f"\n🎉 IMPUTATION COMPLETED!")
        print(f"📁 Imputed features saved to: {self.output_file}")
        print(f"✅ Ready for model testing with complete features!")
        
        return self.output_file

def main():
    """Main execution function"""
    
    test_features_file = 'test_features_2017_2023/monthly_features/combined_monthly_features_FIXED.csv'
    
    if not os.path.exists(test_features_file):
        print(f"❌ Test features file not found: {test_features_file}")
        return
    
    # Create imputer and run
    imputer = SmartFeatureImputer(test_features_file)
    output_file = imputer.run_imputation()
    
    if output_file:
        print(f"\n📋 NEXT STEPS:")
        print("1. ✅ Use imputed features for model testing")
        print("2. 📊 Compare model performance with/without imputation")
        print("3. 📝 Document imputation methodology in results")
        print("4. ⚠️ Note limitations in final analysis")

if __name__ == "__main__":
    main()

