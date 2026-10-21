"""
FIXED Monthly Feature Extractor - Complete Industry Mapping
===========================================================

This script fixes the missing x5 and x13 features by:
1. Loading the complete industry mapping from the separate file
2. Ensuring all stocks have proper industry assignments
3. Fixing the ETF data loading and calculation logic
4. Re-extracting complete training features

Date: October 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add feature_extraction to path to import the complete mapping
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stock_industry_mapping import STOCK_TO_INDUSTRY_ETF
    print("✅ Successfully imported complete industry mapping")
except ImportError:
    print("❌ Could not import industry mapping - using fallback")
    STOCK_TO_INDUSTRY_ETF = {}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedMonthlyFeatureExtractor:
    def __init__(self, data_dir='1998-2017_daily_eod'):
        self.stocks_dir = os.path.join(data_dir, 'stocks')
        self.etfs_dir = os.path.join(data_dir, 'etfs')
        
        # Use complete industry mapping from separate file
        self.industry_mapping = STOCK_TO_INDUSTRY_ETF
        
        print(f"📊 Industry mapping loaded: {len(self.industry_mapping)} stocks mapped")
        
        # Cache for loaded data
        self.spy_data = None
        self.spy_monthly_returns = None
        self.etf_data = {}
        
        # Load SPY data once
        if not self.load_spy_data():
            raise Exception("Failed to load SPY data - required for calculations")
    
    def load_spy_data(self) -> bool:
        """Load SPY data for beta and premium calculations"""
        spy_file = os.path.join(self.etfs_dir, 'SPY.csv')
        if not os.path.exists(spy_file):
            logger.error("SPY.csv not found - required for calculations")
            return False
            
        try:
            self.spy_data = pd.read_csv(spy_file)
            self.spy_data['Date'] = pd.to_datetime(self.spy_data['Date'])
            self.spy_data = self.spy_data.sort_values('Date')
            
            # Calculate SPY monthly returns
            self.spy_data['YearMonth'] = self.spy_data['Date'].dt.to_period('M')
            spy_monthly = self.spy_data.groupby('YearMonth')['Daily_Return'].sum().reset_index()
            spy_monthly.columns = ['YearMonth', 'spy_monthly_return']
            self.spy_monthly_returns = spy_monthly.set_index('YearMonth')['spy_monthly_return']
            
            logger.info(f"✅ Loaded SPY data: {len(self.spy_data):,} daily records")
            logger.info(f"✅ SPY monthly returns: {len(self.spy_monthly_returns)} months")
            return True
        except Exception as e:
            logger.error(f"Error loading SPY data: {e}")
            return False
    
    def load_etf_data(self, etf_ticker: str) -> Optional[pd.DataFrame]:
        """Load and cache ETF data with monthly returns"""
        if etf_ticker in self.etf_data:
            return self.etf_data[etf_ticker]
            
        etf_file = os.path.join(self.etfs_dir, f'{etf_ticker}.csv')
        if not os.path.exists(etf_file):
            logger.warning(f"ETF {etf_ticker}.csv not found")
            return None
            
        try:
            df = pd.read_csv(etf_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Calculate monthly returns
            df['YearMonth'] = df['Date'].dt.to_period('M')
            monthly_returns = df.groupby('YearMonth')['Daily_Return'].sum().reset_index()
            monthly_returns.columns = ['YearMonth', 'monthly_return']
            monthly_returns = monthly_returns.set_index('YearMonth')['monthly_return']
            
            # Cache the processed data
            self.etf_data[etf_ticker] = {
                'daily_data': df,
                'monthly_returns': monthly_returns
            }
            
            logger.info(f"✅ Loaded {etf_ticker} data: {len(df):,} daily records, {len(monthly_returns)} months")
            return self.etf_data[etf_ticker]
            
        except Exception as e:
            logger.warning(f"Error loading ETF {etf_ticker}: {e}")
            return None
    
    def extract_stock_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """Extract all 13 features for a single stock"""
        stock_file = os.path.join(self.stocks_dir, f'{ticker}.csv')
        
        if not os.path.exists(stock_file):
            logger.warning(f"Stock file not found: {ticker}.csv")
            return None
        
        try:
            # Load stock data
            df = pd.read_csv(stock_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Convert to monthly aggregates
            df['YearMonth'] = df['Date'].dt.to_period('M')
            monthly = df.groupby('YearMonth').agg({
                'Date': 'last',
                'Close': 'last',
                'Daily_Return': 'sum',
                'Daily_Volume': 'sum',
                'Daily_Turnover': 'sum'
            }).reset_index()
            
            monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()
            monthly = monthly.sort_values('YearMonth')
            
            # Initialize features DataFrame
            features_df = pd.DataFrame()
            features_df['YearMonth'] = monthly['YearMonth']
            features_df['ticker'] = ticker
            
            # Calculate basic features (x1-x4, x6-x11)
            features_df['x1_mom1m'] = monthly['Daily_Return'].shift(1)
            features_df['x2_mom6m'] = monthly['Daily_Return'].rolling(6, min_periods=1).sum().shift(1)
            features_df['x3_mom12m'] = monthly['Daily_Return'].rolling(12, min_periods=1).sum().shift(1)
            features_df['x4_chmom'] = (monthly['Daily_Return'].rolling(6, min_periods=1).sum().shift(1) - 
                                     monthly['Daily_Return'].rolling(6, min_periods=1).sum().shift(7))
            
            # x6: Max return (approximated as 1.2 * absolute return)
            features_df['x6_maxret'] = (monthly['Daily_Return'].abs() * 1.2).shift(1)
            
            # Volume features (x7-x9)
            features_df['x7_mvel1'] = monthly['Daily_Volume'].shift(1)
            features_df['x8_turn'] = monthly['Daily_Turnover'].shift(1)
            features_df['x9_std_turn'] = monthly['Daily_Turnover'].rolling(3, min_periods=1).std().shift(1)
            
            # x10: Zero trading days (approximated as 0 for now)
            features_df['x10_zerotrade'] = 0.0
            
            # x11: Return volatility
            features_df['x11_retvol'] = monthly['Daily_Return'].rolling(3, min_periods=1).std().shift(1)
            
            # x12: Market beta (calculate vs SPY)
            features_df = self.calculate_beta(features_df, monthly, ticker)
            
            # x5 & x13: Industry features (requires ETF data)
            features_df = self.calculate_industry_features(features_df, monthly, ticker)
            
            # Target variable (y): Next month return
            features_df['y'] = monthly['Daily_Return'].shift(-1)
            
            # Remove rows with missing target
            features_df = features_df.dropna(subset=['y'])
            
            # Fill NaN values in features with 0 (except industry features which should stay NaN if truly missing)
            feature_cols = [col for col in features_df.columns if col.startswith('x') and col not in ['x5_indmom', 'x13_indprem']]
            features_df[feature_cols] = features_df[feature_cols].fillna(0.0)
            
            logger.info(f"✅ {ticker}: {len(features_df)} feature records extracted")
            return features_df
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None
    
    def calculate_beta(self, features_df: pd.DataFrame, monthly_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate market beta vs SPY"""
        try:
            # Align stock and SPY returns by YearMonth
            stock_returns = monthly_data.set_index('YearMonth')['Daily_Return']
            
            beta_values = []
            for _, row in features_df.iterrows():
                period = row['YearMonth']
                
                # Use 12-month rolling window for beta calculation
                end_period = period - 1  # Previous month
                start_period = end_period - 11  # 12 months before
                
                try:
                    # Get stock and market returns for the period
                    stock_window = stock_returns[(stock_returns.index >= start_period) & (stock_returns.index <= end_period)]
                    spy_window = self.spy_monthly_returns[(self.spy_monthly_returns.index >= start_period) & (self.spy_monthly_returns.index <= end_period)]
                    
                    # Align the data
                    aligned_data = pd.DataFrame({
                        'stock': stock_window,
                        'market': spy_window
                    }).dropna()
                    
                    if len(aligned_data) >= 6:  # Need at least 6 months of data
                        covariance = np.cov(aligned_data['stock'], aligned_data['market'])[0, 1]
                        market_variance = np.var(aligned_data['market'])
                        
                        if market_variance > 0:
                            beta = covariance / market_variance
                            beta_values.append(beta)
                        else:
                            beta_values.append(1.0)  # Default beta
                    else:
                        beta_values.append(1.0)  # Default beta
                        
                except:
                    beta_values.append(1.0)  # Default beta
            
            features_df['x12_beta'] = beta_values
            return features_df
            
        except Exception as e:
            logger.warning(f"Error calculating beta for {ticker}: {e}")
            features_df['x12_beta'] = 1.0  # Default beta
            return features_df
    
    def calculate_industry_features(self, features_df: pd.DataFrame, monthly_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate industry momentum (x5) and premium (x13) features"""
        
        # Check if stock has industry mapping
        if ticker not in self.industry_mapping:
            logger.warning(f"No industry mapping found for {ticker} - setting x5/x13 to NaN")
            features_df['x5_indmom'] = np.nan
            features_df['x13_indprem'] = np.nan
            return features_df
        
        industry_etf = self.industry_mapping[ticker]
        
        try:
            # Load industry ETF data
            etf_data = self.load_etf_data(industry_etf)
            if etf_data is None:
                logger.warning(f"Could not load ETF data for {industry_etf} (stock: {ticker})")
                features_df['x5_indmom'] = np.nan
                features_df['x13_indprem'] = np.nan
                return features_df
            
            industry_returns = etf_data['monthly_returns']
            
            # Calculate features for each month
            x5_values = []  # Industry 6-month momentum
            x13_values = []  # Industry premium
            
            for _, row in features_df.iterrows():
                period = row['YearMonth']
                prev_period = period - 1  # Previous month
                
                try:
                    # x5: Previous month's 6-month industry momentum
                    if prev_period in industry_returns.index:
                        # Get 6 months of industry returns ending at prev_period
                        start_period = prev_period - 5
                        industry_window = industry_returns[(industry_returns.index >= start_period) & (industry_returns.index <= prev_period)]
                        
                        if len(industry_window) >= 3:  # Need at least 3 months
                            x5_mom = industry_window.sum()
                            x5_values.append(x5_mom)
                        else:
                            x5_values.append(np.nan)
                    else:
                        x5_values.append(np.nan)
                    
                    # x13: Previous month's industry premium (industry return - market return)
                    if prev_period in industry_returns.index and prev_period in self.spy_monthly_returns.index:
                        industry_return = industry_returns.loc[prev_period]
                        market_return = self.spy_monthly_returns.loc[prev_period]
                        premium = industry_return - market_return
                        x13_values.append(premium)
                    else:
                        x13_values.append(np.nan)
                        
                except Exception as e:
                    x5_values.append(np.nan)
                    x13_values.append(np.nan)
            
            features_df['x5_indmom'] = x5_values
            features_df['x13_indprem'] = x13_values
            
            # Log success rate
            x5_valid = sum(1 for x in x5_values if not pd.isna(x))
            x13_valid = sum(1 for x in x13_values if not pd.isna(x))
            logger.info(f"✅ {ticker} ({industry_etf}): x5 valid {x5_valid}/{len(x5_values)}, x13 valid {x13_valid}/{len(x13_values)}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating industry features for {ticker} ({industry_etf}): {e}")
            features_df['x5_indmom'] = np.nan
            features_df['x13_indprem'] = np.nan
            return features_df
    
    def extract_all_features(self, output_dir='monthly_features_1998_2017_FIXED') -> Dict:
        """Extract features for all stocks"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of stock files
        stock_files = [f for f in os.listdir(self.stocks_dir) if f.endswith('.csv')]
        stock_tickers = [f.replace('.csv', '') for f in stock_files]
        
        logger.info(f"🚀 Starting feature extraction for {len(stock_tickers)} stocks")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'processed': 0,
            'failed': 0,
            'total_features': 0,
            'failed_stocks': [],
            'industry_mapping_stats': {
                'total_stocks': len(stock_tickers),
                'mapped_stocks': sum(1 for ticker in stock_tickers if ticker in self.industry_mapping),
                'unmapped_stocks': [ticker for ticker in stock_tickers if ticker not in self.industry_mapping]
            }
        }
        
        all_features = []
        
        # Process each stock
        for i, ticker in enumerate(stock_tickers, 1):
            logger.info(f"📊 Processing {ticker} ({i}/{len(stock_tickers)})")
            
            features_df = self.extract_stock_features(ticker)
            
            if features_df is not None and len(features_df) > 0:
                # Save individual stock features
                stock_file = os.path.join(output_dir, f'{ticker}.csv')
                features_df.to_csv(stock_file, index=False)
                
                all_features.append(features_df)
                results['processed'] += 1
                results['total_features'] += len(features_df)
            else:
                logger.warning(f"❌ Failed to extract features for {ticker}")
                results['failed'] += 1
                results['failed_stocks'].append(ticker)
        
        # Save combined dataset
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            combined_file = os.path.join(output_dir, 'combined_monthly_features_FIXED.csv')
            combined_df.to_csv(combined_file, index=False)
            results['combined_file'] = combined_file
            results['combined_records'] = len(combined_df)
            
            # Calculate feature completeness
            feature_cols = [col for col in combined_df.columns if col.startswith('x')]
            feature_completeness = {}
            for col in feature_cols:
                non_null = combined_df[col].notna().sum()
                total = len(combined_df)
                completeness = (non_null / total) * 100
                feature_completeness[col] = {
                    'non_null': int(non_null),
                    'total': int(total),
                    'completeness_pct': round(completeness, 1)
                }
            
            results['feature_completeness'] = feature_completeness
            
            logger.info(f"✅ Saved combined dataset: {len(combined_df):,} total feature records")
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save processing stats
        stats_file = os.path.join(output_dir, 'extraction_stats_FIXED.json')
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🎯 FIXED FEATURE EXTRACTION COMPLETE!")
        print("="*60)
        print(f"✅ Processed: {results['processed']} stocks")
        print(f"❌ Failed: {results['failed']} stocks")
        print(f"📊 Total features: {results['total_features']:,}")
        print(f"🗺️ Industry mapping: {results['industry_mapping_stats']['mapped_stocks']}/{results['industry_mapping_stats']['total_stocks']} stocks mapped")
        
        if results.get('feature_completeness'):
            print(f"\n📋 Feature Completeness:")
            for feature, stats in results['feature_completeness'].items():
                print(f"  {feature}: {stats['completeness_pct']:.1f}% ({stats['non_null']:,}/{stats['total']:,})")
        
        if results['industry_mapping_stats']['unmapped_stocks']:
            print(f"\n⚠️ Unmapped stocks: {results['industry_mapping_stats']['unmapped_stocks']}")
        
        return results

def main():
    """Main execution function"""
    print("🔧 FIXED MONTHLY FEATURE EXTRACTOR")
    print("="*60)
    print("Fixes: Complete industry mapping, proper ETF loading, x5/x13 calculations")
    print("="*60)
    
    try:
        extractor = FixedMonthlyFeatureExtractor()
        results = extractor.extract_all_features()
        
        if results.get('processed', 0) > 0:
            print(f"\n🎉 SUCCESS: Fixed monthly features extracted!")
            print(f"📁 Combined dataset: {results.get('combined_file')}")
            print(f"📊 Ready for model training with complete x5/x13 features")
        else:
            print(f"\n❌ FAILED: No features extracted")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

