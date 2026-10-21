#!/usr/bin/env python3
"""
Efficient Daily End-of-Day Extraction for Monthly Features (1998-2017)
======================================================================

This script extracts only the LAST MINUTE of each trading day, plus daily aggregates
needed for monthly feature calculation. Reduces data from 285M records to ~950K records.

Extracts:
- Last minute of each trading day (for daily closing prices)
- Daily volume totals (for x7, x8)
- Daily turnover calculations (for x8, x9)
- Zero volume day flags (for x10)
- Daily returns (for x6, x11)

Date: September 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_eod_extraction_1998_2017.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyEODExtractor:
    def __init__(self):
        self.base_dir = Path(".")
        self.stock_source_dir = self.base_dir / "s&p100-1mindata"
        self.etf_source_dir = self.base_dir / "50etfs-1mindata"
        self.output_dir = self.base_dir / "1998-2017_daily_eod"
        self.start_date = datetime(1998, 1, 1)
        self.end_date = datetime(2017, 12, 31)
        
        # Create output directories
        self.stock_output_dir = self.output_dir / "stocks"
        self.etf_output_dir = self.output_dir / "etfs"
        self.stock_output_dir.mkdir(parents=True, exist_ok=True)
        self.etf_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'stocks_processed': 0,
            'stocks_failed': 0,
            'etfs_processed': 0,
            'etfs_failed': 0,
            'start_time': datetime.now(),
            'failed_files': [],
            'total_original_records': 0,
            'total_daily_records': 0
        }
        
    def process_file_to_daily_eod(self, file_path, is_etf=False):
        """Process file to extract daily end-of-day data with aggregates"""
        try:
            logger.info(f"Processing {file_path}")
            
            # Read the TXT file
            df = pd.read_csv(file_path, header=None, 
                           names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                return None
            
            original_records = len(df)
            self.stats['total_original_records'] += original_records
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            
            # Filter for 1998-2017 only
            mask = (df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)
            df_filtered = df[mask].copy()
            
            if df_filtered.empty:
                logger.warning(f"No data in 1998-2017 range for {file_path}")
                return None
            
            # Create DateTime column
            df_filtered['DateTime'] = pd.to_datetime(
                df_filtered['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
                df_filtered['Time'].astype(str),
                format='%Y-%m-%d %H:%M',
                errors='coerce'
            )
            
            # Remove rows with invalid datetime or prices
            df_filtered = df_filtered.dropna(subset=['DateTime'])
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
            df_filtered = df_filtered.dropna(subset=numeric_cols)
            
            # Sort by DateTime
            df_filtered = df_filtered.sort_values('DateTime')
            
            # Group by date and extract daily aggregates
            daily_data = []
            
            for date, group in df_filtered.groupby('Date'):
                if len(group) == 0:
                    continue
                
                # Get last minute of the day (end-of-day)
                eod_row = group.iloc[-1]
                
                # Calculate daily aggregates
                daily_volume = group['Volume'].sum()
                daily_turnover = (group['Volume'] * group['Close']).sum()
                daily_high = group['High'].max()
                daily_low = group['Low'].min()
                daily_open = group.iloc[0]['Open']  # First minute open
                daily_close = eod_row['Close']  # Last minute close
                
                # Daily return calculation
                if len(daily_data) > 0:  # Need previous day for return
                    prev_close = daily_data[-1]['Close']
                    daily_return = np.log(daily_close / prev_close) if prev_close > 0 else 0
                else:
                    daily_return = 0  # First day
                
                # Zero volume day flag
                zero_volume_day = 1 if daily_volume == 0 else 0
                
                # Minute count (for validation)
                minute_count = len(group)
                
                daily_record = {
                    'Date': date,
                    'DateTime': eod_row['DateTime'],
                    'Open': daily_open,
                    'High': daily_high,
                    'Low': daily_low,
                    'Close': daily_close,
                    'Volume': eod_row['Volume'],  # Last minute volume
                    'Daily_Volume': daily_volume,  # Total daily volume
                    'Daily_Turnover': daily_turnover,  # Total daily turnover
                    'Daily_Return': daily_return,  # Daily log return
                    'Zero_Volume_Day': zero_volume_day,  # Flag for zero volume
                    'Minute_Count': minute_count,  # Number of minutes traded
                    'Time': eod_row['Time']  # Last minute time
                }
                
                daily_data.append(daily_record)
            
            if not daily_data:
                logger.warning(f"No daily data created for {file_path}")
                return None
            
            # Convert to DataFrame
            daily_df = pd.DataFrame(daily_data)
            
            # Add some validation
            daily_df['Symbol'] = file_path.stem
            daily_df['Year'] = daily_df['Date'].dt.year
            daily_df['Month'] = daily_df['Date'].dt.month
            
            logger.info(f"Converted {original_records:,} records to {len(daily_df):,} daily records")
            self.stats['total_daily_records'] += len(daily_df)
            
            return daily_df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_stock_files(self):
        """Process all stock TXT files"""
        logger.info("Starting efficient stock file processing...")
        
        if not self.stock_source_dir.exists():
            logger.error(f"Stock source directory not found: {self.stock_source_dir}")
            return
        
        stock_files = list(self.stock_source_dir.glob("*.txt"))
        logger.info(f"Found {len(stock_files)} stock files to process")
        
        for file_path in stock_files:
            try:
                stock_symbol = file_path.stem
                
                # Process to daily EOD
                daily_df = self.process_file_to_daily_eod(file_path, is_etf=False)
                
                if daily_df is not None and not daily_df.empty:
                    # Save as CSV
                    output_file = self.stock_output_dir / f"{stock_symbol}.csv"
                    daily_df.to_csv(output_file, index=False)
                    
                    self.stats['stocks_processed'] += 1
                    logger.info(f"✓ Saved {stock_symbol}: {len(daily_df):,} daily records")
                else:
                    self.stats['stocks_failed'] += 1
                    self.stats['failed_files'].append(f"STOCK: {stock_symbol}")
                    logger.warning(f"❌ Failed {stock_symbol}: No valid daily data")
                    
            except Exception as e:
                self.stats['stocks_failed'] += 1
                self.stats['failed_files'].append(f"STOCK: {file_path.stem} - {str(e)}")
                logger.error(f"❌ Error processing {file_path}: {str(e)}")
    
    def process_etf_files(self):
        """Process all ETF TXT files"""
        logger.info("Starting efficient ETF file processing...")
        
        if not self.etf_source_dir.exists():
            logger.error(f"ETF source directory not found: {self.etf_source_dir}")
            return
        
        etf_files = list(self.etf_source_dir.glob("*.txt"))
        logger.info(f"Found {len(etf_files)} ETF files to process")
        
        for file_path in etf_files:
            try:
                etf_symbol = file_path.stem
                
                # Process to daily EOD
                daily_df = self.process_file_to_daily_eod(file_path, is_etf=True)
                
                if daily_df is not None and not daily_df.empty:
                    # Save as CSV
                    output_file = self.etf_output_dir / f"{etf_symbol}.csv"
                    daily_df.to_csv(output_file, index=False)
                    
                    self.stats['etfs_processed'] += 1
                    logger.info(f"✓ Saved {etf_symbol}: {len(daily_df):,} daily records")
                else:
                    self.stats['etfs_failed'] += 1
                    self.stats['failed_files'].append(f"ETF: {etf_symbol}")
                    logger.warning(f"❌ Failed {etf_symbol}: No valid daily data")
                    
            except Exception as e:
                self.stats['etfs_failed'] += 1
                self.stats['failed_files'].append(f"ETF: {file_path.stem} - {str(e)}")
                logger.error(f"❌ Error processing {file_path}: {str(e)}")
    
    def save_processing_stats(self):
        """Save processing statistics"""
        self.stats['end_time'] = datetime.now()
        self.stats['total_time_minutes'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds() / 60
        
        # Calculate compression ratio
        compression_ratio = (1 - (self.stats['total_daily_records'] / self.stats['total_original_records'])) * 100 if self.stats['total_original_records'] > 0 else 0
        
        # Save to JSON
        stats_file = self.output_dir / "daily_eod_extraction_stats_1998_2017.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DAILY EOD EXTRACTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Time Period: 1998-2017 (20 years)")
        logger.info(f"Strategy: Last minute of each trading day + daily aggregates")
        logger.info(f"Total Processing Time: {self.stats['total_time_minutes']:.1f} minutes")
        logger.info(f"\nDATA COMPRESSION:")
        logger.info(f"  Original records: {self.stats['total_original_records']:,}")
        logger.info(f"  Daily records: {self.stats['total_daily_records']:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.1f}%")
        logger.info(f"\nSTOCKS:")
        logger.info(f"  ✓ Processed: {self.stats['stocks_processed']}")
        logger.info(f"  ❌ Failed: {self.stats['stocks_failed']}")
        logger.info(f"\nETFs:")
        logger.info(f"  ✓ Processed: {self.stats['etfs_processed']}")
        logger.info(f"  ❌ Failed: {self.stats['etfs_failed']}")
        
        if self.stats['failed_files']:
            logger.info(f"\nFailed Files:")
            for failed in self.stats['failed_files']:
                logger.info(f"  - {failed}")
        
        success_rate = (self.stats['stocks_processed'] + self.stats['etfs_processed']) / (
            self.stats['stocks_processed'] + self.stats['stocks_failed'] + 
            self.stats['etfs_processed'] + self.stats['etfs_failed']
        ) * 100
        
        logger.info(f"\nOverall Success Rate: {success_rate:.1f}%")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*60)
    
    def run_extraction(self):
        """Run the complete daily EOD extraction process"""
        logger.info("Starting Efficient Daily EOD Extraction (1998-2017)")
        logger.info(f"Strategy: Last minute of each trading day + daily aggregates")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Process stocks
        self.process_stock_files()
        
        # Process ETFs
        self.process_etf_files()
        
        # Save statistics
        self.save_processing_stats()
        
        logger.info("Daily EOD extraction complete!")

def main():
    """Main execution function"""
    extractor = DailyEODExtractor()
    extractor.run_extraction()

if __name__ == "__main__":
    main()
