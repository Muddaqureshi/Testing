"""
Data Preparation Utilities for GitHub Octoverse Time Series Forecasting
=========================================================================

This module provides utility functions to help prepare your GitHub developer 
signup data for time series forecasting with Darts.

Author: Muddaqureshi
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from darts import TimeSeries
import warnings
warnings.filterwarnings('ignore')


def load_and_validate_data(file_path, required_columns=None):
    """
    Load CSV data and validate it has the required structure.
    
    Parameters:
    -----------
    file_path : str
        Path to your CSV file
    required_columns : list, optional
        List of required column names
    
    Returns:
    --------
    pd.DataFrame : Validated DataFrame
    """
    
    if required_columns is None:
        required_columns = ['country_name', 'created_cohort', 'new_signups']
    
    print(f"📂 Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {len(df)} rows")
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if 'new_signups' in df.columns:
            df['new_signups'] = pd.to_numeric(df['new_signups'], errors='coerce')
            
        # Remove rows with missing critical data
        original_len = len(df)
        df = df.dropna(subset=['country_name', 'created_cohort', 'new_signups'])
        
        if len(df) < original_len:
            print(f"⚠️  Removed {original_len - len(df)} rows with missing data")
        
        print(f"📊 Final dataset: {len(df)} rows, {df['country_name'].nunique()} countries")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def create_country_time_series(df, country_col='country_name', date_col='created_cohort', 
                             value_col='new_signups'):
    """
    Convert DataFrame to dictionary of Darts TimeSeries objects.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    country_col : str
        Column name containing country names
    date_col : str  
        Column name containing dates (YYYY-MM format)
    value_col : str
        Column name containing values to forecast
    
    Returns:
    --------
    dict : Dictionary with country names as keys, TimeSeries as values
    """
    
    print("🔧 Converting to TimeSeries format...")
    
    # Convert date column to datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df[date_col], format='%Y-%m')
    
    country_series = {}
    
    for country in df[country_col].unique():
        try:
            # Filter data for this country
            country_data = df[df[country_col] == country].copy()
            country_data = country_data.sort_values('date')
            
            # Create TimeSeries
            series = TimeSeries.from_dataframe(
                country_data,
                time_col='date',
                value_cols=value_col
            )
            
            country_series[country] = series
            print(f"  ✅ {country}: {len(series)} months from {series.start_time().strftime('%Y-%m')} to {series.end_time().strftime('%Y-%m')}")
            
        except Exception as e:
            print(f"  ❌ Error processing {country}: {str(e)}")
            continue
    
    return country_series


def add_github_product_covariates(series_dict):
    """
    Add GitHub product release dates and events as covariates.
    
    Parameters:
    -----------
    series_dict : dict
        Dictionary of TimeSeries objects
    
    Returns:
    --------
    dict : Dictionary with enhanced TimeSeries including covariates
    """
    
    print("🚀 Adding GitHub product release covariates...")
    
    enhanced_series = {}
    
    for country, series in series_dict.items():
        try:
            # Convert to DataFrame for feature engineering
            df = series.to_dataframe()
            
            # GitHub Copilot Individual launch (June 21, 2022)
            copilot_individual = pd.Timestamp('2022-06-21')
            df['copilot_individual_launched'] = (df.index >= copilot_individual).astype(int)
            df['months_since_copilot_individual'] = np.maximum(0, 
                (df.index.to_series() - copilot_individual).dt.days / 30.44)
            
            # GitHub Copilot Free launch (December 18, 2024)
            copilot_free = pd.Timestamp('2024-12-18')
            df['copilot_free_launched'] = (df.index >= copilot_free).astype(int)
            df['months_since_copilot_free'] = np.maximum(0,
                (df.index.to_series() - copilot_free).dt.days / 30.44)
            
            # GitHub Universe events (major developer conferences)
            universe_events = [
                pd.Timestamp('2022-11-09'),  # Universe 2022
                pd.Timestamp('2023-11-08'),  # Universe 2023
                pd.Timestamp('2024-10-29'),  # Universe 2024
            ]
            
            for i, universe_date in enumerate(universe_events):
                # Create impact feature (decaying over 3 months)
                days_from_event = (df.index.to_series() - universe_date).dt.days
                df[f'universe_{2022+i}_impact'] = np.where(
                    (days_from_event >= 0) & (days_from_event <= 90),
                    np.exp(-days_from_event / 30),  # Exponential decay
                    0
                )
            
            # Economic indicators (simplified)
            df['linear_trend'] = (df.index - df.index[0]).days / 365.25
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # COVID-19 impact period
            covid_start = pd.Timestamp('2020-03-01')
            covid_end = pd.Timestamp('2020-12-31')
            df['covid_period'] = ((df.index >= covid_start) & (df.index <= covid_end)).astype(int)
            
            # Create enhanced TimeSeries
            value_cols = [col for col in df.columns if 'new_signups' in col]
            covariate_cols = [col for col in df.columns if col not in value_cols]
            
            enhanced_ts = TimeSeries.from_dataframe(df, value_cols=value_cols[0])
            covariates_ts = TimeSeries.from_dataframe(df, value_cols=covariate_cols)
            
            enhanced_series[country] = {
                'series': enhanced_ts,
                'covariates': covariates_ts
            }
            
            print(f"  ✅ {country}: Added {len(covariate_cols)} covariate features")
            
        except Exception as e:
            print(f"  ❌ Error processing {country}: {str(e)}")
            # Fallback to original series
            enhanced_series[country] = {'series': series, 'covariates': None}
    
    return enhanced_series


def create_sample_data(countries=None, start_date='2018-01', end_date='2025-08', 
                      save_to_file=None):
    """
    Create realistic sample data for demonstration purposes.
    
    Parameters:
    -----------
    countries : list, optional
        List of countries to include
    start_date : str
        Start date in 'YYYY-MM' format
    end_date : str
        End date in 'YYYY-MM' format
    save_to_file : str, optional
        Path to save the sample data as CSV
    
    Returns:
    --------
    pd.DataFrame : Sample dataset
    """
    
    if countries is None:
        countries = [
            'India', 'United States', 'Brazil', 'China', 'Japan',
            'Germany', 'Indonesia', 'United Kingdom', 'Canada',
            'Nigeria', 'Kenya', 'Egypt', 'South Africa', 'Morocco'
        ]
    
    print(f"📊 Creating sample data for {len(countries)} countries...")
    
    # Generate date range
    date_range = pd.date_range(start_date, end_date, freq='MS')
    
    sample_data = []
    
    for country in countries:
        # Different base parameters for different countries
        country_params = {
            'India': {'base': 75000, 'growth_rate': 0.025, 'volatility': 0.15},
            'United States': {'base': 60000, 'growth_rate': 0.015, 'volatility': 0.10},
            'Brazil': {'base': 45000, 'growth_rate': 0.030, 'volatility': 0.18},
            'China': {'base': 55000, 'growth_rate': 0.020, 'volatility': 0.12},
            'Nigeria': {'base': 25000, 'growth_rate': 0.040, 'volatility': 0.25},
            'Indonesia': {'base': 35000, 'growth_rate': 0.035, 'volatility': 0.20}
        }
        
        # Default parameters for countries not specifically defined
        params = country_params.get(country, 
            {'base': 30000, 'growth_rate': 0.025, 'volatility': 0.15})
        
        for i, date in enumerate(date_range):
            # Base growth
            base_signups = params['base'] * (1 + params['growth_rate']) ** (i / 12)
            
            # Add seasonality (stronger in Q1, weaker in Q4)
            seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * i / 12 + np.pi/2)
            
            # COVID-19 impact (boost in 2020-2021)
            covid_boost = 1.0
            if 2020 <= date.year <= 2021:
                covid_boost = 1.3 if date.month in [3, 4, 5, 6] else 1.1
            
            # GitHub Copilot impact (boost after June 2022)
            copilot_boost = 1.0
            if date >= pd.Timestamp('2022-06-01'):
                months_since_copilot = (date - pd.Timestamp('2022-06-01')).days / 30.44
                copilot_boost = 1 + 0.2 * (1 - np.exp(-months_since_copilot / 12))
            
            # Random noise
            noise_factor = np.random.normal(1, params['volatility'])
            
            # Calculate final signups
            signups = int(base_signups * seasonal_factor * covid_boost * 
                         copilot_boost * noise_factor)
            signups = max(signups, 1000)  # Minimum threshold
            
            # Calculate year-over-year data
            prev_year_signups = None
            yoy_percent = None
            
            if i >= 12:  # We have previous year data
                prev_year_idx = i - 12
                prev_year_signups = sample_data[prev_year_idx * len(countries) + 
                                               countries.index(country)]['new_signups']
                yoy_percent = ((signups - prev_year_signups) / prev_year_signups) * 100
            
            sample_data.append({
                'country_name': country,
                'created_cohort': date.strftime('%Y-%m'),
                'new_signups': signups,
                'prev_year_signups': prev_year_signups,
                'yoy_percent': yoy_percent
            })
    
    df = pd.DataFrame(sample_data)
    
    if save_to_file:
        df.to_csv(save_to_file, index=False)
        print(f"💾 Sample data saved to: {save_to_file}")
    
    print(f"✅ Created sample dataset with {len(df)} rows")
    return df


def validate_forecast_quality(actual_series, predicted_series, model_name="Model"):
    """
    Calculate and display forecast quality metrics.
    
    Parameters:
    -----------
    actual_series : TimeSeries
        Actual observed values
    predicted_series : TimeSeries
        Predicted values
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    
    from darts.metrics import mape, mae, rmse, r2_score
    
    try:
        # Calculate metrics
        mape_score = mape(actual_series, predicted_series)
        mae_score = mae(actual_series, predicted_series)
        rmse_score = rmse(actual_series, predicted_series)
        r2 = r2_score(actual_series, predicted_series)
        
        metrics = {
            'MAPE': mape_score,
            'MAE': mae_score,
            'RMSE': rmse_score,
            'R²': r2
        }
        
        print(f"\n📊 {model_name} Quality Metrics:")
        print("-" * 40)
        print(f"MAPE (Mean Absolute % Error): {mape_score:.1f}%")
        print(f"MAE (Mean Absolute Error):    {mae_score:,.0f}")
        print(f"RMSE (Root Mean Square Error): {rmse_score:,.0f}")
        print(f"R² (Coefficient of Determination): {r2:.3f}")
        
        # Interpretation
        if mape_score < 10:
            interpretation = "Excellent accuracy"
        elif mape_score < 20:
            interpretation = "Good accuracy"
        elif mape_score < 30:
            interpretation = "Moderate accuracy"
        else:
            interpretation = "Poor accuracy - consider model improvements"
        
        print(f"📈 Model Assessment: {interpretation}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    print("🚀 GitHub Octoverse Data Preparation Utilities")
    print("=" * 50)
    
    # Create sample data
    sample_df = create_sample_data(save_to_file='sample_github_data.csv')
    
    # Convert to TimeSeries format
    country_series = create_country_time_series(sample_df)
    
    # Add covariates
    enhanced_series = add_github_product_covariates(country_series)
    
    print(f"\n✅ Successfully processed {len(enhanced_series)} countries")
    print("📝 Ready for time series forecasting!")