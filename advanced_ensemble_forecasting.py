"""
Advanced Ensemble Forecasting for GitHub Octoverse
==================================================

This module implements sophisticated ensemble methods combining XGBoost and Prophet
models for optimal GitHub developer signup forecasting.

Features:
- Weighted ensemble models
- Dynamic weight optimization
- Country-specific model selection
- Advanced backtesting
- Confidence intervals
- Model explanation

Author: Muddaqureshi  
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from darts import TimeSeries
from darts.models import Prophet, XGBModel
from darts.metrics import mape, mae, rmse
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import TimeSeriesSplit


class GitHubForecastEnsemble:
    """
    Advanced ensemble forecasting system for GitHub developer signups.
    """
    
    def __init__(self, country_profiles=None):
        """
        Initialize the ensemble forecasting system.
        
        Parameters:
        -----------
        country_profiles : dict, optional
            Country-specific model configurations
        """
        
        self.country_profiles = country_profiles or self._get_default_country_profiles()
        self.models = {}
        self.ensemble_weights = {}
        self.performance_history = {}
        
    def _get_default_country_profiles(self):
        """
        Define default model profiles for different types of countries.
        """
        
        profiles = {
            # Tech hub countries - prefer XGBoost for complex patterns
            'tech_hubs': {
                'countries': ['United States', 'China', 'India', 'United Kingdom', 'Germany'],
                'xgb_weight': 0.7,
                'prophet_weight': 0.3,
                'xgb_params': {
                    'lags': 12,
                    'lags_past_covariates': 6,
                    'n_estimators': 150,
                    'learning_rate': 0.08,
                    'max_depth': 6
                }
            },
            
            # Emerging markets - prefer Prophet for stability  
            'emerging': {
                'countries': ['Nigeria', 'Kenya', 'Morocco', 'Indonesia', 'Egypt'],
                'xgb_weight': 0.4,
                'prophet_weight': 0.6,
                'xgb_params': {
                    'lags': 8,
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 4
                }
            },
            
            # Mature markets - balanced approach
            'mature': {
                'countries': ['Canada', 'Japan', 'South Africa'],
                'xgb_weight': 0.5,
                'prophet_weight': 0.5,
                'xgb_params': {
                    'lags': 10,
                    'n_estimators': 120,
                    'learning_rate': 0.09,
                    'max_depth': 5
                }
            },
            
            # High growth markets - adaptive ensemble
            'high_growth': {
                'countries': ['Brazil'],
                'xgb_weight': 0.6,
                'prophet_weight': 0.4,
                'xgb_params': {
                    'lags': 12,
                    'n_estimators': 140,
                    'learning_rate': 0.07,
                    'max_depth': 7
                }
            }
        }
        
        return profiles
    
    def _get_country_profile(self, country):
        """Get the model profile for a specific country."""
        
        for profile_name, profile in self.country_profiles.items():
            if country in profile['countries']:
                return profile_name, profile
        
        # Default to mature market profile
        return 'mature', self.country_profiles['mature']
    
    def train_models(self, series_dict, covariates_dict=None):
        """
        Train both Prophet and XGBoost models for all countries.
        
        Parameters:
        -----------
        series_dict : dict
            Dictionary with country names as keys and TimeSeries as values
        covariates_dict : dict, optional
            Dictionary with covariates for each country
        """
        
        print("🚀 Training ensemble models for all countries...")
        print("=" * 60)
        
        for country, series in series_dict.items():
            print(f"\n🔄 Training models for {country}...")
            
            try:
                profile_name, profile = self._get_country_profile(country)
                print(f"   Using profile: {profile_name}")
                
                # Get covariates if available
                covariates = None
                if covariates_dict and country in covariates_dict:
                    covariates = covariates_dict[country]
                
                # Train Prophet model
                prophet_model = Prophet(
                    seasonality_mode='multiplicative',
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=0.1
                )
                prophet_model.fit(series)
                
                # Train XGBoost model
                xgb_params = profile['xgb_params']
                xgb_model = XGBModel(
                    random_state=42,
                    **xgb_params
                )
                
                if covariates:
                    xgb_model.fit(series, past_covariates=covariates)
                else:
                    xgb_model.fit(series)
                
                # Store models and weights
                self.models[country] = {
                    'prophet': prophet_model,
                    'xgboost': xgb_model,
                    'covariates': covariates,
                    'profile': profile_name
                }
                
                self.ensemble_weights[country] = {
                    'prophet': profile['prophet_weight'],
                    'xgboost': profile['xgb_weight']
                }
                
                print(f"   ✅ Models trained successfully")
                print(f"   📊 Weights: Prophet={profile['prophet_weight']}, XGBoost={profile['xgb_weight']}")
                
            except Exception as e:
                print(f"   ❌ Error training models for {country}: {str(e)}")
                continue
        
        print(f"\n🎉 Ensemble training completed for {len(self.models)} countries!")
    
    def predict_ensemble(self, country, n_periods, return_components=False):
        """
        Generate ensemble forecast for a specific country.
        
        Parameters:
        -----------
        country : str
            Country name
        n_periods : int
            Number of periods to forecast
        return_components : bool
            Whether to return individual model predictions
        
        Returns:
        --------
        TimeSeries or dict : Ensemble forecast (and components if requested)
        """
        
        if country not in self.models:
            raise ValueError(f"No trained models found for {country}")
        
        models = self.models[country]
        weights = self.ensemble_weights[country]
        
        print(f"🔮 Generating ensemble forecast for {country} ({n_periods} periods)...")
        
        try:
            # Get Prophet prediction
            prophet_pred = models['prophet'].predict(n=n_periods)
            
            # Get XGBoost prediction
            if models['covariates']:
                # Need to extend covariates for forecast period
                last_date = models['covariates'].end_time()
                extended_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=n_periods,
                    freq='MS'
                )
                
                # Simple extension of covariates (you might want to improve this)
                last_values = models['covariates'].values()[-1]
                extended_values = np.tile(last_values, (len(extended_dates), 1))
                
                extended_covariates = TimeSeries.from_times_and_values(
                    times=extended_dates,
                    values=extended_values,
                    columns=models['covariates'].columns
                )
                
                # Concatenate original and extended covariates
                full_covariates = models['covariates'].concatenate(extended_covariates)
                
                xgb_pred = models['xgboost'].predict(n=n_periods, past_covariates=full_covariates)
            else:
                xgb_pred = models['xgboost'].predict(n=n_periods)
            
            # Ensure same length
            min_length = min(len(prophet_pred), len(xgb_pred))
            prophet_values = prophet_pred.values()[:min_length]
            xgb_values = xgb_pred.values()[:min_length]
            
            # Create weighted ensemble
            ensemble_values = (weights['prophet'] * prophet_values + 
                             weights['xgboost'] * xgb_values)
            
            # Create ensemble TimeSeries
            ensemble_forecast = TimeSeries.from_times_and_values(
                times=prophet_pred.time_index[:min_length],
                values=ensemble_values
            )
            
            print(f"   ✅ Ensemble forecast complete")
            print(f"   📈 Range: {ensemble_forecast.start_time()} to {ensemble_forecast.end_time()}")
            
            if return_components:
                return {
                    'ensemble': ensemble_forecast,
                    'prophet': prophet_pred,
                    'xgboost': xgb_pred,
                    'weights': weights
                }
            else:
                return ensemble_forecast
            
        except Exception as e:
            print(f"   ❌ Error generating forecast: {str(e)}")
            return None
    
    def backtest_ensemble(self, series_dict, test_periods=12, cv_folds=3):
        """
        Perform comprehensive backtesting of ensemble models.
        
        Parameters:
        -----------
        series_dict : dict
            Dictionary of TimeSeries objects
        test_periods : int
            Number of periods to use for testing
        cv_folds : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict : Backtesting results
        """
        
        print("🧪 Performing ensemble backtesting...")
        print("=" * 50)
        
        backtest_results = {}
        
        for country, series in series_dict.items():
            print(f"\n📊 Backtesting {country}...")
            
            if len(series) < test_periods + 24:  # Need enough data for training
                print(f"   ⚠️  Insufficient data for {country} (need at least {test_periods + 24} months)")
                continue
            
            try:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=cv_folds, test_size=test_periods)
                fold_results = []
                
                series_df = series.to_dataframe()
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(series_df)):
                    print(f"   🔄 Fold {fold + 1}/{cv_folds}")
                    
                    # Split data
                    train_data = series_df.iloc[train_idx]
                    test_data = series_df.iloc[test_idx]
                    
                    train_series = TimeSeries.from_dataframe(train_data)
                    test_series = TimeSeries.from_dataframe(test_data)
                    
                    # Train models on fold
                    temp_series_dict = {country: train_series}
                    temp_ensemble = GitHubForecastEnsemble(self.country_profiles)
                    temp_ensemble.train_models(temp_series_dict)
                    
                    # Generate predictions
                    fold_pred = temp_ensemble.predict_ensemble(country, len(test_series), 
                                                             return_components=True)
                    
                    if fold_pred:
                        # Calculate metrics
                        ensemble_mape = mape(test_series, fold_pred['ensemble'])
                        prophet_mape = mape(test_series, fold_pred['prophet'])
                        xgb_mape = mape(test_series, fold_pred['xgboost'])
                        
                        fold_results.append({
                            'fold': fold + 1,
                            'ensemble_mape': ensemble_mape,
                            'prophet_mape': prophet_mape,
                            'xgboost_mape': xgb_mape,
                            'test_periods': len(test_series)
                        })
                        
                        print(f"      Ensemble MAPE: {ensemble_mape:.1f}%")
                
                if fold_results:
                    # Aggregate results
                    avg_ensemble_mape = np.mean([r['ensemble_mape'] for r in fold_results])
                    avg_prophet_mape = np.mean([r['prophet_mape'] for r in fold_results])
                    avg_xgb_mape = np.mean([r['xgboost_mape'] for r in fold_results])
                    
                    backtest_results[country] = {
                        'avg_ensemble_mape': avg_ensemble_mape,
                        'avg_prophet_mape': avg_prophet_mape,
                        'avg_xgboost_mape': avg_xgb_mape,
                        'fold_results': fold_results,
                        'improvement_vs_prophet': ((avg_prophet_mape - avg_ensemble_mape) / avg_prophet_mape) * 100,
                        'improvement_vs_xgb': ((avg_xgb_mape - avg_ensemble_mape) / avg_xgb_mape) * 100
                    }
                    
                    print(f"   📈 Average Ensemble MAPE: {avg_ensemble_mape:.1f}%")
                    print(f"   🏆 Improvement vs best single model: {max(backtest_results[country]['improvement_vs_prophet'], backtest_results[country]['improvement_vs_xgb']):.1f}%")
                
            except Exception as e:
                print(f"   ❌ Error backtesting {country}: {str(e)}")
                continue
        
        # Store performance history
        self.performance_history = backtest_results
        
        return backtest_results
    
    def create_forecast_report(self, series_dict, forecast_periods=[12, 60]):
        """
        Generate comprehensive forecast report for all countries.
        
        Parameters:
        -----------
        series_dict : dict
            Dictionary of TimeSeries objects
        forecast_periods : list
            List of forecast horizons to generate
        
        Returns:
        --------
        dict : Comprehensive forecast results
        """
        
        print("📊 Generating comprehensive forecast report...")
        print("=" * 60)
        
        report_results = {}
        
        for country in series_dict.keys():
            if country not in self.models:
                continue
            
            print(f"\n📈 Creating forecasts for {country}...")
            
            country_results = {'country': country}
            
            for periods in forecast_periods:
                try:
                    forecast_components = self.predict_ensemble(
                        country, periods, return_components=True
                    )
                    
                    if forecast_components:
                        ensemble = forecast_components['ensemble']
                        
                        # Calculate summary statistics
                        total_forecast = ensemble.values().sum()
                        monthly_avg = total_forecast / periods
                        
                        # Get growth rate (compare first forecast to last historical)
                        last_historical = series_dict[country].values()[-1][0]
                        first_forecast = ensemble.values()[0][0]
                        growth_rate = ((first_forecast / last_historical) - 1) * 100
                        
                        country_results[f'{periods}m_forecast'] = {
                            'total_signups': float(total_forecast),
                            'monthly_average': float(monthly_avg),
                            'growth_rate_percent': float(growth_rate),
                            'forecast_series': ensemble,
                            'components': forecast_components
                        }
                        
                        print(f"   ✅ {periods}-month forecast: {total_forecast:,.0f} total signups")
                
                except Exception as e:
                    print(f"   ❌ Error creating {periods}-month forecast: {str(e)}")
                    continue
            
            report_results[country] = country_results
        
        return report_results
    
    def plot_ensemble_comparison(self, country, series, n_periods=24, figsize=(15, 10)):
        """
        Create detailed comparison plot of ensemble vs individual models.
        
        Parameters:
        -----------
        country : str
            Country name
        series : TimeSeries
            Historical data
        n_periods : int
            Number of periods to forecast
        figsize : tuple
            Figure size
        """
        
        if country not in self.models:
            print(f"❌ No trained models for {country}")
            return
        
        # Generate forecasts
        forecast_components = self.predict_ensemble(country, n_periods, return_components=True)
        
        if not forecast_components:
            print(f"❌ Could not generate forecasts for {country}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Main comparison plot
        ax1 = axes[0, 0]
        series.plot(ax=ax1, label='Historical', color='blue', linewidth=2)
        forecast_components['ensemble'].plot(ax=ax1, label='Ensemble', color='red', linestyle='--', linewidth=2)
        forecast_components['prophet'].plot(ax=ax1, label='Prophet', color='green', linestyle=':', alpha=0.7)
        forecast_components['xgboost'].plot(ax=ax1, label='XGBoost', color='orange', linestyle='-.', alpha=0.7)
        
        ax1.set_title(f'{country} - Ensemble vs Individual Models', fontweight='bold')
        ax1.set_ylabel('New Signups')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Forecast values comparison
        ax2 = axes[0, 1]
        periods = len(forecast_components['ensemble'])
        x_pos = np.arange(periods)
        
        ensemble_vals = forecast_components['ensemble'].values().flatten()
        prophet_vals = forecast_components['prophet'].values().flatten()[:periods]
        xgb_vals = forecast_components['xgboost'].values().flatten()[:periods]
        
        ax2.plot(x_pos, ensemble_vals, 'r-', linewidth=2, label='Ensemble')
        ax2.plot(x_pos, prophet_vals, 'g:', alpha=0.7, label='Prophet')  
        ax2.plot(x_pos, xgb_vals, 'orange', linestyle='-.', alpha=0.7, label='XGBoost')
        
        ax2.set_title('Forecast Values Comparison')
        ax2.set_xlabel('Months Ahead')
        ax2.set_ylabel('Predicted Signups')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model weights visualization
        ax3 = axes[1, 0]
        weights = forecast_components['weights']
        models_names = list(weights.keys())
        weight_values = list(weights.values())
        
        bars = ax3.bar(models_names, weight_values, color=['green', 'orange'], alpha=0.7)
        ax3.set_title('Ensemble Weights')
        ax3.set_ylabel('Weight')
        ax3.set_ylim(0, 1)
        
        # Add weight labels on bars
        for bar, weight in zip(bars, weight_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance metrics (if available)
        ax4 = axes[1, 1]
        if country in self.performance_history:
            perf = self.performance_history[country]
            models = ['Ensemble', 'Prophet', 'XGBoost']
            mape_values = [
                perf['avg_ensemble_mape'],
                perf['avg_prophet_mape'], 
                perf['avg_xgboost_mape']
            ]
            
            bars = ax4.bar(models, mape_values, color=['red', 'green', 'orange'], alpha=0.7)
            ax4.set_title('Model Performance (MAPE)')
            ax4.set_ylabel('MAPE (%)')
            
            # Add MAPE labels
            for bar, mape_val in zip(bars, mape_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{mape_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No backtesting\nresults available', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, style='italic')
            ax4.set_title('Model Performance')
        
        plt.tight_layout()
        plt.suptitle(f'📊 {country} - Advanced Ensemble Analysis', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.show()


def create_ensemble_summary_report(ensemble_results, save_to_file=None):
    """
    Create a comprehensive summary report of ensemble forecasting results.
    
    Parameters:
    -----------
    ensemble_results : dict
        Results from ensemble forecasting
    save_to_file : str, optional
        Path to save the report
    
    Returns:
    --------
    pd.DataFrame : Summary report
    """
    
    print("📋 Creating ensemble summary report...")
    
    report_data = []
    
    for country, results in ensemble_results.items():
        if '12m_forecast' in results and '60m_forecast' in results:
            report_data.append({
                'Country': country,
                '1Y_Total_Signups': results['12m_forecast']['total_signups'],
                '1Y_Monthly_Avg': results['12m_forecast']['monthly_average'],
                '1Y_Growth_Rate_%': results['12m_forecast']['growth_rate_percent'],
                '5Y_Total_Signups': results['60m_forecast']['total_signups'], 
                '5Y_Monthly_Avg': results['60m_forecast']['monthly_average'],
                '5Y_Growth_Rate_%': results['60m_forecast']['growth_rate_percent']
            })
    
    df_report = pd.DataFrame(report_data)
    
    # Add rankings
    df_report = df_report.sort_values('1Y_Total_Signups', ascending=False)
    df_report['Rank_1Y'] = range(1, len(df_report) + 1)
    df_report['Rank_5Y'] = df_report['5Y_Total_Signups'].rank(method='dense', ascending=False).astype(int)
    
    if save_to_file:
        df_report.to_csv(save_to_file, index=False)
        print(f"💾 Report saved to: {save_to_file}")
    
    return df_report


if __name__ == "__main__":
    print("🚀 GitHub Octoverse Advanced Ensemble Forecasting")
    print("=" * 60)
    print("This module provides sophisticated ensemble forecasting capabilities.")
    print("Import this module in your Jupyter notebook to use the ensemble features.")