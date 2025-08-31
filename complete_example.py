"""
GitHub Octoverse Complete Example Script
=======================================

This script demonstrates how to use all the time series forecasting utilities
together to create comprehensive GitHub developer growth projections.

Usage:
    python complete_example.py

Author: Muddaqureshi
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function that runs the complete forecasting pipeline.
    """
    
    print("🚀 GitHub Octoverse - Complete Forecasting Example")
    print("=" * 60)
    print(f"📅 Analysis started at: {pd.Timestamp.now()}")
    
    # Step 1: Load sample data
    print("\n📊 Step 1: Loading sample data...")
    try:
        df = pd.read_csv('Sample_GitHub_Developer_Signups.csv')
        print(f"   ✅ Loaded {len(df)} records for {df['country_name'].nunique()} countries")
        print(f"   📅 Date range: {df['created_cohort'].min()} to {df['created_cohort'].max()}")
    except FileNotFoundError:
        print("   ❌ Sample data file not found. Please run the data generation script first.")
        return
    
    # Step 2: Convert to time series format
    print("\n🔧 Step 2: Preparing time series data...")
    
    df['date'] = pd.to_datetime(df['created_cohort'])
    country_series = {}
    
    for country in df['country_name'].unique():
        country_data = df[df['country_name'] == country].copy()
        country_data = country_data.sort_values('date')
        
        # Create simple time series (without Darts for basic example)
        country_series[country] = country_data[['date', 'new_signups']].copy()
        
    print(f"   ✅ Prepared time series for {len(country_series)} countries")
    
    # Step 3: Simple Prophet-style forecasting (conceptual)
    print("\n🔮 Step 3: Creating simple trend forecasts...")
    
    forecast_results = {}
    
    for country, data in country_series.items():
        try:
            # Simple linear trend forecasting for demonstration
            # In real usage, you'd use Prophet or XGBoost here
            
            # Calculate recent trend (last 12 months)
            recent_data = data.tail(12)
            if len(recent_data) < 6:
                continue
                
            # Simple linear regression on recent data
            x = np.arange(len(recent_data))
            y = recent_data['new_signups'].values
            
            # Fit line: y = mx + b
            m, b = np.polyfit(x, y, 1)
            
            # Generate 12-month and 60-month forecasts
            forecast_12m = []
            forecast_60m = []
            
            last_x = len(recent_data) - 1
            
            for i in range(1, 61):
                # Linear extrapolation with some seasonality
                base_forecast = m * (last_x + i) + b
                
                # Add simple seasonality (±10%)
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)
                
                # Add small growth acceleration for emerging markets
                if country in ['Nigeria', 'Kenya', 'Indonesia', 'Brazil']:
                    growth_factor = 1 + 0.02 * i / 12  # 2% annual acceleration
                else:
                    growth_factor = 1
                
                forecast_value = max(1000, int(base_forecast * seasonal_factor * growth_factor))
                
                if i <= 12:
                    forecast_12m.append(forecast_value)
                forecast_60m.append(forecast_value)
            
            # Store results
            total_1y = sum(forecast_12m)
            total_5y = sum(forecast_60m)
            
            forecast_results[country] = {
                'total_1y': total_1y,
                'total_5y': total_5y,
                'monthly_avg_1y': total_1y / 12,
                'monthly_avg_5y': total_5y / 60,
                'last_actual': int(recent_data['new_signups'].iloc[-1]),
                'growth_rate': ((forecast_12m[0] / recent_data['new_signups'].iloc[-1]) - 1) * 100
            }
            
            print(f"   ✅ {country}: 1Y={total_1y:,.0f}, 5Y={total_5y:,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error forecasting {country}: {str(e)}")
            continue
    
    # Step 4: Create summary analysis
    print("\n📊 Step 4: Creating analysis summary...")
    
    if not forecast_results:
        print("   ❌ No forecast results to analyze")
        return
    
    # Create summary DataFrame
    summary_data = []
    for country, results in forecast_results.items():
        summary_data.append({
            'Country': country,
            '1Y_Total_Signups': results['total_1y'],
            '1Y_Monthly_Avg': int(results['monthly_avg_1y']),
            '5Y_Total_Signups': results['total_5y'],
            '5Y_Monthly_Avg': int(results['monthly_avg_5y']),
            'Last_Actual': results['last_actual'],
            'Growth_Rate_%': round(results['growth_rate'], 1)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('1Y_Total_Signups', ascending=False)
    summary_df['Rank'] = range(1, len(summary_df) + 1)
    
    # Display top results
    print(f"\n🏆 TOP 10 COUNTRIES - 1 YEAR PROJECTIONS:")
    print("-" * 80)
    print(summary_df.head(10)[['Rank', 'Country', '1Y_Total_Signups', '1Y_Monthly_Avg', 'Growth_Rate_%']].to_string(index=False))
    
    # Step 5: Create visualizations
    print("\n📈 Step 5: Creating visualizations...")
    
    # Bar chart of top countries
    plt.figure(figsize=(14, 8))
    
    top_10 = summary_df.head(10)
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(top_10)), top_10['1Y_Total_Signups'] / 1_000_000, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('1-Year Projections - Top 10 Countries', fontsize=14, fontweight='bold')
    plt.xlabel('Countries')
    plt.ylabel('Projected Signups (Millions)')
    plt.xticks(range(len(top_10)), top_10['Country'], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_10['1Y_Total_Signups'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value/1_000_000:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 2, 2)  
    # Growth rate chart
    growth_top = summary_df.sort_values('Growth_Rate_%', ascending=False).head(10)
    colors = ['green' if x > 0 else 'red' for x in growth_top['Growth_Rate_%']]
    bars2 = plt.bar(range(len(growth_top)), growth_top['Growth_Rate_%'], 
                    color=colors, alpha=0.7)
    plt.title('Growth Rate Projections - Top 10', fontsize=14, fontweight='bold')
    plt.xlabel('Countries')
    plt.ylabel('Growth Rate (%)')
    plt.xticks(range(len(growth_top)), growth_top['Country'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, growth_top['Growth_Rate_%'])):
        height = bar.get_height()
        y_pos = height + (0.5 if height > 0 else -1.5)
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('📊 GitHub Octoverse - Developer Growth Projections', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save plot
    plt.savefig('GitHub_Octoverse_Projections.png', dpi=300, bbox_inches='tight')
    print("   ✅ Chart saved as 'GitHub_Octoverse_Projections.png'")
    
    plt.show()
    
    # Step 6: Export results
    print("\n💾 Step 6: Exporting results...")
    
    # Save summary to CSV
    summary_df.to_csv('GitHub_Octoverse_Summary.csv', index=False)
    print("   ✅ Summary saved to 'GitHub_Octoverse_Summary.csv'")
    
    # Create detailed monthly breakdown
    detailed_data = []
    for country, results in forecast_results.items():
        # Add sample monthly breakdown (first few months)
        base_monthly = results['monthly_avg_1y']
        for month in range(1, 13):
            # Simple monthly variation
            seasonal_adj = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
            monthly_forecast = int(base_monthly * seasonal_adj)
            
            detailed_data.append({
                'Country': country,
                'Month': month,
                'Month_Name': pd.Timestamp(2025, month, 1).strftime('%b'),
                'Projected_Signups': monthly_forecast
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('GitHub_Octoverse_Monthly_Details.csv', index=False)
    print("   ✅ Monthly details saved to 'GitHub_Octoverse_Monthly_Details.csv'")
    
    # Step 7: Executive Summary
    print("\n📋 Step 7: Executive Summary")
    print("=" * 60)
    
    global_1y_total = summary_df['1Y_Total_Signups'].sum()
    global_5y_total = summary_df['5Y_Total_Signups'].sum()
    
    print(f"\n🌍 GLOBAL PROJECTIONS:")
    print(f"   📈 Total 1-year signups: {global_1y_total:,.0f}")
    print(f"   📈 Total 5-year signups: {global_5y_total:,.0f}")
    print(f"   📊 Average monthly (1Y): {global_1y_total/12:,.0f}")
    print(f"   📊 Average monthly (5Y): {global_5y_total/60:,.0f}")
    
    print(f"\n🏆 KEY INSIGHTS:")
    top_country = summary_df.iloc[0]
    fastest_growth = summary_df.sort_values('Growth_Rate_%', ascending=False).iloc[0]
    
    print(f"   • Largest market (1Y): {top_country['Country']} ({top_country['1Y_Total_Signups']:,.0f} signups)")
    print(f"   • Fastest growing: {fastest_growth['Country']} ({fastest_growth['Growth_Rate_%']:+.1f}% growth)")
    print(f"   • Countries analyzed: {len(summary_df)}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Focus developer outreach on top 3 growth markets")
    print(f"   • Plan infrastructure capacity for {global_1y_total:,.0f} new users in next 12 months")
    print(f"   • Monitor actual performance vs forecasts monthly")
    print(f"   • Consider regional customization for high-growth countries")
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📁 Output files created:")
    print(f"   • GitHub_Octoverse_Summary.csv")
    print(f"   • GitHub_Octoverse_Monthly_Details.csv") 
    print(f"   • GitHub_Octoverse_Projections.png")


if __name__ == "__main__":
    main()