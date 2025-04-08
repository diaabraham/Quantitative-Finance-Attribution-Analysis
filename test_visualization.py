import pandas as pd
import numpy as np
from attribution.visualization import (
    plot_factor_contributions,
    format_brinson_table,
    plot_attribution_breakdown,
    plot_time_series_attribution,
    save_visualization
)

# Generate sample factor contributions
factor_contributions = pd.Series({
    'market': 0.008,
    'size': -0.002,
    'value': 0.001,
    'momentum': 0.001,
    'quality': 0.0005
})

# Generate sample Brinson attribution results
attribution_results = {
    'allocation': {
        'Technology': 0.012,
        'Healthcare': -0.005,
        'Financials': 0.003,
        'Total': 0.01
    },
    'selection': {
        'Technology': 0.008,
        'Healthcare': 0.003,
        'Financials': 0.002,
        'Total': 0.013
    },
    'interaction': {
        'Technology': 0.002,
        'Healthcare': -0.001,
        'Financials': 0.001,
        'Total': 0.002
    },
    'total_effect': 0.025
}

# Generate sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
time_series_results = {
    'allocation': pd.Series(np.random.normal(0.0001, 0.001, len(dates)), index=dates),
    'selection': pd.Series(np.random.normal(0.0002, 0.001, len(dates)), index=dates),
    'interaction': pd.Series(np.random.normal(0.00005, 0.0005, len(dates)), index=dates)
}

# Create visualizations
print("Generating visualizations...")

# 1. Factor Contributions Bar Chart
fig1 = plot_factor_contributions(factor_contributions)
save_visualization(fig1, 'factor_contributions.png')
print("Saved factor_contributions.png")

# 2. Brinson Attribution Table
table = format_brinson_table(attribution_results)
print("\nBrinson Attribution Results:")
print(table)

# 3. Attribution Breakdown
fig2 = plot_attribution_breakdown(attribution_results)
save_visualization(fig2, 'attribution_breakdown.png')
print("Saved attribution_breakdown.png")

# 4. Time Series Attribution
fig3 = plot_time_series_attribution(time_series_results)
save_visualization(fig3, 'time_series_attribution.png')
print("Saved time_series_attribution.png")

print("\nAll visualizations generated successfully!") 