# Quantitative Finance Attribution Analysis (QuantFAA)

## TL;DR

Quantitative Finance Attribution Analysis (QuantFAA) is a Python library designed to help portfolio managers, quant researchers, and risk analysts explain performance vs. benchmarks using Brinson, factor, and risk attribution. It includes support for Fama-French models, transaction cost analysis, and risk metrics like VaR, Tail Ratio, and more, with modular, testable design and real-world application support.

#
A comprehensive Python library for performance attribution analysis in quantitative finance, supporting multiple attribution methods, factor models, and risk metrics. This library helps portfolio managers, quantitative analysts, and researchers understand the sources of portfolio returns and risks.

## What is Performance Attribution?

Performance attribution is the process of explaining why a portfolio's returns differ from its benchmark. It helps answer questions like:
- Why did my portfolio outperform/underperform the benchmark?
- Which investment decisions contributed most to performance?
- How much of the performance came from stock selection vs. sector allocation?
- What are the risk factors driving my portfolio's returns?

## Key Features

- **Multiple Attribution Methods**
  - Brinson Attribution: Decompose returns into allocation, selection, and interaction effects
  - Factor Attribution: Analyze returns through factor exposures (e.g., value, momentum, quality)
  - Risk Attribution: Understand sources of portfolio risk
  - Transaction Cost Attribution: Analyze the impact of trading costs on performance

- **Factor Models**
  - Fama-French 3-Factor Model: Market, size, and value factors
  - BARRA Model: Industry-standard risk model
  - Axioma Model: Advanced risk and attribution model
  - Custom Factor Creation: Build and test your own factors

- **Advanced Risk Metrics**
  - Value at Risk (VaR)
  - Modified VaR
  - Conditional VaR
  - M3 and M4 Measures
  - Omega Ratio
  - Tail Ratio

- **Transaction Cost Analysis**
  - Fixed and Variable Costs
  - Market Impact Modeling
  - Implementation Shortfall
  - Trading Schedule Optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/diaabraham/Quantitative-Finance-Attribution-Analysis.git
cd quant-finance-attribution-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Real-World Examples

### 1. Portfolio Performance Attribution

```python
import pandas as pd
import numpy as np
from attribution import AttributionAnalysis

# Load your portfolio and benchmark data
portfolio_returns = pd.read_csv('data/portfolio_returns.csv', index_col=0, parse_dates=True)
benchmark_returns = pd.read_csv('data/benchmark_returns.csv', index_col=0, parse_dates=True)
sector_weights = pd.read_csv('data/sector_weights.csv', index_col=0)

# Initialize attribution analysis
analysis = AttributionAnalysis(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    sector_weights=sector_weights
)

# Run complete attribution analysis
results = analysis.run_attribution_analysis()

# Get detailed breakdown
allocation_effect = results['allocation']
selection_effect = results['selection']
interaction_effect = results['interaction']

# Visualize results
analysis.plot_attribution_results()
```

### 2. Factor Model Analysis

```python
from factor_models import FamaFrenchModel

# Load factor returns data
factor_returns = pd.read_csv('data/factor_returns.csv', index_col=0, parse_dates=True)

# Initialize Fama-French model
model = FamaFrenchModel(
    returns=portfolio_returns,
    factor_returns=factor_returns
)

# Estimate factor exposures
exposures = model.estimate_exposures()
print("Factor Exposures:")
print(exposures)

# Calculate factor contributions
contributions = model.calculate_contributions()
print("\nFactor Contributions:")
print(contributions)

# Calculate alpha
alpha = model.calculate_alpha()
print(f"\nPortfolio Alpha: {alpha:.2%}")
```

### 3. Risk Analysis

```python
from risk import risk_manager

# Initialize risk manager
risk_metrics = risk_manager.RiskMetrics(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns
)

# Calculate Value at Risk
var_95 = risk_metrics.calculate_var(confidence_level=0.95)
print(f"95% VaR: {var_95:.2%}")

# Calculate risk-adjusted returns
sharpe_ratio = risk_metrics.calculate_sharpe_ratio()
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Analyze drawdowns
drawdown_analysis = risk_metrics.analyze_drawdowns()
print("\nDrawdown Analysis:")
print(drawdown_analysis)
```

## Data Preparation and Formatting

### Required Data Formats

1. **Portfolio and Benchmark Returns**
```python
# Expected format for returns data
# CSV file: portfolio_returns.csv
# Date,Return
# 2023-01-01,0.0123
# 2023-01-02,-0.0045
# ...

# Load returns data
returns = pd.read_csv('data/portfolio_returns.csv', 
                     index_col=0, 
                     parse_dates=True)
```

2. **Sector Weights**
```python
# Expected format for sector weights
# CSV file: sector_weights.csv
# Sector,Portfolio_Weight,Benchmark_Weight
# Technology,0.25,0.20
# Healthcare,0.15,0.10
# ...

# Load sector weights
sector_weights = pd.read_csv('data/sector_weights.csv', 
                           index_col='Sector')
```

3. **Factor Returns**
```python
# Expected format for factor returns
# CSV file: factor_returns.csv
# Date,market,size,value,momentum,quality
# 2023-01-01,0.01,0.005,0.003,0.002,0.004
# ...

# Load factor returns
factor_returns = pd.read_csv('data/factor_returns.csv',
                           index_col=0,
                           parse_dates=True)
```

### Data Validation

```python
from attribution.utils import validate_returns_data

# Validate returns data
validate_returns_data(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    check_frequency=True,  # Ensure consistent frequency
    check_missing=True,    # Check for missing values
    check_duplicates=True  # Check for duplicate dates
)
```

### Detailed Data Validation

```python
from attribution.utils import (
    validate_returns_data,
    validate_sector_weights,
    validate_factor_returns
)

# 1. Returns Data Validation
returns_validation = validate_returns_data(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    check_frequency=True,    # Daily, weekly, monthly consistency
    check_missing=True,      # Handle missing values
    check_duplicates=True,   # Remove duplicate dates
    check_returns_range=True,# Validate return ranges
    max_return=0.5,         # Maximum allowed return (50%)
    min_return=-0.5         # Minimum allowed return (-50%)
)

# Example output:
# {
#     'frequency_consistent': True,
#     'missing_values': {'portfolio': 0, 'benchmark': 0},
#     'duplicate_dates': [],
#     'returns_in_range': True
# }

# 2. Sector Weights Validation
weights_validation = validate_sector_weights(
    portfolio_weights=sector_weights['Portfolio_Weight'],
    benchmark_weights=sector_weights['Benchmark_Weight'],
    check_sum=True,         # Weights sum to 1
    tolerance=0.001,        # Allow 0.1% tolerance
    check_negative=True     # No negative weights
)

# Example output:
# {
#     'portfolio_sum': 1.0,
#     'benchmark_sum': 1.0,
#     'negative_weights': [],
#     'validation_passed': True
# }

# 3. Factor Returns Validation
factor_validation = validate_factor_returns(
    factor_returns=factor_returns,
    check_correlation=True,  # Check for highly correlated factors
    correlation_threshold=0.9,
    check_stationarity=True  # Check for stationary returns
)

# Example output:
# {
#     'high_correlations': [],
#     'stationary_factors': ['market', 'size', 'value'],
#     'non_stationary_factors': []
# }
```

## Advanced Examples

### 1. Multi-Period Attribution Analysis

```python
from attribution import AttributionAnalysis
from attribution.utils import aggregate_attribution

# Initialize analysis
analysis = AttributionAnalysis(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    sector_weights=sector_weights
)

# Run attribution for each period
monthly_results = []
for period in portfolio_returns.resample('M').groups:
    period_returns = portfolio_returns[period]
    period_benchmark = benchmark_returns[period]
    results = analysis.run_attribution_analysis(
        period_returns,
        period_benchmark
    )
    monthly_results.append(results)

# Aggregate results
aggregated_results = aggregate_attribution(monthly_results)
print("\nAggregated Attribution Results:")
print(aggregated_results)
```

### 2. Custom Factor Model Creation

```python
from factor_models import FactorModel
from factor_models.utils import calculate_factor_ic

# Create custom factors
def create_quality_factor(data):
    # Example: Quality = ROE + Profit Margin + Asset Turnover
    quality = (data['roe'] + data['profit_margin'] + data['asset_turnover']) / 3
    return quality

# Calculate custom factor returns
quality_factor = create_quality_factor(fundamental_data)

# Initialize custom factor model
custom_model = FactorModel(
    returns=portfolio_returns,
    factor_returns=quality_factor
)

# Analyze factor effectiveness
ic = calculate_factor_ic(
    factor_returns=quality_factor,
    forward_returns=portfolio_returns.shift(-1)  # Next period returns
)
print(f"Quality Factor IC: {ic:.4f}")
```

### 3. Transaction Cost Analysis

```python
from attribution.transaction_cost import TransactionCostAttribution

# Load trade data
trades = pd.read_csv('data/trades.csv',
                    parse_dates=['timestamp'])

# Initialize transaction cost analysis
cost_analysis = TransactionCostAttribution(
    trades=trades,
    market_data=market_data,  # Price and volume data
    cost_parameters={
        'fixed_cost': 0.001,  # 10 bps fixed cost
        'variable_cost': 0.0005,  # 5 bps variable cost
        'market_impact': 0.0001  # 1 bp market impact
    }
)

# Calculate costs by category
costs = cost_analysis.analyze_costs()
print("\nTransaction Cost Breakdown:")
print(costs)

# Optimize trading schedule
optimal_schedule = cost_analysis.optimize_trading(
    target_quantity=10000,
    time_horizon=5,  # 5 days
    constraints={
        'max_daily_volume': 0.2,  # Max 20% of daily volume
        'min_execution': 0.1      # Min 10% execution per day
    }
)
```

### 4. Risk Attribution with Custom Factors

```python
from risk import risk_manager
from factor_models import BarraModel

# Initialize risk manager
risk_metrics = risk_manager.RiskMetrics(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns
)

# Calculate risk decomposition
risk_decomposition = risk_metrics.decompose_risk(
    factor_model=BarraModel(
        returns=portfolio_returns,
        factor_returns=factor_returns
    )
)

print("\nRisk Decomposition:")
print(risk_decomposition)

# Calculate marginal contributions to risk
marginal_risk = risk_metrics.calculate_marginal_risk()
print("\nMarginal Risk Contributions:")
print(marginal_risk)
```

### 5. Performance Attribution with Custom Groupings

```python
from attribution import AttributionAnalysis
from attribution.utils import create_custom_groupings

# Define custom groupings (e.g., by investment style)
style_groups = {
    'Growth': ['AAPL', 'MSFT', 'AMZN'],
    'Value': ['JPM', 'BAC', 'WFC'],
    'Defensive': ['PG', 'JNJ', 'KO']
}

# Create custom grouping weights
style_weights = create_custom_groupings(
    portfolio_holdings=portfolio_holdings,
    benchmark_holdings=benchmark_holdings,
    groupings=style_groups
)

# Run attribution with custom groupings
analysis = AttributionAnalysis(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    grouping_weights=style_weights
)

# Get style-based attribution
style_results = analysis.run_attribution_analysis()
print("\nStyle-Based Attribution:")
print(style_results)
```

## Testing

### Running Tests

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run all tests:
```bash
python -m pytest tests/
```

3. Run specific test categories:
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Specific test file
python -m pytest tests/integration/test_pipeline.py
```

4. Generate coverage report:
```bash
python -m pytest --cov=attribution tests/
```

### Writing Tests

When adding new features, follow these testing guidelines:

1. Create unit tests in `tests/unit/`:
```python
# tests/unit/test_brinson.py
def test_brinson_attribution():
    # Create test data
    portfolio_returns = pd.Series([0.01, 0.02, -0.01])
    benchmark_returns = pd.Series([0.005, 0.015, -0.005])
    
    # Initialize and run attribution
    analysis = AttributionAnalysis(portfolio_returns, benchmark_returns)
    results = analysis.run_attribution_analysis()
    
    # Assert expected results
    assert results['total_effect'] == pytest.approx(0.01, abs=1e-4)
```

2. Create integration tests in `tests/integration/`:
```python
# tests/integration/test_pipeline.py
def test_complete_analysis_pipeline():
    # Test the entire analysis pipeline
    # Load sample data
    # Run analysis
    # Verify results
```

## Project Structure

```
quant-finance-attribution-analysis/
├── attribution/
│   ├── __init__.py           # Main attribution module
│   ├── brinson.py            # Brinson attribution
│   ├── factor_attribution.py # Factor attribution
│   ├── risk_attribution.py   # Risk attribution
│   └── transaction_cost.py   # Transaction cost attribution
├── factor_models/
│   ├── __init__.py           # Factor models package
│   ├── models.py             # Factor model implementations
│   └── utils.py              # Factor model utilities
├── risk/
│   └── risk_manager.py       # Risk management and metrics
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── data/                     # Data directory
├── main.py                   # Main application entry point
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables
├── .gitattributes            # Git attributes
├── LICENSE                   # License file
└── README.md                 # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Academic papers on performance attribution
- Open-source quantitative finance libraries
- Financial industry best practices 