# Quantitative Finance Attribution Analysis

A comprehensive Python library for performance attribution analysis in quantitative finance, supporting multiple attribution methods, factor models, and risk metrics.

## Features

- **Multiple Attribution Methods**
  - Brinson Attribution
  - Factor Attribution
  - Risk Attribution
  - Transaction Cost Attribution

- **Factor Models**
  - Fama-French 3-Factor Model
  - BARRA Model
  - Axioma Model
  - Custom Factor Creation

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
cd quant-finance-projects
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

## Project Structure

```
quant-finance-projects/
├── attribution/
│   ├── __init__.py           # Main attribution module
│   ├── brinson.py            # Brinson attribution
│   ├── factor_attribution.py # Factor attribution
│   ├── risk_attribution.py   # Risk attribution
│   └── transaction_cost.py   # Transaction cost attribution
├── factor_models/
│   ├── models.py             # Factor model implementations
│   └── utils.py              # Factor model utilities
├── risk/
│   ├── advanced_metrics.py   # Advanced risk metrics
│   └── utils.py              # Risk calculation utilities
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── examples/                 # Usage examples
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Usage

### Basic Attribution Analysis

```python
from attribution import AttributionAnalysis
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
portfolio_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
benchmark_returns = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)

# Initialize attribution analysis
analysis = AttributionAnalysis(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns
)

# Run complete attribution analysis
results = analysis.run_attribution_analysis()

# Calculate performance metrics
metrics = analysis.calculate_performance_metrics()

# Plot results
analysis.plot_attribution_results()
```

### Factor Attribution

```python
from factor_models.models import FamaFrenchModel, BarraModel, AxiomaModel

# Create factor returns
factor_returns = pd.DataFrame({
    'market': np.random.normal(0.0003, 0.01, len(dates)),
    'size': np.random.normal(0.0002, 0.008, len(dates)),
    'value': np.random.normal(0.0001, 0.006, len(dates))
}, index=dates)

# Initialize factor model
model = FamaFrenchModel(portfolio_returns, factor_returns)

# Estimate factor exposures
exposures = model.estimate_exposures()

# Calculate factor contributions
contributions = model.calculate_contributions()

# Analyze model fit
r_squared = model.calculate_r_squared()
```

### Risk Attribution

```python
from risk.advanced_metrics import AdvancedRiskMetrics

# Initialize risk metrics
risk_metrics = AdvancedRiskMetrics(portfolio_returns, benchmark_returns)

# Calculate VaR metrics
var_metrics = risk_metrics.calculate_var_metrics()

# Calculate risk-adjusted returns
risk_adjusted = risk_metrics.calculate_risk_adjusted_metrics()

# Analyze drawdowns
drawdown_metrics = risk_metrics.calculate_drawdown_metrics()
```

### Transaction Cost Analysis

```python
from attribution.transaction_cost import TransactionCostAttribution

# Create sample trade data
trades = pd.DataFrame({
    'timestamp': dates,
    'symbol': ['AAPL'] * len(dates),
    'quantity': np.random.randint(-100, 100, len(dates)),
    'price': np.random.normal(150, 10, len(dates)),
    'side': np.where(np.random.randint(0, 2, len(dates)) == 1, 'BUY', 'SELL')
})

# Initialize transaction cost analysis
cost_analysis = TransactionCostAttribution(trades)

# Calculate total costs
total_costs = cost_analysis.calculate_total_costs()

# Optimize trading schedule
schedule = cost_analysis.optimize_trading(target_quantity=1000, time_horizon=5)
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

### Test Structure

- **Unit Tests**: Test individual components in isolation
  - `test_brinson.py`: Brinson attribution tests
  - `test_factor_attribution.py`: Factor attribution tests
  - `test_risk_attribution.py`: Risk attribution tests
  - `test_transaction_cost.py`: Transaction cost tests

- **Integration Tests**: Test component interactions
  - `test_pipeline.py`: Complete pipeline tests
  - `test_factor_models.py`: Factor model integration tests
  - `test_risk_metrics.py`: Risk metrics integration tests
  - `test_transaction_costs.py`: Transaction cost integration tests

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