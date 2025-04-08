import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union
import pandas as pd
from tabulate import tabulate

def plot_factor_contributions(contributions: pd.Series, 
                            title: str = 'Factor Contributions to Portfolio Returns',
                            figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot factor contributions as a bar chart.
    
    Args:
        contributions: Series of factor contributions
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Convert to percentage and sort
    contributions_pct = contributions * 100
    contributions_pct = contributions_pct.sort_values(ascending=True)
    
    # Create bar plot
    ax = contributions_pct.plot(kind='barh', color='skyblue')
    
    # Add value labels
    for i, v in enumerate(contributions_pct):
        ax.text(v, i, f'{v:.2f}%', va='center')
    
    # Customize plot
    plt.title(title, pad=20)
    plt.xlabel('Contribution (%)')
    plt.ylabel('Factor')
    plt.tight_layout()
    
    return plt.gcf()

def format_brinson_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format Brinson attribution results as a table.
    
    Args:
        results: Dictionary of attribution results
        
    Returns:
        Formatted table string
    """
    # Prepare data
    sectors = list(results['allocation'].keys())[:-1]  # Exclude 'Total'
    data = []
    
    for sector in sectors:
        row = [
            sector,
            f"{results['allocation'][sector]:.2%}",
            f"{results['selection'][sector]:.2%}",
            f"{results['interaction'][sector]:.2%}",
            f"{results['allocation'][sector] + results['selection'][sector] + results['interaction'][sector]:.2%}"
        ]
        data.append(row)
    
    # Add total row
    data.append([
        'Total',
        f"{results['allocation']['Total']:.2%}",
        f"{results['selection']['Total']:.2%}",
        f"{results['interaction']['Total']:.2%}",
        f"{results['total_effect']:.2%}"
    ])
    
    # Create table
    headers = ['Sector', 'Allocation', 'Selection', 'Interaction', 'Total']
    table = tabulate(data, headers=headers, tablefmt='grid')
    
    return table

def plot_attribution_breakdown(results: Dict[str, Dict[str, float]],
                             title: str = 'Sector Attribution Breakdown',
                             figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot Brinson attribution breakdown as a stacked bar chart.
    
    Args:
        results: Dictionary of attribution results
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Prepare data
    sectors = list(results['allocation'].keys())[:-1]  # Exclude 'Total'
    x = range(len(sectors))
    width = 0.8
    
    # Plot stacked bars
    plt.bar(x, [results['allocation'][s] * 100 for s in sectors], 
            width, label='Allocation', color='skyblue')
    plt.bar(x, [results['selection'][s] * 100 for s in sectors], 
            width, bottom=[results['allocation'][s] * 100 for s in sectors],
            label='Selection', color='lightgreen')
    plt.bar(x, [results['interaction'][s] * 100 for s in sectors], 
            width, bottom=[(results['allocation'][s] + results['selection'][s]) * 100 
                          for s in sectors],
            label='Interaction', color='salmon')
    
    # Customize plot
    plt.title(title, pad=20)
    plt.xlabel('Sector')
    plt.ylabel('Contribution (%)')
    plt.xticks(x, sectors, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_time_series_attribution(results: Dict[str, pd.Series],
                               title: str = 'Time Series Attribution',
                               figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot attribution effects over time.
    
    Args:
        results: Dictionary of time series attribution results
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'Allocation': results['allocation'] * 100,
        'Selection': results['selection'] * 100,
        'Interaction': results['interaction'] * 100
    })
    
    # Plot stacked area
    plt.stackplot(df.index, 
                 df['Allocation'], 
                 df['Selection'], 
                 df['Interaction'],
                 labels=['Allocation', 'Selection', 'Interaction'],
                 colors=['skyblue', 'lightgreen', 'salmon'])
    
    # Customize plot
    plt.title(title, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Contribution (%)')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def save_visualization(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Save visualization to file.
    
    Args:
        fig: matplotlib Figure object
        filename: Output filename
        dpi: Dots per inch for saved image
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig) 