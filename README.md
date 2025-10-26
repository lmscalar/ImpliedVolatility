# ImpliedVolatility
Implied Volatility surface generator using OLS fit for NYMEX Natural Gas options.

## Installation

1. Install Python 3.11 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the program:

```bash
python implied_vol.py
```

When prompted, enter a Natural Gas contract (e.g., 'NGQ20'):
```
prompt> Enter Natural Gas Contract (e.g. NGQ20): NGQ20
```

## Features

- Scrapes option prices from CME website or FTP
- Downloads and caches prices to Python pickle files
- Calculates implied volatilities using Black76 model for futures options
- Generates volatility surface plots using Ordinary Least Squares (OLS) regression
- Supports both call and put options
- Filters options by moneyness for better surface fitting

## Requirements

See `requirements.txt` for full list. Core dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- statsmodels >= 0.14.0

## Recent Updates (2025-10-26)

- Updated to Python 3.11+ compatibility
- Modernized dependencies (pandas 2.x, numpy 1.x)
- Made optional dependencies gracefully handled (splinter, xlwings)
- Fixed deprecated pandas syntax
- Added comprehensive requirements.txt

See `CHANGELOG.md` for detailed changes.

