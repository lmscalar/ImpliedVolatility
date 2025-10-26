# Changelog

## 2025-10-26 - Code Modernization Update

### Changes Made
1. **Created requirements.txt** - Added comprehensive dependency management
   - Updated to modern versions of core libraries (pandas 2.x, numpy 1.x, scipy 1.x)
   - Added visualization libraries (matplotlib, seaborn)
   - Added statsmodels for regression analysis
   - Replaced deprecated dependencies with modern alternatives

2. **Fixed Import Issues**
   - Made `splinter` import optional (browser automation is rarely needed)
   - Made `xlwings` import optional (Excel integration not required for core functionality)
   - Added graceful fallbacks when optional dependencies are missing

3. **Updated pandas Compatibility**
   - Added `engine='openpyxl'` to `pd.read_excel()` calls for modern pandas
   - Removed redundant `.loc[:]` syntax that causes warnings
   - Code now compatible with pandas 2.x

4. **Python 3.11 Compatibility**
   - All imports and syntax verified for Python 3.11+
   - Removed deprecated warnings

### Installation
```bash
pip install -r requirements.txt
```

### Optional Dependencies
- **splinter/selenium**: Only needed for web scraping (alternative to FTP data source)
- **xlwings**: Only needed for viewing data in Excel (alternative: use print/pandas display)

### Usage
The main functionality remains unchanged:
```bash
python implied_vol.py
```

When prompted, enter a Natural Gas contract (e.g., 'NGQ20')
