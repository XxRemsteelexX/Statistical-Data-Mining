# Statistical Data Mining - Housing Price Prediction

## Project Overview
This repository contains a statistical data mining project focused on predicting housing prices using multiple linear regression. The analysis examines various housing features to build a predictive model for real estate valuation.

## Dataset Description
The project uses a comprehensive housing dataset with 7,000 property records containing 22 features:

### Target Variable
- **Price**: Property sale price (range: $85,000 - $1,046,676)

### Key Features Used in Model
- **SquareFootage**: Living area in square feet (550 - 2,875 sq ft)
- **NumBathrooms**: Number of bathrooms (1.0 - 5.5)
- **NumBedrooms**: Number of bedrooms (1 - 7)
- **RenovationQuality**: Quality of renovations (0.01 - 10.0 scale)
- **DistanceToCityCenter**: Distance in miles (0 - 65.2 miles)  
- **AgeOfHome**: Age of property in years (0.01 - 178.68 years)

### Additional Features in Original Dataset
- BackyardSpace, CrimeRate, SchoolRating, EmploymentRate, PropertyTaxRate
- LocalAmenities, TransportAccess, Fireplace, HouseColor, Garage
- Floors, Windows, PreviousSalePrice, IsLuxury

## Project Structure
```
Statistical-Data-Mining/
├── housing_analysis.ipynb     # main analysis notebook
├── training_dataset.csv       # processed training data (5,577 records)
├── test_dataset.csv           # test data (1,395 records)
└── README.md                  # project documentation
```

## Analysis Workflow

### 1. Data Preprocessing
- Loaded and explored 7,000 housing records
- Identified and handled data quality issues:
  - Removed 28 records with negative previous sale prices
  - Corrected 50 records with negative window counts
  - Capped bathroom counts at reasonable maximum (5.5)
- Final cleaned dataset: 6,972 records

### 2. Feature Selection
Features selected based on correlation analysis and domain knowledge:
- Strong positive correlations: SquareFootage (0.55), NumBathrooms (0.46), NumBedrooms (0.46), RenovationQuality (0.48)
- Negative correlations: DistanceToCityCenter (-0.21), AgeOfHome (-0.14)
- Note: SchoolRating showed unexpected negative coefficient despite positive correlation (0.38) and was excluded from final model

### 3. Model Development
- **Method**: Ordinary Least Squares (OLS) regression
- **Data Split**: 80% training (5,577 records), 20% testing (1,395 records)
- **Standardization**: Applied to all predictors for coefficient comparison

### 4. Model Performance

#### Final Model Statistics
- **R-squared**: 0.600 (60% variance explained)
- **Adjusted R-squared**: 0.599
- **F-statistic**: 1,390 (p < 0.001)
- **Training RMSE**: $95,305.06
- **Test RMSE**: $93,791.23

#### Regression Equation (Standardized)
```
Price = $309,598.26 
        + ($62,477.41 × SquareFootage_std)
        + ($48,877.37 × NumBathrooms_std)
        + ($53,505.78 × NumBedrooms_std)
        + ($15,639.28 × RenovationQuality_std)
        + (-$4,647.77 × DistanceToCityCenter_std)
        + (-$4,527.67 × AgeOfHome_std)
```

## Key Findings

### Significant Predictors (all p < 0.05)
1. **SquareFootage**: Strongest predictor (t = 44.77)
   - Each additional square foot adds ~$147.77 to price
2. **NumBedrooms**: High impact (t = 39.91)
   - Each bedroom adds ~$53,340
3. **NumBathrooms**: Substantial effect (t = 35.96)
   - Each bathroom adds ~$51,370
4. **RenovationQuality**: Moderate positive effect (t = 10.34)
   - Each quality point adds ~$8,507
5. **DistanceToCityCenter**: Negative impact (t = -3.51)
   - Each mile from center reduces price by ~$398
6. **AgeOfHome**: Depreciation factor (t = -3.49)
   - Each year reduces price by ~$148

### Model Diagnostics
- **Durbin-Watson**: 1.999 (no autocorrelation)
- **Condition Number**: 1.92 (no multicollinearity issues after removing SchoolRating)
- Some positive skewness (0.64) suggesting slight non-normality in residuals

## Requirements
```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
statsmodels >= 0.12.0
```

## Installation & Usage

### Setting Up Environment
```bash
# create virtual environment
python3 -m venv sdm_env

# activate environment
source sdm_env/bin/activate

# install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
```

### Running the Analysis
```bash
# start jupyter notebook
jupyter notebook housing_analysis.ipynb
```

## Files Description
- **housing_analysis.ipynb**: Complete analysis including data cleaning, EDA, model building, and visualization
- **training_dataset.csv**: Cleaned training data with selected features
- **test_dataset.csv**: Hold-out test set for model validation

## Future Improvements
1. Investigate non-linear relationships and interaction terms
2. Apply advanced regression techniques (Ridge, Lasso, Elastic Net)
3. Explore ensemble methods (Random Forest, Gradient Boosting)
4. Address residual skewness through transformations
5. Conduct deeper analysis of outliers and influential points
6. Implement cross-validation for more robust performance estimates

## Author
Statistical Data Mining Project

## License
This project is for educational purposes.