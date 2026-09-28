import statsmodels.api as sm
from sklearn.metrics import r2_score
import scipy.stats as stats


def compute_correlations(df, x_col, y_col):
    # Drop NaN values and align the data
    x_data = df[x_col].dropna()
    y_data = df[y_col].dropna()
    common_index = x_data.index.intersection(y_data.index)
    x_data = x_data.loc[common_index]
    y_data = y_data.loc[common_index]
    
    # Compute Spearman correlation
    spearman_corr = stats.spearmanr(x_data, y_data)[0]
    
    # Compute Pearson correlation
    pearson_corr = stats.pearsonr(x_data, y_data)[0]
    
    # Compute R² using statsmodels OLS
    X = sm.add_constant(x_data)  # Add intercept for regression
    model = sm.OLS(y_data, X).fit()
    r_squared = model.rsquared
    
    return spearman_corr, pearson_corr, r_squared

