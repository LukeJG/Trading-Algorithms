from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib
# Prefer a GUI backend for interactive plots. If PyQt5 is installed, Qt5Agg will provide a window.
try:
    matplotlib.use('Qt5Agg')
except Exception:
    # fall back to default if setting backend fails
    pass
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import warnings
import requests
from pathlib import Path
warnings.filterwarnings('ignore')






# Download / Load S&P Stock Data

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
}

response = requests.get(url, headers=headers)

response.raise_for_status()

sp500 = pd.read_html(response.text)[0]


sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)


symbol_list = sp500['Symbol'].unique().tolist()


end_date = '2025-09-16'
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=8)

df = yf.download(tickers=symbol_list,
                 start=start_date,
                 end=end_date, auto_adjust=False).stack()

df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()


# Calculate Features and Technical Indicators for Each Stock

# Garman-Klass Volatility
df['garman_klass_vol'] = (
    ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2
    - (2 * np.log(2) - 1) * (np.log(df['adj close']) - np.log(df['open'])) ** 2
)

# RSI
df['rsi'] = df.groupby('ticker')['adj close'].transform(lambda x: ta.rsi(x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                          
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                          
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.bbands(close=np.log1p(x), length=20).iloc[:,2])


# ATR
def compute_atr(group):
    atr = ta.atr(group['high'], group['low'], group['close'], length=14)
    return (atr - atr.mean()) / atr.std()

df['atr'] = df.groupby('ticker', group_keys=False).apply(compute_atr)

# MACD
def compute_macd(x):
    macd_line = ta.macd(x, length=20).iloc[:, 0]
    return (macd_line - macd_line.mean()) / macd_line.std()

df['macd'] = df.groupby('ticker')['adj close'].transform(compute_macd)

# Dollar Volume
df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6



# Aggregate to monthly level and filter top 150 most liquid stocks for each month


last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]

# Unstack once and reuse to reduce repeated expensive operations
df_unstacked = df.unstack('ticker')

data = (
    pd.concat([
        df_unstacked['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
        df_unstacked[last_cols].resample('M').last().stack('ticker')],
        axis=1
    )
).dropna()

data['dollar_volume'] = (
    data['dollar_volume'].unstack('ticker')
        .rolling(5 * 12, min_periods=12)
        .mean()
        .stack()
)
data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
# Keep top 199 by liquidity (rank < 200) and drop the helper 'dollar_vol_rank'
data = data[data['dollar_vol_rank'] < 200].drop(columns=['dollar_vol_rank'], axis=1)


# Calculate Monthly Returns for different time horizens as features

def calculate_returns(df):
    outlier_cutoff = 0.005

    lags = [1,2,3,6,9,12]

    for lag in lags:
      df[f'return_{lag}m'] = (df['adj close']
                            .pct_change(lag)
                            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                  upper=x.quantile(1-outlier_cutoff)))
                            .add(1)
                            .pow(1/lag)
                            .sub(1))

    return df

data = data.groupby('ticker', group_keys=False).apply(calculate_returns).dropna()



# Download Fama-French Factors / Rolling Factor Betas

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

factor_data = factor_data.join(data['return_1m']).sort_index()


observations = factor_data.groupby('ticker').size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]


betas = (factor_data.groupby('ticker', group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                      window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params.drop('const', axis=1)))


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.drop(['adj close','dollar_volume'], axis=1)

data = data.dropna()


def scale_group(group):
    scaler = StandardScaler()
    numeric_cols = group.columns.difference(['cluster'])
    group[numeric_cols] = scaler.fit_transform(group[numeric_cols])
    return group

df_scaled = data.groupby('date', group_keys=False).apply(scale_group)


def get_initial_centroids(group, target_rsi_values):
    # convert targets into z-scores for this month’s RSI distribution
    rsi_mean = group['rsi'].mean()
    rsi_std = group['rsi'].std()
    scaled_targets = [(t - rsi_mean) / rsi_std for t in target_rsi_values]

    centroids = np.zeros((len(scaled_targets), group.shape[1]))  # match feature count
    centroids[:, group.columns.get_loc('rsi')] = scaled_targets
    return centroids



TARGET_RSI_RAW = [30, 45, 55, 70]  # raw RSI thresholds

def get_clusters(month_df: pd.DataFrame) -> pd.DataFrame:
    # 1) choose numeric feature set (exclude existing 'cluster' if present)
    numeric_cols = month_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')

    X = month_df[numeric_cols].copy()

    # 2) fit cross-sectional scaler for THIS month and transform
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=numeric_cols
    )

    # 3) convert raw RSI targets -> THIS month's z-scores
    #    using the scaler stats for the RSI column
    rsi_col = 'rsi'
    if rsi_col not in numeric_cols:
        raise ValueError(f"'{rsi_col}' must be in columns to seed centroids.")

    rsi_idx = numeric_cols.index(rsi_col)
    rsi_mean = scaler.mean_[rsi_idx]
    rsi_std  = scaler.scale_[rsi_idx]

    if np.isclose(rsi_std, 0.0):
        # degenerate case: no dispersion in RSI this month
        rsi_targets_z = [0.0] * len(TARGET_RSI_RAW)
    else:
        rsi_targets_z = [(t - rsi_mean) / rsi_std for t in TARGET_RSI_RAW]

    # 4) build centroid matrix (n_clusters x n_features), default 0 (mean in z-space)
    n_features = len(numeric_cols)
    initial_centroids = np.zeros((len(rsi_targets_z), n_features))
    initial_centroids[:, rsi_idx] = rsi_targets_z

    # (optional) nudge more features than RSI, e.g. high/low 1m return:
    # initial_centroids[0, numeric_cols.index('return_1m')] = -1.0   # oversold + weak momentum
    # initial_centroids[-1, numeric_cols.index('return_1m')] =  1.0  # overbought + strong momentum

    # 5) run KMeans on scaled data with explicit init
    km = KMeans(
        n_clusters=len(TARGET_RSI_RAW),
        init=initial_centroids,
        n_init=1,            # must be 1 when passing explicit centroids
        random_state=0
    )
    labels = km.fit_predict(X_scaled)

    # 6) return original month_df plus cluster labels (don’t overwrite your scaled features unless you want to)
    out = month_df.copy()
    out['cluster'] = labels
    return out

# usage: group by month (date) and apply
data = (
    data.dropna()
        .groupby('date', group_keys=False)
        .apply(get_clusters)
)



# Choose Cluster 

filtered_df = data[data['cluster']==3].copy()  # HERE

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index+pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1], auto_adjust=False)




returns_dataframe = np.log(new_df['Adj Close']).diff()

def robust_optimize_weights(prices, lower_bound=0.0, risk_free_rate=0.04, verbose=False):
    """
    Robust portfolio optimization with multiple fallback strategies
    
    Parameters:
    - prices: DataFrame of asset prices
    - lower_bound: minimum weight per asset
    - risk_free_rate: risk-free rate for Sharpe ratio
    - verbose: print debug information
    
    Returns:
    - dict: asset weights
    - str: method used for optimization
    """
    
    if len(prices) < 5:  # Need minimum data
        if verbose:
            print("⚠️ Insufficient data points, using equal weights")
        return {ticker: 1 / prices.shape[1] for ticker in prices.columns}, "equal_weights_insufficient_data"
    
    try:
        # Calculate returns and covariance
        returns = expected_returns.mean_historical_return(prices, frequency=252)
        cov = risk_models.sample_cov(prices, frequency=252)
        
        num_assets = prices.shape[1]
        
        if verbose:
            print(f"Expected returns range: {returns.min():.4f} to {returns.max():.4f}")
            print(f"Risk-free rate: {risk_free_rate:.4f}")
        
        # Dynamic upper bound
        max_weight = min(0.15, 1.0 / num_assets * 3)  # Allow slightly higher concentration
        
        # Strategy 1: Max Sharpe with original risk-free rate
        try:
            if max(returns) > risk_free_rate:
                ef = EfficientFrontier(
                    expected_returns=returns,
                    cov_matrix=cov,
                    weight_bounds=(lower_bound, max_weight),
                    solver='ECOS'  # Try ECOS first, often more reliable
                )
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()
                if verbose:
                    print("✅ Max Sharpe optimization successful")
                return weights, "max_sharpe"
            else:
                raise ValueError("No assets exceed risk-free rate")
                
        except (OptimizationError, ValueError) as e:
            if verbose:
                print(f"Max Sharpe failed: {e}")
            
            # Strategy 2: Lower risk-free rate
            try:
                lower_rf = min(risk_free_rate * 0.5, max(returns) * 0.8)
                if max(returns) > lower_rf:
                    ef = EfficientFrontier(
                        expected_returns=returns,
                        cov_matrix=cov,
                        weight_bounds=(lower_bound, max_weight),
                        solver='ECOS'
                    )
                    ef.max_sharpe(risk_free_rate=lower_rf)
                    weights = ef.clean_weights()
                    if verbose:
                        print(f"✅ Max Sharpe with reduced risk-free rate ({lower_rf:.4f}) successful")
                    return weights, "max_sharpe_reduced_rf"
                else:
                    raise ValueError("Still no assets exceed reduced risk-free rate")
                    
            except (OptimizationError, ValueError):
                if verbose:
                    print("Reduced risk-free rate also failed")
                
                # Strategy 3: Minimum Volatility
                try:
                    ef = EfficientFrontier(
                        expected_returns=returns,
                        cov_matrix=cov,
                        weight_bounds=(lower_bound, max_weight),
                        solver='ECOS'
                    )
                    ef.min_volatility()
                    weights = ef.clean_weights()
                    if verbose:
                        print("✅ Minimum volatility optimization successful")
                    return weights, "min_volatility"
                    
                except OptimizationError:
                    if verbose:
                        print("Minimum volatility failed")
                    
                    # Strategy 4: Relaxed bounds
                    try:
                        relaxed_lower = max(0.0, lower_bound * 0.5)
                        relaxed_upper = min(1.0, max_weight * 1.5)
                        
                        ef = EfficientFrontier(
                            expected_returns=returns,
                            cov_matrix=cov,
                            weight_bounds=(relaxed_lower, relaxed_upper),
                            solver='SCS'  # Try different solver
                        )
                        ef.min_volatility()
                        weights = ef.clean_weights()
                        if verbose:
                            print("✅ Relaxed bounds minimum volatility successful")
                        return weights, "min_volatility_relaxed"
                        
                    except OptimizationError:
                        if verbose:
                            print("Relaxed bounds also failed")
                        
                        # Strategy 5: Risk parity (inverse volatility weighting)
                        try:
                            individual_vols = np.sqrt(np.diag(cov))
                            inv_vol_weights = 1 / individual_vols
                            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
                            
                            # Apply bounds
                            inv_vol_weights = np.maximum(inv_vol_weights, lower_bound)
                            inv_vol_weights = np.minimum(inv_vol_weights, max_weight)
                            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()  # Renormalize
                            
                            weights = dict(zip(prices.columns, inv_vol_weights))
                            if verbose:
                                print("✅ Risk parity (inverse volatility) successful")
                            return weights, "risk_parity"
                            
                        except Exception:
                            if verbose:
                                print("Risk parity failed")
                            
                            # Strategy 6: Return-weighted (avoid negative returns)
                            try:
                                # Shift returns to be positive
                                adjusted_returns = returns - returns.min() + 0.01
                                return_weights = adjusted_returns / adjusted_returns.sum()
                                
                                # Apply bounds
                                return_weights = np.maximum(return_weights, lower_bound)
                                return_weights = np.minimum(return_weights, max_weight)
                                return_weights = return_weights / return_weights.sum()
                                
                                weights = dict(zip(prices.columns, return_weights))
                                if verbose:
                                    print("✅ Adjusted return weighting successful")
                                return weights, "return_weighted"
                                
                            except Exception:
                                if verbose:
                                    print("Return weighting failed, using equal weights")
                                
                                # Final fallback: Equal weights
                                equal_weights = {ticker: 1 / prices.shape[1] for ticker in prices.columns}
                                return equal_weights, "equal_weights_final_fallback"
    
    except Exception as e:
        if verbose:
            print(f"Unexpected error: {e}")
        # Ultimate fallback
        equal_weights = {ticker: 1 / prices.shape[1] for ticker in prices.columns}
        return equal_weights, "equal_weights_error"

def analyze_portfolio_weights(weights_dict, method_used, returns_data=None):
    """
    Analyze and display portfolio weight characteristics
    """
    weights_series = pd.Series(weights_dict)
    
    print(f"\n📊 Portfolio Analysis (Method: {method_used})")
    print(f"Number of assets: {len(weights_series)}")
    print(f"Weight range: {weights_series.min():.4f} to {weights_series.max():.4f}")
    print(f"Weight concentration (max/min): {weights_series.max()/weights_series.min():.2f}")
    print(f"Effective number of stocks: {1/np.sum(weights_series**2):.2f}")
    
    # Show top 5 weights
    print(f"\nTop 5 weights:")
    print(weights_series.nlargest(5).round(4))
    
    if returns_data is not None:
        expected_returns = expected_returns.mean_historical_return(returns_data, frequency=252)
        portfolio_return = np.sum(weights_series * expected_returns)
        print(f"\nExpected portfolio return: {portfolio_return:.4f}")

# Updated main optimization function
def optimize_weights_robust(prices, lower_bound=0.0, verbose=False):
    """
    Main function that calls the robust optimizer
    """
    weights, method = robust_optimize_weights(
        prices=prices, 
        lower_bound=lower_bound,
        verbose=verbose
    )
    
    if verbose:
        analyze_portfolio_weights(weights, method, prices)
    
    return weights

# Your updated portfolio loop
def run_portfolio_optimization(new_df, returns_dataframe, fixed_dates, lower_bound=0.012, verbose=False):
    """
    Run the complete portfolio optimization loop with robust error handling
    """
    portfolio_df = pd.DataFrame()
    optimization_results = []  # Track which method was used for each period
    
    for i, start_date in enumerate(fixed_dates.keys()):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing period {i+1}/{len(fixed_dates)}: {start_date}")
        
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]
        
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        adj_close_df = new_df.xs('Adj Close', axis=1, level=0)
        optimization_df = adj_close_df.loc[optimization_start_date:optimization_end_date, cols]
        
        # Check data quality
        if len(optimization_df) < 20:  # Need minimum trading days
            print(f"⚠️ Insufficient data for {start_date}, skipping...")
            continue
            
        # Remove any columns with insufficient data
        valid_cols = optimization_df.dropna(axis=1, thresh=len(optimization_df)*0.8).columns
        if len(valid_cols) < len(cols):
            print(f"⚠️ Removed {len(cols) - len(valid_cols)} assets due to missing data")
            optimization_df = optimization_df[valid_cols]
            cols = valid_cols.tolist()
        
        if len(cols) == 0:
            print(f"⚠️ No valid assets for {start_date}, skipping...")
            continue
        
        # Get robust weights
        weights_dict, method_used = robust_optimize_weights(
            prices=optimization_df, 
            lower_bound=lower_bound,
            verbose=verbose
        )
        
        optimization_results.append({
            'date': start_date,
            'method': method_used,
            'n_assets': len(cols),
            'data_length': len(optimization_df)
        })
        
        # Convert weights dict to DataFrame
        weights = pd.Series(weights_dict).to_frame('weight')
        
        # Slice returns for this period
        temp_returns = returns_dataframe.loc[start_date:end_date, cols]
        
        # Reshape and merge returns and weights
        temp_df = temp_returns.stack().to_frame('return')
        temp_df.index.names = ['Date', 'Ticker']
        
        # Join weights
        temp_df = temp_df.join(weights, on='Ticker')
        temp_df['weight'] = temp_df['weight'].fillna(0)  # Handle missing weights
        
        # Compute weighted returns
        temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']
        
        # Sum by day to get portfolio return
        daily_returns = temp_df.groupby(level='Date')['weighted_return'].sum().to_frame('Strategy Return')
        
        # Append to full portfolio
        portfolio_df = pd.concat([portfolio_df, daily_returns], axis=0)
    
    # Remove duplicate index entries
    portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='first')]
    
    # Print optimization summary
    if verbose:
        print(f"\n{'='*50}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*50}")
        methods_used = pd.DataFrame(optimization_results)['method'].value_counts()
        print("Methods used:")
        for method, count in methods_used.items():
            print(f"  {method}: {count} periods")
    
    return portfolio_df, optimization_results

def main():
    # Usage example and plotting when run as a script
    portfolio_df, results = run_portfolio_optimization(
        new_df,
        returns_dataframe,
        fixed_dates,
        lower_bound=0.012,
        verbose=True,  # Set to False to reduce output
    )

    spy = yf.download(tickers='SPY', start='2017-01-01', end=dt.date.today(), auto_adjust=False)
    # Flatten the multi-level index of spy_ret to a single level ('Date')
    spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close': 'SPY Buy&Hold'}, axis=1).droplevel(level=1, axis=1)

    portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True)

    import matplotlib.ticker as mtick

    plt.style.use('ggplot')

    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1

    # Only attempt to plot if there is data
    if portfolio_cumulative_return.empty:
        print("No portfolio data available to plot. Check earlier processing steps.")
        return

    ax = portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16, 6))
    ax.set_title('Unsupervised Learning Trading Strategy Returns Over Time')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel('Return')

    plt.tight_layout()

    # Print current backend for diagnostics
    try:
        backend = plt.get_backend()
        print(f"Matplotlib backend: {backend}")
    except Exception:
        print("Unable to determine matplotlib backend")

    # Blocking show so GUI windows stay open when available
    try:
        plt.show(block=True)
        print('plt.show() returned successfully')
    except Exception as e:
        print(f"plt.show() failed or was suppressed: {e}")


if __name__ == '__main__':
    main()