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


# Bollinger Bands
def compute_bb(group, key):
  bb = ta.bbands(close=group['adj close'], length=20)
  return bb.iloc[:,key]

df['bb_low'] = df.groupby('ticker', group_keys=False).apply(compute_bb, 0)
df['bb_mid'] = df.groupby('ticker', group_keys=False).apply(compute_bb, 1)
df['bb_high'] = df.groupby('ticker', group_keys=False).apply(compute_bb, 2)

df['bb_position'] = ((df['adj close'] - df['bb_low']) / (df['bb_high'] - df['bb_low']))


# ATR
def compute_atr(group):
    atr = ta.atr(group['high'], group['low'], group['close'], length=14)
    return atr

df['atr'] = df.groupby('ticker', group_keys=False).apply(compute_atr)


# MACD
def compute_macd(group):
    macd_line = ta.macd(group, length=20).iloc[:, 0]
    return macd_line

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

data = data.drop(['adj close','dollar_volume','bb_low', 'bb_mid', 'bb_high'], axis=1)

data = data.dropna()




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



filtered_df = data[data['cluster']==3].copy()    # ----- Variable that user can change ----- #

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



def robust_optimize_weights(prices, lower_bound=0.0, risk_free_rate=0.04, verbose=False):
    try:
        if len(prices) < 5:
            if verbose:
                print("⚠️ Insufficient data points, using equal weights")
            return {ticker: 1 / prices.shape[1] for ticker in prices.columns}, "equal_weights_insufficient_data"

        # --- inputs ---
        num_assets = prices.shape[1]

        # Feasibility guard: sum of lower bounds must be ≤ 1
        if lower_bound * num_assets > 1.0:
            if verbose:
                print(f"⚠️ Lower bound {lower_bound:.4f} infeasible for {num_assets} assets; using 0.0")
            lower_bound = 0.0

        # Expected returns & covariance (use shrinkage for stability)
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        if verbose:
            print(f"Expected returns range: {mu.min():.4f} to {mu.max():.4f}")
            print(f"Risk-free rate: {risk_free_rate:.4f}")

        # Dynamic cap
        max_weight = min(0.15, 3.0 / num_assets)

        # ---- Strategy 1: Max Sharpe ----
        try:
            if mu.max() > risk_free_rate:
                ef = EfficientFrontier(mu, cov, weight_bounds=(lower_bound, max_weight))
                ef.max_sharpe(risk_free_rate=risk_free_rate)  # don't pass solver here
                w = ef.clean_weights()
                if verbose: print("✅ Max Sharpe optimization successful")
                return w, "max_sharpe"
            else:
                raise ValueError("No assets exceed risk-free rate")
        except (OptimizationError, ValueError) as e:
            if verbose: print(f"Max Sharpe failed: {e}")

        # ---- Strategy 2: Max Sharpe with reduced RF ----
        try:
            lower_rf = min(risk_free_rate * 0.5, float(mu.max()) * 0.8)
            if mu.max() > lower_rf:
                ef = EfficientFrontier(mu, cov, weight_bounds=(lower_bound, max_weight))
                ef.max_sharpe(risk_free_rate=lower_rf)
                w = ef.clean_weights()
                if verbose: print(f"✅ Max Sharpe with reduced RF ({lower_rf:.4f}) successful")
                return w, "max_sharpe_reduced_rf"
            else:
                raise ValueError("Still no assets exceed reduced RF")
        except (OptimizationError, ValueError):
            if verbose: print("Reduced RF also failed")

        # ---- Strategy 3: Min Vol ----
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(lower_bound, max_weight))
            ef.min_volatility()
            w = ef.clean_weights()
            if verbose: print("✅ Minimum volatility optimization successful")
            return w, "min_volatility"
        except OptimizationError:
            if verbose: print("Minimum volatility failed")

        # ---- Strategy 4: Relaxed bounds + Min Vol ----
        try:
            relaxed_lower = max(0.0, lower_bound * 0.5)
            relaxed_upper = min(1.0, max_weight * 1.5)
            ef = EfficientFrontier(mu, cov, weight_bounds=(relaxed_lower, relaxed_upper))
            ef.min_volatility()
            w = ef.clean_weights()
            if verbose: print("✅ Relaxed bounds minimum volatility successful")
            return w, "min_volatility_relaxed"
        except OptimizationError:
            if verbose: print("Relaxed bounds also failed")

        # ---- Strategy 5: Risk parity (inverse vol) ----
        try:
            vols = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            inv_vol = 1.0 / vols
            inv_vol /= inv_vol.sum()

            # apply bounds then renormalize
            inv_vol = np.clip(inv_vol, lower_bound, max_weight)
            inv_vol /= inv_vol.sum()
            w = dict(zip(prices.columns, inv_vol))
            if verbose: print("✅ Risk parity (inverse volatility) successful")
            return w, "risk_parity"
        except Exception:
            if verbose: print("Risk parity failed")

        # ---- Strategy 6: Return-weighted (shifted positive) ----
        try:
            # shifted positives
            adj = mu - float(mu.min()) + 0.01
            rw = (adj / adj.sum()).values  # numpy array
            rw = np.clip(rw, lower_bound, max_weight)
            rw = rw / rw.sum()
            w = dict(zip(prices.columns, rw))
            if verbose: print("✅ Adjusted return weighting successful")
            return w, "return_weighted"
        except Exception:
            if verbose: print("Return weighting failed, using equal weights")

        # ---- Final fallback ----
        eq = {t: 1 / num_assets for t in prices.columns}
        return eq, "equal_weights_final_fallback"

    except Exception as e:
        if verbose:
            print(f"Unexpected error: {e}")
        eq = {t: 1 / prices.shape[1] for t in prices.columns}
        return eq, "equal_weights_error"


def analyze_portfolio_weights(weights_dict, method_used, price_data=None):
    """
    Analyze portfolio weight characteristics (optional helper)
    """
    w = pd.Series(weights_dict).sort_values(ascending=False)
    print(f"\n📊 Portfolio Analysis (Method: {method_used})")
    print(f"Number of assets: {len(w)}")
    w_min = float(w.min())
    w_max = float(w.max())
    print(f"Weight range: {w_min:.4f} to {w_max:.4f}")
    denom = w_min if w_min > 1e-8 else 1e-8
    print(f"Weight concentration (max/min): {w_max/denom:.2f}")
    print(f"Effective number of stocks: {1/np.sum(w**2):.2f}")
    print("\nTop 5 weights:")
    print(w.head(5).round(4))

    if price_data is not None:
        exp_rets = expected_returns.mean_historical_return(price_data, frequency=252)
        port_mu = float((w.reindex(exp_rets.index).fillna(0) * exp_rets).sum())
        print(f"\nExpected portfolio return: {port_mu:.4f}")


def run_portfolio_optimization(new_df, returns_dataframe, fixed_dates, lower_bound=0.012, verbose=False):
    portfolio_df = pd.DataFrame()
    optimization_results = []

    for i, start_date in enumerate(fixed_dates.keys()):
        if verbose:
            print(f"\n{'='*50}\nProcessing period {i+1}/{len(fixed_dates)}: {start_date}")

        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]

        # price panel
        adj_close_df = new_df.xs('Adj Close', axis=1, level=0)

        # 12-month lookback (exclude current month for fit)
        optimization_start = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end   = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        optimization_df = adj_close_df.loc[optimization_start:optimization_end, cols]

        if len(optimization_df) < 20:
            if verbose: print(f"⚠️ Insufficient data for {start_date}, skipping...")
            continue

        valid_cols = optimization_df.dropna(axis=1, thresh=int(len(optimization_df)*0.8)).columns
        if len(valid_cols) < len(cols) and verbose:
            print(f"⚠️ Removed {len(cols) - len(valid_cols)} assets due to missing data")
        optimization_df = optimization_df[valid_cols]
        cols = valid_cols.tolist()
        if not cols:
            if verbose: print(f"⚠️ No valid assets for {start_date}, skipping...")
            continue

        # get weights
        weights_dict, method_used = robust_optimize_weights(
            prices=optimization_df,
            lower_bound=lower_bound,
            verbose=verbose
        )
        if verbose:
            analyze_portfolio_weights(weights_dict, method_used, optimization_df)

        optimization_results.append({
            'date': start_date,
            'method': method_used,
            'n_assets': len(cols),
            'data_length': len(optimization_df)
        })

        # daily returns for current month
        temp_returns = returns_dataframe.loc[start_date:end_date, cols]

        temp_df = temp_returns.stack().to_frame('return')
        temp_df.index.names = ['Date', 'Ticker']

        weights = pd.Series(weights_dict, name='weight')
        temp_df = temp_df.join(weights, on='Ticker').fillna({'weight': 0.0})
        temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']

        daily_returns = temp_df.groupby(level='Date')['weighted_return'].sum().to_frame('Strategy Return')
        portfolio_df = pd.concat([portfolio_df, daily_returns], axis=0)

    portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='first')].sort_index()

    if verbose and optimization_results:
        print(f"\n{'='*50}\nOPTIMIZATION SUMMARY\n{'='*50}")
        methods_used = pd.DataFrame(optimization_results)['method'].value_counts()
        print("Methods used:")
        for method, count in methods_used.items():
            print(f"  {method}: {count} periods")

    return portfolio_df, optimization_results


returns_dataframe = np.log(new_df['Adj Close']).diff()

portfolio_df, results = run_portfolio_optimization(
     new_df,
     returns_dataframe,
     fixed_dates,
     lower_bound=0.012,
     verbose=True  # Set to False to reduce output
 )


spy = yf.download(tickers='SPY',
                  start='2017-01-01',
                  end=dt.date.today(), auto_adjust=False)

# Flatten the multi-level index of spy_ret to a single level ('Date')
spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1).droplevel(level=1, axis=1)


portfolio_df = portfolio_df.merge(spy_ret,
                                  left_index=True,
                                  right_index=True)

portfolio_df


import matplotlib.ticker as mtick

plt.style.use('ggplot')

portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6))

plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()


import matplotlib.pyplot as plt

# Calculate cumulative returns
gross_rets = (1 + portfolio_df['Strategy Return']).cumprod()
bench_rets = (1 + portfolio_df['SPY Buy&Hold']).cumprod()

# Calculate investment values starting with $1000
initial_investment = 1000000
rets = gross_rets * initial_investment
b_rets = bench_rets * initial_investment

# Calculate ROI
roi = ((rets.iloc[-1] - initial_investment) / initial_investment) * 100
b_roi = ((b_rets.iloc[-1] - initial_investment) / initial_investment) * 100

# Print ROI
print(f"Strategy ROI: {roi:.2f}%")
print(f"Benchmark ROI: {b_roi:.2f}%")

# Plot both investment values
plt.figure(figsize=(10, 6))
plt.plot(rets, label=f'Strategy (${roi:.2f}% ROI)', linewidth=2)
plt.plot(b_rets, label=f'SPY Buy & Hold (${b_roi:.2f}% ROI)', linestyle='--', linewidth=2)

plt.title('Value of $1000 Investment Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
