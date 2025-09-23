from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import warnings
import requests
warnings.filterwarnings('ignore')


class DataCollector:
    def __init__(self):
        pass

    def fetch_data(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers)

        response.raise_for_status()

        sp500 = pd.read_html(response.text)[0]


        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')


        symbol_list = sp500['Symbol'].unique().tolist()


        end_date = '2025-09-16'
        start_date = pd.to_datetime(end_date) - pd.DateOffset(years=10)

        df = yf.download(tickers=symbol_list,
                        start=start_date,
                        end=end_date, auto_adjust=False).stack()

        df.index.names = ['date', 'ticker']
        df.columns = df.columns.str.lower()
        return df



class DataPrep:
    def __init__(self):
        pass
    
    def compute_bb(self, group, key):
        bb = ta.bbands(close = group['adj close'], length=20)
        return bb.iloc[:,key]

    def compute_atr(self, group):
        atr = ta.atr(group['high'], group['low'], group['close'], length=14)
        return atr

    def compute_macd(self, group):
        macd_line = ta.macd(group, length=20).iloc[:, 0]
        return macd_line
    
    def calculate_returns(self, data):
        outlier_cutoff = 0.005

        lags = [1,2,3,6,9,12]

        for lag in lags:
            data[f'return_{lag}m'] = (data['adj close']
                                    .pct_change(lag)
                                    .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                        upper=x.quantile(1-outlier_cutoff)))
                                    .add(1)
                                    .pow(1/lag)
                                    .sub(1))
        return data
    
    def fff_betas(self, data):
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
        return data

    
    def calculate_features(self, df):
        df['garman_klass_vol'] = (
            ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2
            - (2 * np.log(2) - 1) * (np.log(df['adj close']) - np.log(df['open'])) ** 2)
        
        df['rsi'] = df.groupby('ticker')['adj close'].transform(lambda x: ta.rsi(x, length=20))

        df['bb_low'] = df.groupby('ticker', group_keys=False).apply(self.compute_bb, 0)
        df['bb_mid'] = df.groupby('ticker', group_keys=False).apply(self.compute_bb, 1)
        df['bb_high'] = df.groupby('ticker', group_keys=False).apply(self.compute_bb, 2)
        df['bb_position'] = ((df['adj close'] - df['bb_low']) / (df['bb_high'] - df['bb_low']))
        df.drop(['bb_low', 'bb_mid', 'bb_high'], axis=1, inplace=True)

        df['atr'] = df.groupby('ticker', group_keys=False).apply(self.compute_atr)

        df['macd'] = df.groupby('ticker')['adj close'].transform(self.compute_macd)

        df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
        
        last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
        data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                        df.unstack()[last_cols].resample('M').last().stack('ticker')],axis=1)).dropna()

        data['dollar_volume'] = data['dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack()
        data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
        data = data[data['dollar_vol_rank'] < 150].drop(columns=['dollar_vol_rank'],axis=1)

        data = data.groupby('ticker', group_keys=False).apply(self.calculate_returns).dropna()

        data = self.fff_betas(data)

        return data
    


class Clustering:
    def __init__(self):
        pass

    def get_clusters(self, month_df: pd.DataFrame):
        TARGET_RSI_RAW = [30, 45, 55, 70] 
        numeric_cols = month_df.select_dtypes(include=[np.number]).columns.tolist()

        X = month_df[numeric_cols].copy()

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=numeric_cols
        )

        rsi_idx = numeric_cols.index('rsi')
        rsi_mean = scaler.mean_[rsi_idx]
        rsi_std  = scaler.scale_[rsi_idx]

        if np.isclose(rsi_std, 0.0):
            # degenerate case: no dispersion in RSI this month
            rsi_targets_z = [0.0] * len(TARGET_RSI_RAW)
        else:
            rsi_targets_z = [(t - rsi_mean) / rsi_std for t in TARGET_RSI_RAW]


        n_features = len(numeric_cols)
        initial_centroids = np.zeros((len(rsi_targets_z), n_features))
        initial_centroids[:, rsi_idx] = rsi_targets_z

        km = KMeans(
            n_clusters=len(TARGET_RSI_RAW),
            init=initial_centroids,
            n_init=1,            
            random_state=0
        )
        labels = km.fit_predict(X_scaled)

        output = month_df.copy()
        output['cluster'] = labels
        return output
    
    def run_clustering(self, data, cluster):
        data = (data.dropna().groupby('date', group_keys=False).apply(self.get_clusters))

        filtered_df = data[data['cluster']==cluster].copy()

        filtered_df = filtered_df.reset_index(level=1)

        filtered_df.index = filtered_df.index+pd.DateOffset(1)

        filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

        dates = filtered_df.index.get_level_values('date').unique().tolist()

        fixed_dates = {}

        for d in dates:

            fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

        stocks = data.index.get_level_values('ticker').unique().tolist()

        stock_price_per_month_df = yf.download(tickers=stocks,
                            start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                            end=data.index.get_level_values('date').unique()[-1], auto_adjust=False)

        return fixed_dates, stock_price_per_month_df



class ClusterResult:
    def __init__(self, cluster_id, portfolio_df, results):
        self.cluster_id = cluster_id
        self.portfolio_df = portfolio_df
        self.results = results

    def summary(self):
        rets = self.portfolio_df["Strategy Return"]
        np.exp(np.log1p(rets).cumsum())-1
        return pd.DataFrame({"cluster f'{self.cluster_id}'": rets})



class PortfolioOptimizer:
    def __init__(self):
        self.data_collector = DataCollector()
        self.data_prep = DataPrep()
        self.clustering = Clustering()

    def analyze_portfolio_weights(self, weights_dict, method_used, optimization_df):
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
            exp_rets = expected_returns.mean_historical_return(optimization_df, frequency=252)
            port_mu = float((w.reindex(exp_rets.index).fillna(0) * exp_rets).sum())
            print(f"\nExpected portfolio return: {port_mu:.4f}")

    def optimize_weights(self, optimization_prices_df, lower_bound = 0, risk_free_rate=0.04, verbose=False):
        num_assets = optimization_prices_df.shape[1]

        mu = expected_returns.mean_historical_return(optimization_prices_df, frequency=252) 
        cov = risk_models.CovarianceShrinkage(optimization_prices_df).ledoit_wolf()

        max_weight = min(0.15, 3.0 / num_assets)

        #   ---- Strategy 1: Max Sharpe Ratio ----   #
        try: 
            if mu.max() > risk_free_rate:
                ef = EfficientFrontier(mu, cov, weight_bounds=(lower_bound, max_weight))
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()
                if verbose:
                    print("Optimization successful using Max Sharpe Ratio.")
                    return weights, 'Max Sharpe Ratio'
                else:
                    raise ValueError("No assets exceed the risk-free rate.")           
        except (OptimizationError, ValueError) as e:
            if verbose: print(f"Max Sharpe failed: {e}")
        
        #   ---- Strategy 2: Max Sharpe with Reduced RF ----   #
        try:
            lower_rf = min(risk_free_rate * 0.5, float(mu.max()) * 0.8)
            if mu.max() > lower_rf:
                ef = EfficientFrontier(mu, cov, weight_bounds=(lower_bound, max_weight))
                ef.max_sharpe(risk_free_rate=lower_rf)
                weights = ef.clean_weights()
                if verbose: print(f"Max Sharpe with reduced RF ({lower_rf:.4f}) successful")
                return weights, "max_sharpe_reduced_rf"
            else:
                raise ValueError("Still no assets exceed reduced RF")
        except (OptimizationError, ValueError):
            if verbose: print("Reduced RF also failed")
        
        #     ---- Strategy 3: Min Vol ----    #
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(lower_bound, max_weight))
            ef.min_volatility()
            weights = ef.clean_weights()
            if verbose: print("✅ Minimum volatility optimization successful")
            return weights, "min_volatility"
        except OptimizationError:
            if verbose: print("Minimum volatility failed")

        #     ---- Strategy 4: Relaxed bounds + Min Vol ----    #
        try:
            relaxed_lower = max(0.0, lower_bound * 0.5)
            relaxed_upper = min(1.0, max_weight * 1.5)
            ef = EfficientFrontier(mu, cov, weight_bounds=(relaxed_lower, relaxed_upper))
            ef.min_volatility()
            weights = ef.clean_weights()
            if verbose: print("✅ Relaxed bounds minimum volatility successful")
            return weights, "min_volatility_relaxed"
        except OptimizationError:
            if verbose: print("Relaxed bounds also failed")

        #     ---- Strategy 5: Risk parity (inverse vol) ----    #
        try:
            vols = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            inv_vol = 1.0 / vols
            inv_vol /= inv_vol.sum()
            inv_vol = np.clip(inv_vol, lower_bound, max_weight)
            inv_vol /= inv_vol.sum()
            weights = dict(zip(optimization_prices_df.columns, inv_vol))
            if verbose: print("✅ Risk parity (inverse volatility) successful")
            return weights, "risk_parity"
        except Exception:
            if verbose: print("Risk parity failed")

        #    ---- Strategy 6: Return-weighted (shifted positive) ----    #
        try:
            adj = mu - float(mu.min()) + 0.01
            rw = (adj / adj.sum()).values  
            rw = np.clip(rw, lower_bound, max_weight)
            rw = rw / rw.sum()
            weights = dict(zip(optimization_prices_df.columns, rw))
            if verbose: print("✅ Adjusted return weighting successful")
            return weights, "return_weighted"
        except Exception:
            if verbose: print("Return weighting failed, using equal weights")

        #     ---- Final fallback ----    #
        eq = {t: 1 / num_assets for t in optimization_prices_df.columns}

        return eq, "equal_weights_final_fallback"

    def portfolio_optimization(self, stock_price_per_month_df, returns_dataframe, fixed_dates, lower_bound=0.012, verbose=False):
        portfolio_df = pd.DataFrame()
        optimization_results = []

        for i, start_date in enumerate(fixed_dates.keys()):       
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            
            cols = fixed_dates[start_date]

            adj_close_df = stock_price_per_month_df.xs('Adj Close', axis=1, level=0)
            
            optimization_start = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end   = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            optimization_prices_df = adj_close_df.loc[optimization_start:optimization_end, cols]

            valid_cols = optimization_prices_df.dropna(axis=1, thresh=int(len(optimization_prices_df)*0.8)).columns
            optimization_prices_df = optimization_prices_df[valid_cols]
            cols = valid_cols.tolist()

            weights_dict, method_used = self.optimize_weights(optimization_prices_df, lower_bound, verbose)
            if verbose:
                self.analyze_portfolio_weights(weights_dict, method_used, optimization_prices_df)

            optimization_results.append({
                'date': start_date,
                'method': method_used,
                'n_assets': len(cols),
                'data_length': len(optimization_prices_df)
            })

            weights = pd.Series(weights_dict, name='weight')

            rets_curr_month = returns_dataframe.loc[pd.to_datetime(start_date):end_date, cols]
            rets_curr_month_df = rets_curr_month.stack().to_frame('return')
            rets_curr_month_df.index.names = ['date', 'ticker']

            rets_curr_month_df = rets_curr_month_df.join(weights, on='ticker').fillna({'weight':0.0})
           
            rets_curr_month_df['weighted_return'] = rets_curr_month_df['return'] * rets_curr_month_df['weight']

            daily_returns = rets_curr_month_df.groupby('date')['weighted_return'].sum().to_frame('Strategy Return')

            portfolio_df = pd.concat([portfolio_df, daily_returns], axis=0)
        
        portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='first')].sort_index()

        return portfolio_df, optimization_results
    
    def run_optimization(self, data, cluster):
        fixed_dates, stock_price_per_month_df = self.clustering.run_clustering(data, cluster)
    
        returns_dataframe = np.log(stock_price_per_month_df['Adj Close']).diff()

        portfolio_df, results = self.portfolio_optimization(
            stock_price_per_month_df,
            returns_dataframe,
            fixed_dates,
            lower_bound=0.012,
            verbose=False
        )
        #return portfolio_df, results

        #return ClusterResult(cluster, portfolio_df, results)
        rets = portfolio_df["Strategy Return"]
        #cum_rets = np.exp(np.log1p(rets).cumsum())-1
        #print(pd.DataFrame({f"cluster {cluster}": cum_rets}).head(10))
        return pd.DataFrame({f"cluster {cluster}": rets})
    
    def run_all_clusters(self, verbose=False):
        data = self.data_collector.fetch_data()
        data = self.data_prep.calculate_features(data)
        clusters = [0,1,2,3]  
        all_results = pd.DataFrame()
        for c in clusters:
            print(f"\n🚀 Running Optimization for Cluster {c}")
            result = self.run_optimization(data, c)
            all_results = all_results.join(result, how='outer')
        return all_results



def main(run_all):
    portfolio_optimizer = PortfolioOptimizer()
    #portfolio_df, results = portfolio_optimizer.run_optimization(cluster)

    if run_all:
        all_results = portfolio_optimizer.run_all_clusters(verbose=True)
    
    print(all_results.head(10))
    
    spy = yf.download(tickers='SPY',
                  start='2017-01-01',
                  end=dt.date.today(), auto_adjust=False)

    spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1).droplevel(level=1, axis=1)

    # cluster_choice represents what the user wants to visualize
    # cluster_choice not initialized yet
    portfolio_df = portfolio_df[cluster_choice].merge(spy_ret,left_index=True,right_index=True)

    plt.style.use('ggplot')

    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

    portfolio_cumulative_return[:'2024-09-29'].plot(figsize=(16,6))

    plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.ylabel('Return')

    plt.show()"""


if __name__ == "__main__":
    main(run_all=True)



