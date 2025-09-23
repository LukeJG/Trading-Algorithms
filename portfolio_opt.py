from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import warnings
import requests
warnings.filterwarnings('ignore')






class DataPrep:
    def __init__(self):
        return self
    
    def compute_bb(group, key):
        bb = ta.bbands(close = group['adj close'], length=20)
        return bb.iloc[:,key]

    def compute_atr(group):
        atr = ta.atr(group['high'], group['low'], group['close'], length=14)
        return atr

    def compute_macd(group):
        macd_line = ta.macd(group, length=20).iloc[:, 0]
        return macd_line
    
    def calculate_returns(data):
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
    
    def fff_betas(data):
        factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

        factor_data.index = factor_data.index.to_timestamp()

        factor_data = factor_data.resample('M').last().div(100)

        factor_data.index.name = 'date'

        factor_data = factor_data.join(data['return_1m']).sort_index()

    
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

        return data
    

    def func(self, df):
        last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]

        data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                        df.unstack()[last_cols].resample('M').last().stack('ticker')],axis=1)).dropna()

    
    def 


class PortfolioOptimizer:
    def __init__(self):
        return self

    def analyze_portfolio_weights(weights_dict, method_used, optimization_df):
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
            exp_rets = expected_returns.mean_historical_return(price_data, frequency=252)
            port_mu = float((w.reindex(exp_rets.index).fillna(0) * exp_rets).sum())
            print(f"\nExpected portfolio return: {port_mu:.4f}")

    def optimize_weights(optimization_prices_df, lower_bound = 0, risk_free_rate=0.04, verbose=False):
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
            w = ef.clean_weights()
            if verbose: print("✅ Minimum volatility optimization successful")
            return w, "min_volatility"
        except OptimizationError:
            if verbose: print("Minimum volatility failed")

        #     ---- Strategy 4: Relaxed bounds + Min Vol ----    #
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

        #     ---- Strategy 5: Risk parity (inverse vol) ----    #
        try:
            vols = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            inv_vol = 1.0 / vols
            inv_vol /= inv_vol.sum()
            inv_vol = np.clip(inv_vol, lower_bound, max_weight)
            inv_vol /= inv_vol.sum()
            w = dict(zip(prices.columns, inv_vol))
            if verbose: print("✅ Risk parity (inverse volatility) successful")
            return w, "risk_parity"
        except Exception:
            if verbose: print("Risk parity failed")

        #    ---- Strategy 6: Return-weighted (shifted positive) ----    #
        try:
            adj = mu - float(mu.min()) + 0.01
            rw = (adj / adj.sum()).values  
            rw = np.clip(rw, lower_bound, max_weight)
            rw = rw / rw.sum()
            w = dict(zip(prices.columns, rw))
            if verbose: print("✅ Adjusted return weighting successful")
            return w, "return_weighted"
        except Exception:
            if verbose: print("Return weighting failed, using equal weights")

        #     ---- Final fallback ----    #
        eq = {t: 1 / num_assets for t in prices.columns}

        return eq, "equal_weights_final_fallback"

        

    def run_portfolio_optimization(self, stock_price_per_month_df, returns_dataframe, fixed_dates, lower_bound=0.012, verbose=False):
        portfolio_df = pd.DataFrame()
        optimization_results = []

        for i, start_date in enumerate(fixed_dates.keys()):       
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            
            cols = fixed_dates[start_date]

            adj_close_df = stock_price_per_month_df.xs('Adj Close', axis=1, level=0)
            
            optimization_start = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end   = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            optimization_prices_df = adj_close_df.loc[optimization_start:optimization_end, cols]

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

            rets_curr_month = returns_dataframe.loc[start_date, cols]
            rets_curr_month_df = rets_curr_month.stack().to_frame('return')
            rets_curr_month_df.index.names = ['date', 'ticker']

            rets_curr_month_df = rets_curr_month_df.join(weights, on='tcker').fillna(['wieght':0.0])
           
            rets_curr_month_df['weighted_return'] = rets_curr_month_df['return'] * rets_curr_month_df['weight']

            daily_returns = rets_curr_month_df.groupby('date')['weighted_return'].sum().to_frame('Strategy Return')

            portfolio_df = pd.concat([portfolio_df, daily_returns], axis=0)
        
        return portfolio_df, optimization_results





def main():
    pass
