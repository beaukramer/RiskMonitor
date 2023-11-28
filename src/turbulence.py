import numpy as np
import pandas as pd

class Turbulence:

    def __init__(self, returns, window_size, quantile=0.95):
        self.returns = returns
        self.window_size = window_size
        self.quantile = quantile
        self.turbulence = self.calculate_turbulence()
        self.filtered_turbulence = self.filter_turbulence()

    def calculate_turbulence(self):
        turbulence = pd.DataFrame(index=self.returns.index[(self.window_size - 1):], columns=['Turbulence'])
        start = 0
        for end in range(len(turbulence)):
            sample_returns = self.returns.iloc[start:self.window_size + end, :]
            sample_means = sample_returns.mean()
            inv_cov = np.linalg.inv(np.cov(sample_returns, rowvar=False))
            current_returns = sample_returns.iloc[-1, :]
            delta = np.array((current_returns - sample_means)).reshape(1, -1)
            turbulence.iloc[end, 0] = delta.dot(inv_cov).dot(delta.transpose())[0][0]
            start += 1
        return turbulence

    def filter_turbulence(self):
        filter = self.turbulence.expanding(min_periods=10).quantile(self.quantile)
        filtered_turbulence = self.turbulence[self.turbulence > filter].fillna(0.)
        return filtered_turbulence

if __name__ == '__main__':
    from data_fetchers import get_yahoo_data
    from constants import *
    returns = get_yahoo_data(['SPY', 'TLT', 'LQD', 'HYG', 'GLD', 'VNQ'], start_date=START_DATE,
                             end_date=END_DATE).pct_change()
    returns.dropna(inplace=True)
    turbulence = Turbulence(returns=returns, window_size=WINDOW_SIZE)