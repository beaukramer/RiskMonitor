import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from denoise_covariance import DenoiseCovariance

class AbsorptionRatio:

    def __init__(self, returns, window_size):
        self.returns = returns
        self.window_size = window_size
        self.absorption_ratio_raw = self.calculate_absorption_ratio()
        self.absorption_ratio_standardized = self.standardize()

    def estimate_cov(self, ret, bwidth=.01):
        cov = ret.cov()
        q = ret.shape[0] / ret.shape[1]
        return DenoiseCovariance(cov=cov, q=q, bwidth=bwidth).deNoiseCov()


    def standardize(self,short_window=21, long_window=252):
        short_mean = self.absorption_ratio_raw.rolling(short_window).mean()
        long_mean = self.absorption_ratio_raw.rolling(long_window).mean()
        std = self.absorption_ratio_raw.rolling(long_window).std()
        return ((short_mean - long_mean) / std).dropna()


    # calculate_systemic_risk() → systemic risk series
    def calculate_absorption_ratio(self):

        n_components = int(round(0.2 * self.returns.shape[1]))
        absorption_ratio_raw = pd.DataFrame(index=self.returns.index[(self.window_size - 1):], columns=['Absorption_Ratio'])
        start = 0
        for end in range(len(absorption_ratio_raw)):
            subsample_returns = self.returns.iloc[start:self.window_size + end, :]
            sigma = np.cov(subsample_returns, rowvar=False)
            # sigma = self.estimate_cov(subsample_returns)
            absorption_ratio = sum(PCA(n_components=n_components).fit(sigma).explained_variance_ratio_)
            absorption_ratio_raw.iloc[end, 0] = absorption_ratio
            start += 1
        return absorption_ratio_raw


if __name__ == '__main__':
    from data_fetchers import get_yahoo_data
    from constants import *
    returns = get_yahoo_data(['SPY', 'TLT', 'LQD', 'HYG', 'GLD', 'VNQ'], start_date=START_DATE,
                             end_date=END_DATE).pct_change()
    returns.dropna(inplace=True)
    absorption_ratio_raw = AbsorptionRatio(returns, window_size=WINDOW_SIZE)