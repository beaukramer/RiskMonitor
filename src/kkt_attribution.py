import numpy as np
import pandas as pd
from scipy.spatial import distance
from data_fetchers import get_fred_data, get_shiller_data
from constants import *

class KKT_Attribution:

    def __init__(self, df, econ_vars, recession_var):
        self.econ_vars = econ_vars
        self.df = df
        self.means_r = self.df.groupby(recession_var)[self.econ_vars].mean()
        self.cov_r = self.df.groupby(recession_var)[self.econ_vars].cov()
        self.regimes = self.means_r.index.values.tolist()
        self.calculate_mahalanobis_distance_by_regime()
        self.calculate_statistical_likelihood()
        self.calculate_sensitivities()
        self.calculate_variable_importance()

    def calculate_mahalanobis_distance_by_regime(self):
        for r in self.regimes:
            self.df['Distance' + str(r)] = self.df.apply(lambda x: distance.mahalanobis(x[self.econ_vars],
                                                                                        self.means_r.loc[r],
                                                                                        self.cov_r.loc[r]), axis=1)

    def calculate_statistical_likelihood(self):
        for r in self.regimes:
            self.df['Likelihood' + str(r)] = self.df.apply(lambda x:
                                                           np.linalg.det(2 * np.pi * self.cov_r.loc[r]) ** (
                                                               -0.5) * np.exp(-x['Distance' + str(r)] / 2),
                                                           axis=1)
        total_likelihoods = self.df[self.df.columns[self.df.columns.str.contains('Likelihood')]].sum(axis=1)
        for r in self.regimes:
            self.df['Probability' + str(r)] = self.df['Likelihood' + str(r)] / total_likelihoods

    def calculate_sensitivities(self):
        for r in self.regimes:
            self.df['PartialDeriv' + str(r)] = self.df.apply(
                lambda x: np.dot(np.linalg.inv(self.cov_r.loc[r]), (x[self.econ_vars] - self.means_r.loc[r]).T), axis=1)
        for r in self.regimes:
            not_r = [x for x in self.regimes if x != r]
            self.df['Sensitivity' + str(r)] = (1 / len(self.regimes)) * abs(self.df['Probability' + str(r)].T * sum(
                self.df['Probability' + str(z)] * self.df['PartialDeriv' + str(z)] for z in not_r) - (1 - self.df[
                'Probability' + str(r)] * self.df['PartialDeriv' + str(r)]))
        self.df['Sensitivity'] = self.df[self.df.columns[self.df.columns.str.contains('Sensitivity')]].sum(axis=1)

    def calculate_variable_importance(self):
        sigma = self.df[self.econ_vars].std()
        self.variable_importance = pd.DataFrame(index=self.df.index, columns=self.econ_vars)
        for dt, y in self.df['Sensitivity'].iteritems():
            self.variable_importance.loc[dt, :] = (y.flatten() * sigma) / sum(abs(y.flatten() * sigma))

if __name__ =="__main__":
    kkt_data = get_fred_data(KKT_BUSINESS_CYCLE_INDICATOR_SERIES, start_date=None, end_date=END_DATE)
    kkt_data[[INDUSTRIAL_PRODCUTION, NONFARM_PAYROLLS]] = kkt_data[
        [INDUSTRIAL_PRODCUTION, NONFARM_PAYROLLS]].pct_change(12).dropna()
    kkt_data[TEN_YEAR_TREASURY_YIELD_MINUS_FF].ffill(inplace=True)
    kkt_data[TEN_YEAR_TREASURY_YIELD_MINUS_FF] = kkt_data[TEN_YEAR_TREASURY_YIELD_MINUS_FF].rolling(
        window=12).mean()
    kkt_data.dropna(subset=[INDUSTRIAL_PRODCUTION, NONFARM_PAYROLLS, TEN_YEAR_TREASURY_YIELD_MINUS_FF], how='any',
                      inplace=True)
    shiller_data = get_shiller_data(SHILLER_URL, SHILLER_SHEET_NAME, SHILLER_SKIP_ROW)
    sp500 = shiller_data['S&P'].copy()
    sp500 = sp500.pct_change(12).dropna()
    kkt_data = pd.merge(kkt_data, sp500, how='outer', left_index=True, right_index=True)
    kkt_data.ffill(inplace=True)
    kkt_data.dropna(how='any', inplace=True)
    kkt_attribution = KKT_Attribution(kkt_data, econ_vars=ECON_VARIABLES, recession_var=NBER_RECESSION)