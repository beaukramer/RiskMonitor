import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

class DenoiseCovariance:

    def __init__(self, cov, q, bwidth):
        self.cov = cov
        self.q = q
        self.bwidth = bwidth

    def mpPDF(self, var, q, pts):
        # Marcenko-Pasture pdf
        # q = T/N
        if len(np.shape(var)) == 1:
            var = float(var[0])
        eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eVal = np.linspace(eMin, eMax, pts)
        pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** .5
        pdf = pd.Series(pdf, index=eVal)
        return pdf

    def getPCA(self,matrix):
        # Get eVal, eVec from a Hermitian Matrix
        eVal, eVec = np.linalg.eigh(matrix)
        indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
        eVal, eVec = eVal[indices], eVec[:, indices]
        eVal = np.diagflat(eVal)
        return eVal, eVec

    def fitKDE(self, obs, bWidth=.25, kernel='gaussian', x=None):
        # Fit kernel to a series of obs, and derive the prob of obvs
        # x is the array of values on which the fit KDE will be evaluated
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
        if x is None:
            x = np.unique(obs).reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        logProb = kde.score_samples(x)  # log(density)
        pdf = pd.Series(np.exp(logProb), index=x.flatten())
        return pdf

    def errPDFs(self, var, eVal, q, bWidth, pts=1000):
        # Fit error
        pdf0 = self.mpPDF(var, q, pts)  # theoretical pdf
        pdf1 = self.fitKDE(eVal, bWidth, x=pdf0.index.values)  # emprical pdf
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse

    def findMaxEval(self, eVal, q, bWidth):
        # Find max random eVal by fitting Mercenko's dist
        out = minimize(lambda *x: self.errPDFs(*x), .5, args=(eVal, q, bWidth), bounds=((1E-5, 1 - 1E-5),))
        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        eMax = var * (1 + (1. / q) ** .5) ** 2
        return eMax, var

    def denoisedCorr(self, eVal, eVec, nFacts):
        # Remove noise from corr by fixing random eigenvalues
        eVal_ = np.diag(eVal).copy()
        eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
        eVal_ = np.diag(eVal_)
        corr1 = np.dot(eVec, eVal_).dot(eVec.T)
        corr1 = self.cov2corr(corr1)
        return corr1

    def cov2corr(self, cov):
        # Derive the correlation matrix from a covariance matrix
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
        return corr

    def corr2cov(self, corr, std):
        cov = corr * np.outer(std, std)
        return cov

    def deNoiseCov(self):
        corr0 = self.cov2corr(self.cov)
        eVal0, eVec0 = self.getPCA(corr0)
        eMax0, var0 = self.findMaxEval(np.diag(eVal0), self.q, self.bwidth)
        nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
        corr1 = self.denoisedCorr(eVal0, eVec0, nFacts0)
        cov1 = self.corr2cov(corr1, np.diag(self.cov) ** .5)
        return cov1