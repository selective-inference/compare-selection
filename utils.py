from scipy.stats import norm as ndist
import numpy as np
import pandas as pd

import regreg.api as rr

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(selectiveInference); library(knockoff)') # libraries we will use

from selection.tests.instance import gaussian_instance

def randomize_signs(beta):
    return beta * np.random.binomial(1, 0.5, size=beta.shape)

class instance(object):

    def generate(self):
        raise NotImplementedError('abstract method should return (X,Y,beta)')

class equicor_instance(instance):

    name = 'Exchangeable'

    def __init__(self, n=500, p=200, s=20, rho=0.5, signal_fac=1.5):
        (self.n,
         self.p,
         self.s,
         self.rho, 
         self.signal_fac) = (n, p, s, rho, signal_fac)

    @property
    def params(self):
        if not hasattr(self, 'rho'):
            df = pd.DataFrame([[self.name, self.n, self.p, self.s, self.signal]],
                              columns=['name', 'n', 'p', 's', 'signal'])
        else:
            df = pd.DataFrame([[self.name, self.n, self.p, self.s, self.signal, self.rho]],
                              columns=['name', 'n', 'p', 's', 'signal', 'rho'])
        return df

    def generate(self):

        (n, p, s, rho, signal_fac) = (self.n,
                                      self.p,
                                      self.s,
                                      self.rho, 
                                      self.signal_fac)

        X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=s)[0]
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)
        l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)
        self.signal = signal_fac * np.max(l_theory)
        beta[:s] = self.signal
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        beta /= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

class jelena_instance(instance):

    name = 'Jelena'
    n = 1000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)

    @property
    def params(self):
        val = {'n':self.n,
               'p':self.p,
               's':self.s,
               'signal':self.signal,
               'name':self.name}
        return val

    def generate(self):

        n, p, s = self.n, self.p, self.s
        X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=s)[0]
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)
        beta[:s] = self.signal
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

class mixed_instance(equicor_instance):

    name = 'Mixed'

    def generate(self):

        (n, p, s, rho, signal_fac) = (self.n,
                                      self.p,
                                      self.s,
                                      self.rho, 
                                      self.signal_fac)

        X0 = gaussian_instance(n=n, p=p, equicorrelated=True, rho=0.25)[0]
        X1 = gaussian_instance(n=n, p=p, equicorrelated=False, rho=rho)[0]

        X = X0 + X1
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)
        l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)
        self.signal = signal_fac * np.max(l_theory)
        beta[:s] = self.signal
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        beta /= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

class indep_instance(equicor_instance):

    name = 'Independent'

    def __init__(self, n=500, p=200, s=20, signal_fac=1.5):
        (self.n,
         self.p,
         self.s,
         self.signal_fac) = (n, p, s, signal_fac)
        self.rho = 0.

class AR_instance(equicor_instance):

    name = 'AR'

    def generate(self):

        n, p, s, rho, signal_fac = self.n, self.p, self.s, self.rho, self.signal_fac
        X = gaussian_instance(n=n, p=p, equicorrelated=False, rho=rho)[0]
        l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)
        beta = np.zeros(p)
        self.signal = signal_fac * np.max(l_theory)
        beta[:s] = self.signal
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        beta /= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

def lagrange_vals(X, Y):
    n, p = X.shape
    numpy2ri.activate()
    rpy.r.assign('X', X)
    rpy.r.assign('Y', Y)
    rpy.r('X=as.matrix(X)')
    rpy.r('Y=as.numeric(Y)')
    rpy.r('G = cv.glmnet(X, Y, intercept=FALSE, standardize=FALSE)')
    rpy.r("L = G[['lambda.min']]")
    rpy.r("L1 = G[['lambda.1se']]")
    L = rpy.r('L')
    L1 = rpy.r('L1')
    numpy2ri.deactivate()
    l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)
    return L * np.sqrt(X.shape[0]), L1 * np.sqrt(X.shape[0]), l_theory


def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)





