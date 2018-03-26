from scipy.stats import norm as ndist
import numpy as np
import regreg.api as rr

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(selectiveInference); library(knockoff)') # libraries we will use

from selection.tests.instance import gaussian_instance

def randomize_signs(beta):
    return beta * np.random.binomial(1, 0.5, size=beta.shape)

def equicor_instance(n=500, p=200, s=20, rho=0.5, signal_fac=1.5):

    X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=s)[0]
    X /= np.sqrt((X**2).sum(0))[None, :] 
    l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)

    beta = np.zeros(p)
    beta[:s] = signal_fac * np.max(l_theory)
    beta = randomize_signs(beta)

    X *= np.sqrt(n)
    beta /= np.sqrt(n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    return X, Y, beta, l_theory

def jelena_instance(n=1000, p=2000, s=30, rho=0., signal_fac=None):

    X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=s)[0]
    X /= np.sqrt((X**2).sum(0))[None, :] 
    l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)

    beta = np.zeros(p)
    beta[:s] = np.sqrt(2 * np.log(p))
    beta = randomize_signs(beta)

    X *= np.sqrt(n)
    beta /= np.sqrt(n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    return X, Y, beta, l_theory

def jelena_instance2(n=1000, p=2000, s=30, rho=0., signal_fac=None):

    X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=s)[0]
    X /= np.sqrt((X**2).sum(0))[None, :] 
    l_theory = 0.8 * np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)

    beta = np.zeros(p)
    beta[:s] = np.sqrt(2 * np.log(p))
    beta = randomize_signs(beta)

    X *= np.sqrt(n)
    beta /= np.sqrt(n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    return X, Y, beta, l_theory

def mixed_instance(n=800, p=300, s=20, rho=0.5, signal_fac=1.5):

    X0 = gaussian_instance(n=n, p=p, equicorrelated=True, rho=0.25)[0]
    X1 = gaussian_instance(n=n, p=p, equicorrelated=False, rho=rho)[0]

    X = X0 + X1
    X /= np.sqrt((X**2).sum(0))[None, :] 
    l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)

    beta = np.zeros(p)
    beta[:s] = signal_fac * np.max(l_theory)
    np.random.shuffle(beta)
    beta = randomize_signs(beta)

    X *= np.sqrt(n)
    beta /= np.sqrt(n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    return X, Y, beta, l_theory

def indep_instance(n=800, p=300, s=20, rho=0.5, signal_fac=1.5):

    X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=0.)[0]
    X /= np.sqrt((X**2).sum(0))[None, :] 
    l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)

    beta = np.zeros(p)
    beta[:s] = signal_fac * np.max(l_theory)
    np.random.shuffle(beta)
    beta = randomize_signs(beta)

    X *= np.sqrt(n)
    beta /= np.sqrt(n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    return X, Y, beta, l_theory

def AR_instance(rho=0.75, signal_fac=1.5, n=300, p=100, s=20):

    X = gaussian_instance(n=n, p=p, equicorrelated=False, rho=rho)[0]
    l_theory = np.fabs(X.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p)

    beta = np.zeros(p)
    beta[:s] = signal_fac * np.max(l_theory)
    np.random.shuffle(beta)
    beta = randomize_signs(beta)

    X *= np.sqrt(n)
    beta /= np.sqrt(n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    return X, Y, beta, l_theory

def lagrange_min(X, Y):
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
    return L * np.sqrt(X.shape[0]), L1 * np.sqrt(X.shape[0])


def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)





