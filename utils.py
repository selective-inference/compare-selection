from scipy.stats import norm as ndist
import numpy as np
import pandas as pd

import regreg.api as rr

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('suppressMessages(library(selectiveInference)); suppressMessages(library(knockoff))') # R libraries we will use

from selection.tests.instance import gaussian_instance

def randomize_signs(beta):
    return beta * (2 * np.random.binomial(1, 0.5, size=beta.shape) - 1)

instances = {}

class instance(object):

    signature = None

    def generate(self):
        raise NotImplementedError('abstract method should return (X,Y,beta)')

    @classmethod
    def register(cls):
        instances[cls.__name__] = cls

class equicor_instance(instance):

    name = 'Exchangeable'
    signature = ('n', 'p', 's', 'rho')
    signal_fac = 1.5 # used in set_l_theory to set signal strength
                     # as a multiple of l_theory

    def __init__(self, n=500, p=200, s=20, rho=0.5):

        (self.n,
         self.p,
         self.s,
         self.rho) = (n, p, s, rho)

        self.set_l_theory()

    def set_l_theory(self, factor=3):
        """
        Used for setting lambda for a signal size
        """
        nf = 0
        X = []
        self.signal = 0. # will be overwritten below but needed to generate
        self.fixed_l_theory = 0
        while True:
            X.append(self.generate()[0])

            n, p = X[0].shape
            nf += n

            if nf > p * factor:
                break
        X = np.vstack(X)
        X /= np.sqrt((X**2).sum(0))[None, :]

        self.fixed_l_theory = np.fabs(X.T.dot(np.random.standard_normal((nf, 500)))).max(1).mean()

        self.signal = self.signal_fac * self.fixed_l_theory

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = np.ones((self.p, self.p)) * self.rho + (1 - self.rho) * np.identity(self.p)
        return self._sigma

    @property
    def params(self):
        df = pd.DataFrame([[self.name, self.signal] + [getattr(self, arg) for arg in self.signature]],
                          columns=['name', 'signal'] + [arg for arg in self.signature])
        return df

    def generate(self):

        (n, p, s, rho) = (self.n,
                          self.p,
                          self.s,
                          self.rho)

        X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=s)[0]
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)

        beta[:s] = self.signal
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        beta /= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta
equicor_instance.register()

class jelena_instance(instance):

    signature = None
    name = 'Jelena'
    n = 1000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)

    def generate(self):

        n, p, s = self.n, self.p, self.s
        X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=0., s=s)[0]
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)
        beta[:s] = self.signal
        beta = randomize_signs(beta)
        np.random.shuffle(beta)

        X *= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

    @property
    def params(self):
        df = pd.DataFrame([[self.name, str(self.__class__), self.n, self.p, self.s, self.signal * np.sqrt(self.n)]],
                          columns=['name', 'class', 'n', 'p', 's', 'signal'])
        return df

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = np.identity(self.p)
        return self._sigma

jelena_instance.register()

class jelena_instance_flip(jelena_instance):

    signature = None
    name = 'Jelena, n=5000'
    n = 5000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)

jelena_instance_flip.register()

class jelena_instance_flipmore(jelena_instance):

    signature = None
    name = 'Jelena, n=10000'
    n = 10000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)


jelena_instance_flipmore.register()

class jelena_instance_AR(instance):

    signature = None
    name = 'Jelena AR(0.5)'
    n = 1000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)
    rho = 0.5

    def generate(self):

        n, p, s = self.n, self.p, self.s
        X = gaussian_instance(n=n, p=p, equicorrelated=False, rho=self.rho, s=s)[0]
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)
        beta[:s] = self.signal
        beta = randomize_signs(beta)
        np.random.shuffle(beta)

        X *= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

    @property
    def params(self):
        df = pd.DataFrame([[self.name, str(self.__class__), self.n, self.p, self.s, self.rho, self.signal * np.sqrt(self.n)]],
                          columns=['name', 'class', 'n', 'p', 's', 'rho', 'signal'])
        return df

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._sigma

jelena_instance_AR.register()

class jelena_instance_AR75(jelena_instance_AR):

    name = 'Jelena AR(0.75)'
    rho = 0.75

jelena_instance_AR75.register()

class mixed_instance(equicor_instance):

    signature = ('n', 'p', 's', 'rho', 'equicor_rho', 'AR_weight')
    name = 'Mixed'

    def __init__(self, 
                 n=500, 
                 p=200, 
                 s=20, 
                 rho=0.5, 
                 equicor_rho=0.25,
                 AR_weight=0.5):

        (self.n,
         self.p,
         self.s,
         self.rho,
         self.equicor_rho,
         self.AR_weight) = (n, 
                            p, 
                            s, 
                            rho,
                            equicor_rho,
                            AR_weight)

        self.set_l_theory()

    def generate(self):

        (n, p, s, rho) = (self.n,
                          self.p,
                          self.s,
                          self.rho)

        X_equi = gaussian_instance(n=n, 
                                   p=p, 
                                   equicorrelated=True, 
                                   rho=self.equicor_rho)[0]
        X_AR = gaussian_instance(n=n, 
                                 p=p, 
                                 equicorrelated=False, 
                                 rho=rho)[0]

        X = np.sqrt(self.AR_weight) * X_AR + np.sqrt(1 - self.AR_weight) * X_equi
        X /= np.sqrt((X**2).sum(0))[None, :] 

        beta = np.zeros(p)

        beta[:s] = self.signal
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        beta /= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = 0.5 * (self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p))) + 
                                 np.ones((self.p, self.p)) * self.equicor_rho + (1 - self.equicor_rho) * np.identity(self.p))
        return self._sigma

mixed_instance.register()

class indep_instance(equicor_instance):

    signature = ('n', 'p', 's')
    name = 'Independent'

    def __init__(self, n=500, p=200, s=20):
        equicor_instance.__init__(self,
                                  n=n,
                                  p=p,
                                  s=s,
                                  rho=0.)

indep_instance.register()

class AR_instance(equicor_instance):

    signature = ('n', 'p', 's', 'rho')
    name = 'AR'

    def generate(self):

        n, p, s, rho = self.n, self.p, self.s, self.rho
        X = gaussian_instance(n=n, p=p, equicorrelated=False, rho=rho)[0]

        beta = np.zeros(p)
        beta[:s] = self.signal
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        beta /= np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n)

        return X, Y, beta

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._sigma


AR_instance.register()

def lagrange_vals(X, Y, run_CV=True):
    n, p = X.shape

    Xn = X / np.sqrt((X**2).sum(0))[None, :] 

    l_theory = np.fabs(Xn.T.dot(np.random.standard_normal((n, 500)))).max(1).mean() * np.ones(p) * np.std(Y)

    if run_CV:
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
        return L * np.sqrt(X.shape[0]), L1 * np.sqrt(X.shape[0]), l_theory 
    else:
        return None, None, l_theory

def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)





