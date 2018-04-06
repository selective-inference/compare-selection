import numpy as np
import pandas as pd
from selection.tests.instance import gaussian_instance
from instances import data_instance

class jelena_instance(data_instance):

    signature = None
    instance_name = 'Jelena'
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
        df = pd.DataFrame([[self.instance_name, str(self.__class__), self.n, self.p, self.s, self.signal * np.sqrt(self.n)]],
                          columns=['Instance', 'class', 'n', 'p', 's', 'signal'])
        return df

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = np.identity(self.p)
        return self._sigma

jelena_instance.register()

class jelena_instance_flip(jelena_instance):

    signature = None
    instance_name = 'Jelena, n=5000'
    n = 5000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)

jelena_instance_flip.register()

class jelena_instance_flipmore(jelena_instance):

    signature = None
    instance_name = 'Jelena, n=10000'
    n = 10000
    p = 2000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)


jelena_instance_flipmore.register()

class jelena_instance_AR(data_instance):

    signature = None
    instance_name = 'Jelena AR(0.5)'
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
        df = pd.DataFrame([[self.instance_name, str(self.__class__), self.n, self.p, self.s, self.rho, self.signal * np.sqrt(self.n)]],
                          columns=['Instance', 'class', 'n', 'p', 's', 'rho', 'signal'])
        return df

    @property
    def sigma(self):
        if not hasattr(self, "_sigma"):
            self._sigma = self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._sigma

jelena_instance_AR.register()

class jelena_instance_AR75(jelena_instance_AR):

    instance_name = 'Jelena AR(0.75)'
    rho = 0.75

jelena_instance_AR75.register()
