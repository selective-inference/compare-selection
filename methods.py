import numpy as np
import regreg.api as rr

from selection.truncated.gaussian import truncated_gaussian_old as TG
from selection.algorithms.lasso import lasso
from selection.algorithms.sqrt_lasso import choose_lambda
from selection.randomized.lasso import highdim

from utils import BHfilter

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

# Knockoff selection

class generic_method(object):

    q = 0.2
    method_name = 'Generic method'

    def __init__(self, X, Y, l_theory, l_min, l_1se):
        (self.X,
         self.Y,
         self.l_theory,
         self.l_min,
         self.l_1se) = (X,
                        Y,
                        l_theory,
                        l_min,
                        l_1se)

    def select(self):
        raise NotImplementedError('abstract method')

class knockoffs_mf(generic_method):

    method_name = 'ModelX Knockoffs  (full)'

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), None
        except:
            return None, None

class knockoffs_orig(generic_method):
    method_name = 'Candes & Barber (full)'

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, statistic=stat.glmnet_lambdadiff, fdr=q, knockoffs=create.fixed)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), None
        except:
            return None, None

class knockoffs_fixed(generic_method):

    method_name = 'Knockoffs fixed (full)'

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', X)
            rpy.r.assign('Y', Y)
            rpy.r.assign('q', q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=create.fixed)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), None
        except:
            return None, None


# Liu, Markovic, Tibs selection
# put this into library!

def solve_problem(Qbeta_bar, Q, lagrange, initial=None):
    p = Qbeta_bar.shape[0]
    loss = rr.quadratic_loss((p,), Q=Q, quadratic=rr.identity_quadratic(0, 
                                                                        0, 
                                                                        -Qbeta_bar, 
                                                                        0))
    lagrange = np.asarray(lagrange)
    if lagrange.shape in [(), (1,)]:
        lagrange = np.ones(p) * lagrange
    pen = rr.weighted_l1norm(lagrange, lagrange=1.)
    problem = rr.simple_problem(loss, pen)
    if initial is not None:
        problem.coefs[:] = initial
    soln = problem.solve(tol=1.e-12, min_its=10)
    return soln

def truncation_interval(Qbeta_bar, Q, Qi_jj, j, beta_barj, lagrange):
    if lagrange[j] != 0:
        lagrange_cp = lagrange.copy()
    lagrange_cp[j] = np.inf
    restricted_soln = solve_problem(Qbeta_bar, Q, lagrange_cp)

    p = Qbeta_bar.shape[0]
    I = np.identity(p)
    nuisance = Qbeta_bar - I[:,j] / Qi_jj * beta_barj
    
    center = nuisance[j] - Q[j].dot(restricted_soln)
    upper = (lagrange[j] - center) * Qi_jj
    lower = (-lagrange[j] - center) * Qi_jj

    return lower, upper

class liu_theory(generic_method):

    method_name = "Liu (full)"            

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)

        self.lagrange = l_theory * np.std(Y)

    def select(self):
        X, Y, lagrange = self.X, self.Y, self.lagrange
        n, p = X.shape
        X = X / np.sqrt(n)

        Q = X.T.dot(X)
        Qi = np.linalg.inv(Q)

        pvalues = []

        Qbeta_bar = X.T.dot(Y)
        beta_bar = np.linalg.pinv(X).dot(Y)
        sigma = np.linalg.norm(Y - X.dot(beta_bar)) / np.sqrt(n - p)

        soln = solve_problem(Qbeta_bar, Q, lagrange)
        active_set = E = np.nonzero(soln)[0]

        QiE = Qi[E][:,E]
        beta_barE = beta_bar[E]
        for j in range(len(active_set)):
            idx = active_set[j]
            lower, upper =  truncation_interval(Qbeta_bar, Q, QiE[j,j], idx, beta_barE[j], lagrange)
            if not (beta_barE[j] < lower or beta_barE[j] > upper):
                print("Liu constraint not satisfied")
            tg = TG([(-np.inf, lower), (upper, np.inf)], scale=sigma*np.sqrt(QiE[j,j]))
            pvalue = tg.cdf(beta_barE[j])
            pvalue = float(2 * min(pvalue, 1 - pvalue))
            pvalues.append(pvalue)

        if len(pvalues) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
        else:
            selected = []

        return selected, active_set

class liu_CV(liu_theory):
            
    method_name = "Liu + CV (full)" 

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_min * np.std(Y)

class liu_1se(liu_theory):
            
    method_name = "Liu + 1SE (full)" 

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_1se * np.std(Y)

# Unrandomized selected

class lee_theory(generic_method):
    
    method_name = "Lee et al. + theory (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_theory * np.std(Y)

    def select(self):
        X, Y, lagrange = self.X, self.Y, self.lagrange
        n, p = X.shape
        X = X / np.sqrt(n)
        L = lasso.gaussian(X, Y, l_theory)
        L.fit()
        try:
            S = L.summary(compute_intervals=False, alternative='onesided')
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])

            if len(pvalues) > 0:
                selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
            else:
                selected = []
        except:
            selected = None
        return selected, active_set

class lee_CV(lee_theory):
    
    method_name = "Lee et al. + CV (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_min * np.std(Y)

class lee_1se(lee_theory):
    
    method_name = "Lee et al. + 1SE (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_1se * np.std(Y)

class lee_agressive(lee_theory):
    
    method_name = "Lee et al. + theory, aggressive (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = 0.8 * l_theory

# Randomized selected

class randomized_lasso(generic_method):

    method_name = "Randomized LASSO + theory (selected)"

    randomizer_scale = 1
    ndraw = 5000
    burnin = 1000

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_theory * np.std(Y)

    def select(self):
        X, Y, lagrange = self.X, self.Y, self.lagrange

        n, p = X.shape
        X = X / np.sqrt(n)

        rand_lasso = highdim.gaussian(X,
                                      Y,
                                      lagrange,
                                      randomizer_scale=self.randomizer_scale * np.std(Y))

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]
        _, pvalues, _ = rand_lasso.summary(target="selected",
                                           ndraw=self.ndraw,
                                           burnin=self.burnin,
                                           compute_intervals=False)
        if len(pvalues) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
        else:
            selected = []

        return selected, active_set

class randomized_lasso_CV(randomized_lasso):

    method_name = "Randomized LASSO + CV (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_min * np.std(Y)

class randomized_lasso_1se(randomized_lasso):

    method_name = "Randomized LASSO + 1SE (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_1se * np.std(Y)

# Randomized selected smaller randomization

class randomized_lasso_half(randomized_lasso):

    method_name = "Randomized LASSO + theory, smaller (selected)"
    randomizer_scale = 0.5
    pass

class randomized_lasso_half_CV(randomized_lasso_CV):

    method_name = "Randomized LASSO + CV, smaller (selected)"
    randomizer_scale = 0.5
    pass

class randomized_lasso_half_1se(randomized_lasso_1se):

    method_name = "Randomized LASSO + 1SE, smaller (selected)"
    randomizer_scale = 0.5
    pass

# Randomized sqrt selected

class randomized_sqrtlasso(generic_method):

    method_name = "Randomized SqrtLASSO + theory (selected)"
    randomizer_scale = 1
    kappa = 0.7

    def select(self):
        self.X, self.Y = self.X, self.Y
        n, p = X.shape
        X = X / np.sqrt(n)

        lagrange = np.ones(X.shape[1]) * choose_lambda(X) * kappa

        rand_lasso = highdim.sqrt_lasso(X,
                                        Y,
                                        lagrange,
                                        randomizer_scale=self.randomizer_scale / np.sqrt(n))

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]
        _, pvalues, _ = rand_lasso.summary(target="selected",
                                           ndraw=self.ndraw,
                                           burnin=self.burnin,
                                           compute_intervals=False)
        if len(pvalues) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
        else:
            selected = []

        return selected, active_set

class randomized_sqrtlasso_half(randomized_sqrtlasso):

    method_name = "Randomized SqrtLASSO + theory, smaller (selected)"
    randomizer_scale = 0.5
    kappa = 0.7

    pass

class randomized_sqrtlasso_bigger(randomized_sqrtlasso):

    method_name = "Randomized SqrtLASSO + theory, kappa=0.8 (selected)"
    kappa = 0.8

    pass

class randomized_sqrtlasso_bigger_half(randomized_sqrtlasso):

    method_name = "Randomized SqrtLASSO + theory, smaller, kappa=0.8 (selected)"
    kappa = 0.8
    randomizer_scale = 0.5
    pass


# Randomized full

class randomized_lasso_full(generic_method):

    method_name = 'Randomized LASSO + theory (full)'

    def select(self):
        X, Y, lagrange = self.X, self.Y, self.lagrange
        n, p = X.shape
        X = X / np.sqrt(n)

        rand_lasso = highdim.gaussian(X,
                                      Y,
                                      lagrange,
                                      randomizer_scale=self.randomizer_scale)

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]
        _, pvalues, _ = rand_lasso.summary(target="full",
                                           ndraw=self.ndraw,
                                           burnin=self.burnin,
                                           compute_intervals=False)
        if len(pvalues) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
        else:
            selected = []

        return selected, active_set

class randomized_lasso_full_CV(randomized_lasso_full):

    method_name = "Randomized LASSO + CV (full)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_min * np.std(Y)

class randomized_lasso_full_1se(randomized_lasso_full):

    method_name = "Randomized LASSO + 1SE (full)"

    def __init__(self, X, Y, l_theory, l_min, l_1se):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se)
        self.lagrange = l_1se * np.std(Y)
