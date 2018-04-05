import tempfile, os, glob

import numpy as np
import regreg.api as rr

from selection.algorithms.lasso import lasso, lasso_full
from selection.algorithms.sqrt_lasso import choose_lambda
from selection.truncated.gaussian import truncated_gaussian_old as TG
from selection.randomized.lasso import highdim

from utils import BHfilter

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

methods = {}

class generic_method(object):

    q = 0.2
    method_name = 'Generic method'

    @classmethod
    def setup(cls, sigma):
        pass

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):
        (self.X,
         self.Y,
         self.l_theory,
         self.l_min,
         self.l_1se,
         self.sigma_reid) = (X,
                             Y,
                             l_theory,
                             l_min,
                             l_1se,
                             sigma_reid)

    def select(self):
        raise NotImplementedError('abstract method')

    @classmethod
    def register(cls):
        methods[cls.__name__] = cls

# Knockoff selection

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
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_mf.register()

class knockoffs_sigma(generic_method):

    method_name = 'ModelX Knockoffs with Sigma, asdp, (full)'
    factor_method = 'asdp'

    @classmethod
    def setup(cls, sigma):

        numpy2ri.activate()

        # see if we've factored this before

        have_factorization = False
        if not os.path.exists('.knockoff_factorizations'):
            os.mkdir('.knockoff_factorizations')
        factors = glob.glob('.knockoff_factorizations/*npz')
        for factor_file in factors:
            factor = np.load(factor_file)
            sigma_f = factor['sigma']
            if ((sigma_f.shape == sigma.shape) and
                (factor['method'] == cls.factor_method) and
                np.allclose(sigma_f, sigma)):
                have_factorization = True
                print('found factorization: %s' % factor_file)
                cls.knockoff_chol = factor['knockoff_chol']

        if not have_factorization:
            print('doing factorization')
            cls.knockoff_chol = factor_knockoffs(sigma, cls.factor_method)

        numpy2ri.deactivate()

    def select(self):

        numpy2ri.activate()
        rpy.r.assign('chol_k', self.knockoff_chol)
        rpy.r('''
        knockoffs = function(X) {
           mu = rep(0, ncol(X))
           mu_k = X # sweep(X, 2, mu, "-") %*% SigmaInv_s
           X_k = mu_k + matrix(rnorm(ncol(X) * nrow(X)), nrow(X)) %*% 
            chol_k
           return(X_k)
        }
            ''')
        numpy2ri.deactivate()

        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=knockoffs)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_sigma.register()

def factor_knockoffs(sigma, method='asdp'):

    numpy2ri.activate()
    rpy.r.assign('Sigma', sigma)
    rpy.r.assign('method', method)
    rpy.r('''

    # Compute the Cholesky -- from create.gaussian

    diag_s = diag(switch(method, equi = create.solve_equi(Sigma), 
                  sdp = create.solve_sdp(Sigma), asdp = create.solve_asdp(Sigma)))
    if (is.null(dim(diag_s))) {
        diag_s = diag(diag_s, length(diag_s))
    }
    SigmaInv_s = solve(Sigma, diag_s)
    Sigma_k = 2 * diag_s - diag_s %*% SigmaInv_s
    chol_k = chol(Sigma_k)
    ''')
    knockoff_chol = np.asarray(rpy.r('chol_k'))
    SigmaInv_s = np.asarray(rpy.r('SigmaInv_s'))
    diag_s = np.asarray(rpy.r('diag_s'))
    np.savez('.knockoff_factorizations/%s.npz' % (os.path.split(tempfile.mkstemp()[1])[1],),
             method=method,
             sigma=sigma,
             knockoff_chol=knockoff_chol)

    return knockoff_chol

class knockoffs_sigma_equi(knockoffs_sigma):

    method_name = 'ModelX Knockoffs with Sigma, equi, (full)'
    factor_method = 'equi'

knockoffs_sigma_equi.register()

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
            V = np.asarray(V, np.int)
            return V, V
        except:
            return [], []

knockoffs_orig.register()

class knockoffs_fixed(generic_method):

    method_name = 'Knockoffs fixed (full)'

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=create.fixed)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_fixed.register()

# Liu, Markovic, Tibs selection

class liu_theory(generic_method):

    method_name = "Liu + theory (full)"            
    sigma_estimator = 'reid'

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

    def select(self):

        X, Y, lagrange = self.X, self.Y, self.lagrange
        n, p = X.shape

        L = lasso_full.gaussian(X, Y, lagrange * np.sqrt(n))
        L.fit()

        if len(L.active) > 0:
            if self.sigma_estimator == 'reid':
                dispersion = self.sigma_reid**2
            else:
                dispersion = None
            S = L.summary(compute_intervals=False, dispersion=dispersion)
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])

            if len(pvalues) > 0:
                selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
            else:
                selected = []
        else:
            selected, active_set = [], []
        return selected, active_set
liu_theory.register()

class liu_aggressive(liu_theory):

    method_name = "Liu + theory, aggressive (full)"            

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)

        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

liu_aggressive.register()

class liu_CV(liu_theory):
            
    method_name = "Liu + CV (full)" 

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])
liu_CV.register()

class liu_1se(liu_theory):
            
    method_name = "Liu + 1SE (full)" 

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

liu_1se.register()

class liu_R_theory(liu_theory):

    method_name = "Liu + theory (R code)"

    def select(self):
        if True: #try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('y', self.Y)
            rpy.r.assign('sigma_reid', self.sigma_reid)
            rpy.r('y = as.numeric(y)')

            rpy.r.assign('lam', self.lagrange[0])
            rpy.r('''
        p = ncol(X);
        n = nrow(X);

        sigma_est = 1.
        if (p >= n) { 
            sigma_est = sigma_reid
        } else {
            sigma_est = sigma(lm(y ~ X - 1))
        }

        penalty_factor = rep(1, p);
        lam = lam / sqrt(n);  # lambdas are passed a sqrt(n) free from python code
        soln = selectiveInference:::solve_problem_glmnet(X, y, lam, penalty_factor=penalty_factor, loss="ls")
        PVS = selectiveInference:::inference_group_lasso(X, y, 
                                                         soln, groups=1:ncol(X), 
                                                         lambda=lam, penalty_factor=penalty_factor, 
                                                         sigma_est, loss="ls", algo="Q", 
                                                         construct_ci=FALSE)
        active_vars=PVS$active_vars - 1 # for 0-based
        pvalues = PVS$pvalues
        ''')

            pvalues = np.asarray(rpy.r('pvalues'))
            active_set = np.asarray(rpy.r('active_vars'))
            numpy2ri.deactivate()
            if len(active_set) > 0:
                selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
            else:
                selected = []
            return selected, active_set
        else: # except:
            return [np.nan], [np.nan] # some R failure occurred 
liu_R_theory.register()

class liu_R_aggressive(liu_R_theory):

    method_name = "Liu + aggressive (R code)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8
liu_R_aggressive.register()

class lee_full_R_theory(liu_theory):

    method_name = "Lee + theory (R code)"

    def select(self):
        numpy2ri.activate()
        rpy.r.assign('x', self.X)
        rpy.r.assign('y', self.Y)
        rpy.r('y = as.numeric(y)')
        rpy.r.assign('sigma_reid', self.sigma_reid)
        rpy.r.assign('lam', self.lagrange[0])
        rpy.r('''
    sigma_est=sigma_reid
    n = nrow(x);
    gfit = glmnet(x, y, standardize=FALSE, intercept=FALSE)
    lam = lam / sqrt(n);  # lambdas are passed a sqrt(n) free from python code
    if (lam < max(abs(t(x) %*% y) / n)) {
        beta = coef(gfit, x=x, y=y, s=lam, exact=TRUE)[-1]
        out = fixedLassoInf(x, y, beta, lam*n, sigma=sigma_est, type='full', intercept=FALSE)
        active_vars=out$vars - 1 # for 0-based
        pvalues = out$pv
    } else {
        pvalues = NULL
        active_vars = numeric(0)
    }
    ''')

        pvalues = np.asarray(rpy.r('pvalues'))
        active_set = np.asarray(rpy.r('active_vars'))
        numpy2ri.deactivate()
        if len(active_set) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
        else:
            selected = []
        return selected, active_set
lee_full_R_theory.register()

class lee_full_R_aggressive(lee_full_R_theory):

    method_name = "Lee + theory (R code)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8
lee_full_R_aggressive.register()

# Unrandomized selected

class lee_theory(generic_method):
    
    method_name = "Lee et al. + theory (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

    def select(self):

        X, Y, lagrange = self.X, self.Y, self.lagrange
        n, p = X.shape
        X = X / np.sqrt(n)
        L = lasso.gaussian(X, Y, lagrange)
        L.fit()
        if len(L.active) > 0:
            S = L.summary(compute_intervals=False, alternative='onesided')
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])

            if len(pvalues) > 0:
                selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
            else:
                selected = []
        else:
            selected, active_set = [], []
        return selected, active_set

lee_theory.register()

class lee_CV(lee_theory):
    
    method_name = "Lee et al. + CV (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

lee_CV.register()

class lee_1se(lee_theory):
    
    method_name = "Lee et al. + 1SE (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

lee_1se.register()

class lee_aggressive(lee_theory):
    
    method_name = "Lee et al. + theory, aggressive (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = 0.8 * l_theory * np.ones(X.shape[1])

lee_aggressive.register()

class sqrt_lasso(generic_method):

    method_name = 'SqrtLASSO, kappa=0.7 (selected)'
    kappa = 0.7

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = self.kappa * choose_lambda(X)

    def select(self):

        X, Y, lagrange = self.X, self.Y, self.lagrange
        n, p = X.shape
        X = X / np.sqrt(n)
        L = lasso.sqrt_lasso(X, Y, lagrange)
        L.fit()
        if len(L.active) > 0:
            S = L.summary(compute_intervals=False, alternative='onesided')
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])

            if len(pvalues) > 0:
                selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
            else:
                selected = []
        else:
            selected, active_set = [], []
        return selected, active_set

# Randomized selected

class randomized_lasso(generic_method):

    method_name = "Randomized LASSO + theory (selected)"
    target = 'selected'
    randomizer_scale = 1
    ndraw = 5000
    burnin = 1000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

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
        _, pvalues, _ = rand_lasso.summary(target=self.target,
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

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

class randomized_lasso_1se(randomized_lasso):

    method_name = "Randomized LASSO + 1SE (selected)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

randomized_lasso.register(), randomized_lasso_CV.register(), randomized_lasso_1se.register()

# More aggressive lambda choice

class randomized_lasso_aggressive(randomized_lasso):

    method_name = "Randomized LASSO + theory, aggressive (selected)"

    randomizer_scale = 1
    ndraw = 5000
    burnin = 1000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

class randomized_lasso_aggressive_half(randomized_lasso):

    method_name = "Randomized LASSO + theory, aggressive, smaller noise (selected)"

    randomizer_scale = 0.5
    ndraw = 5000
    burnin = 1000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

class randomized_lasso_aggressive_quarter(randomized_lasso):

    method_name = "Randomized LASSO + theory, aggressive, smaller noise (selected)"

    randomizer_scale = 0.25
    ndraw = 5000
    burnin = 1000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

randomized_lasso_aggressive.register(), randomized_lasso_aggressive_half.register(), randomized_lasso_aggressive_quarter.register()

# Randomized selected smaller randomization

class randomized_lasso_half(randomized_lasso):

    method_name = "Randomized LASSO + theory, smaller noise (selected)"
    randomizer_scale = 0.5
    pass

class randomized_lasso_half_CV(randomized_lasso_CV):

    method_name = "Randomized LASSO + CV, smaller noise (selected)"
    randomizer_scale = 0.5
    pass

class randomized_lasso_half_1se(randomized_lasso_1se):

    method_name = "Randomized LASSO + 1SE, smaller noise (selected)"
    randomizer_scale = 0.5
    pass

randomized_lasso_half.register(), randomized_lasso_half_CV.register(), randomized_lasso_half_1se.register()
# Randomized sqrt selected

class randomized_sqrtlasso(generic_method):

    method_name = "Randomized SqrtLASSO, kappa=0.7, (selected)"
    randomizer_scale = 1
    kappa = 0.7
    ndraw = 5000
    burnin = 1000

    def select(self):
        X, Y = self.X, self.Y
        n, p = X.shape
        X = X / np.sqrt(n)

        lagrange = np.ones(X.shape[1]) * choose_lambda(X) * self.kappa

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

    method_name = "Randomized SqrtLASSO, kappa=0.7, smaller noise (selected)"
    randomizer_scale = 0.5
    kappa = 0.7

    pass

randomized_sqrtlasso.register(), randomized_sqrtlasso_half.register()

class randomized_sqrtlasso_bigger(randomized_sqrtlasso):

    method_name = "Randomized SqrtLASSO, kappa=0.8 (selected)"
    kappa = 0.8

    pass

class randomized_sqrtlasso_bigger_half(randomized_sqrtlasso):

    method_name = "Randomized SqrtLASSO, kappa=0.8, smaller noise (selected)"
    kappa = 0.8
    randomizer_scale = 0.5
    pass

randomized_sqrtlasso_bigger.register(), randomized_sqrtlasso_bigger_half.register()

# Randomized full

class randomized_lasso_full(randomized_lasso):

    method_name = 'Randomized LASSO + theory (full)'
    randomizer_scale = 1
    ndraw = 5000
    burnin = 1000
    target = 'full'

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

class randomized_lasso_full_CV(randomized_lasso_full):

    method_name = "Randomized LASSO + CV (full)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

class randomized_lasso_full_1se(randomized_lasso_full):

    method_name = "Randomized LASSO + 1SE (full)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

randomized_lasso_full.register(), randomized_lasso_full_CV.register(), randomized_lasso_full_1se.register()

# Randomized full smaller randomization

class randomized_lasso_full_half(randomized_lasso_full):

    method_name = "Randomized LASSO + theory, smaller noise (full)"
    randomizer_scale = 0.5

class randomized_lasso_full_half_CV(randomized_lasso_full_CV):

    method_name = "Randomized LASSO + CV, smaller noise (full)"
    randomizer_scale = 0.5
    pass

class randomized_lasso_full_half_1se(randomized_lasso_full_1se):

    method_name = "Randomized LASSO + 1SE, smaller noise (full)"
    randomizer_scale = 0.5
    pass

randomized_lasso_full_half.register(), randomized_lasso_full_half_CV.register(), randomized_lasso_full_half_1se.register()

# Aggressive choice of lambda

class randomized_lasso_full_aggressive(randomized_lasso_full):

    method_name = "Randomized LASSO + theory, aggressive, smaller noise (full)"

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

class randomized_lasso_full_aggressive_half(randomized_lasso_full_aggressive):

    method_name = "Randomized LASSO + theory, aggressive, smaller noise (full)"
    randomizer_scale = 0.5

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

randomized_lasso_full_aggressive.register(), randomized_lasso_full_aggressive_half.register()

class randomized_lasso_R_theory(randomized_lasso):

    method_name = "Randomized LASSO + theory (R code, selected)"

    def select(self):
        numpy2ri.activate()
        rpy.r.assign('X', self.X)
        rpy.r.assign('y', self.Y)
        rpy.r('y = as.numeric(y)')
        rpy.r.assign('q', self.q)
        rpy.r.assign('lam', self.lagrange[0])
        rpy.r('''
        n = nrow(X)
        p = ncol(X)
        lam = lam * sqrt(n)
        result = randomizedLasso(X, y, lam, ridge_term=sd(y) * sqrt(n), 
                                 noise_scale = sd(y) * 0.5 * sqrt(n), family='gaussian')
        active_set = result$active_set
        sigma_est = sigma(lm(y ~ X[,active_set] - 1))
        targets = selectiveInference:::compute_target(result, 'partial', sigma_est = sigma_est, 
                                 construct_pvalues=rep(TRUE, length(active_set)), 
                                 construct_ci=rep(FALSE, length(active_set)))
        out = randomizedLassoInf(result,
                                 targets=targets)
        pvalues = out$pvalues
        active_set = active_set - 1
        ''')

        pvalues = np.asarray(rpy.r('pvalues'))
        active_set = np.asarray(rpy.r('active_set'))
        numpy2ri.deactivate()
        if len(active_set) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
        else:
            selected = []
        return selected, active_set
randomized_lasso_R_theory.register()
