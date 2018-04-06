import numpy as np

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('suppressMessages(library(selectiveInference)); suppressMessages(library(knockoff))') # R libraries we will use

def gaussian_setup(X, Y, run_CV=True):
    """

    Some calculations that can be reused by methods:
    
    lambda.min, lambda.1se, lambda.theory and Reid et al. estimate of noise

    """
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
        rpy.r('sigma_reid = selectiveInference:::estimate_sigma(X, Y, coef(G, s="lambda.min")[-1]) # sigma via Reid et al.')
        rpy.r("L = G[['lambda.min']]")
        rpy.r("L1 = G[['lambda.1se']]")
        L = rpy.r('L')
        L1 = rpy.r('L1')
        sigma_reid = rpy.r('sigma_reid')[0]
        numpy2ri.deactivate()
        return L * np.sqrt(X.shape[0]), L1 * np.sqrt(X.shape[0]), l_theory, sigma_reid
    else:
        return None, None, l_theory, None

def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)





