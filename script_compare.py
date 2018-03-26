import time

import numpy as np
import pandas as pd

from utils import (AR_instance,
                   equicor_instance,
                   mixed_instance,
                   jelena_instance,
                   jelena_instance2,
                   indep_instance,
                   lagrange_min)

from methods import (knockoffs_mf, 
                     knockoffs_orig,
                     liu_theory,
                     liu_CV,
                     liu_1se, 
                     randomized_lasso_full,
                     randomized_lasso_full_1se,
                     randomized_lasso_full_CV,
                     lee_theory, 
                     lee_CV,
                     lee_1se,
                     randomized_lasso, 
                     randomized_lasso_half, 
                     randomized_lasso_half_1se,
                     randomized_lasso_half_CV,
                     randomized_sqrtlasso, 
                     randomized_sqrtlasso_half, 
                     randomized_lasso_CV, 
                     randomized_lasso_1se)

def compare(instance, nsim=50, q=0.2,
            methods=[], verbose=False,
            htmlfile=None, instance_args={}):
    
    results = [[] for m in methods]
    
    for i in range(nsim):

        X, Y, beta, lagrange = instance(**instance_args)
        lam_min, lam_1se = lagrange_min(X, Y)
        true_active = set(np.nonzero(beta)[0])

        def summary(result):
            result = np.atleast_2d(result)
            return [np.mean(result[:,0]), 
                    np.std(result[:,0]) / np.sqrt(result.shape[0]), 
                    np.mean(result[:,1]), 
                    np.mean(result[:,2]), 
                    np.std(result[:,2]) / np.sqrt(result.shape[0])]

        for method, result in zip(methods, results):
            toc = time.time()
            M = method(X, Y, lagrange, lam_min, lam_1se)
            M.q = q
            selected, active = M.select()
            tic = time.time()
            if selected is not None:
                TD = len(true_active.intersection(selected))
                FD = len(selected) - TD
                FDP = FD / max(TD + 1. * FD, 1.)
                result.append((TD / (len(true_active)*1.), FD, FDP))

            if verbose:
                print(method, len(result), tic-toc, len(selected), len(active))

            if i > 1:
                df = pd.DataFrame([summary(r) for r in results], 
                                  index=[m.method_name for m in methods],
                                  columns=['Full model power', 'SD(Full model power)', 'False discoveries', 'Full model FDR', 'SD(Full model FDR)'])

                if verbose:
                    print(df)

                if htmlfile is not None:
                    open(htmlfile, 'w').write(df.to_html())
    return df

results = compare(indep_instance,
                  nsim=100,
                  methods=[randomized_lasso, 
                           randomized_lasso_half, 
                           ],
                  verbose=True,
                  htmlfile='lowdim.html',
                  instance_args={'n':800, 'p':300, 's':20, 'signal_fac':1.2})

