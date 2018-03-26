import time

import numpy as np
import pandas as pd

from utils import instances, lagrange_vals
from methods import methods

def compare(instance, nsim=50, q=0.2,
            methods=[], verbose=False,
            htmlfile=None):
    
    results = [[] for m in methods]
    
    runCV = np.any(['CV' in str(m) for m in methods] + ['1se' in str(m) for m in methods])

    for i in range(nsim):

        X, Y, beta = instance.generate()
        l_min, l_1se, l_theory = lagrange_vals(X, Y, runCV=runCV)
        true_active = set(np.nonzero(beta)[0])

        def summary(result):
            result = np.atleast_2d(result)

            return [result.shape[0],
                    np.mean(result[:,0]), 
                    np.std(result[:,0]) / np.sqrt(result.shape[0]), 
                    np.mean(result[:,1]), 
                    np.mean(result[:,2]), 
                    np.std(result[:,2]) / np.sqrt(result.shape[0]),
                    np.mean(result[:,3]),
                    np.mean(result[:,4])]

        for method, result in zip(methods, results):
            toc = time.time()
            M = method(X, Y, l_theory, l_min, l_1se)
            M.q = q
            selected, active = M.select()
            tic = time.time()
            if active is not None:
                TD = len(true_active.intersection(selected))
                FD = len(selected) - TD
                FDP = FD / max(TD + 1. * FD, 1.)
                result.append((TD / (len(true_active)*1.), FD, FDP, tic-toc, len(active)))

            if i > 1:
                df = pd.DataFrame([summary(r) for r in results], 
                                  index=[m.method_name for m in methods],
                                  columns=['Replicates', 'Full model power', 'SD(Full model power)', 'False discoveries', 'Full model FDR', 'SD(Full model FDR)', 'Time', 'Active'])

                if verbose:
                    print(df[['Replicates', 'Full model power', 'Time']])

                if htmlfile is not None:
                    f = open(htmlfile, 'w')
                    f.write(df.to_html() + '\n')
                    f.write(instance.params.to_html())
                    f.close()

    return df


def main(opts):

    if opts.list_instances:
        print('Instances:\n')
        print(sorted(instances.keys()))
    if opts.list_methods:
        print('Methods:\n')
        print(sorted(methods.keys()))
    if opts.list_instances or opts.list_methods:
        return

    _methods = [methods[n] for n in opts.methods]
    _instance = instances[opts.instance]

    if 'jelena' in opts.instance:
        instance = _instance()
    elif not opts.rho:
        instance = _instance(n=opts.nsample,
                             p=opts.nfeature,
                             s=opts.nsignal,
                             signal_fac=opts.signal_fac)
    else:
        instance = _instance(n=opts.nsample,
                             p=opts.nfeature,
                             s=opts.nsignal,
                             signal_fac=opts.signal_fac,
                             rho=rho)

    results = compare(instance,
                      nsim=opts.nsim,
                      methods=_methods,
                      verbose=opts.verbose,
                      htmlfile=opts.htmlfile)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Compare different LASSO methods in terms of full model FDR and Power.

Try:
    python compare.py --instance indep_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile indep.html
''')
    parser.add_argument('--instance',
                        dest='instance', help='Which instance to generate data from -- only one choice. To see choices run --list_instances.')
    parser.add_argument('--list_instances',
                        dest='list_instances', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Which methods to use -- choose many. To see choices run --list_methods.', dest='methods')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true')
    parser.add_argument('--nsample', default=800, type=int,
                        help='number of data points, n (default 800)')
    parser.add_argument('--nfeature', default=300, type=int,
                        help='the number of features, p (default 300)')
    parser.add_argument('--nsignal', default=20, type=int,
                        help='the number of nonzero coefs, s (default 20)')
    parser.add_argument('--signal_fac', default=1.2, type=float,
                        help='Scale applied to theoretical lambda to get signal size.')
    parser.add_argument('--rho', type=float,
                        dest='rho',
                        help='Value of AR(1), equicor or mixed param.')
    parser.add_argument('--q', default=0.2, type=float,
                        help='target for FDR (default 0.2)')
    parser.add_argument('--nsim', default=100, type=int,
                        help='How many repetitions?')
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--htmlfile', help='HTML file to store results.',
                        dest='htmlfile')

    opts = parser.parse_args()

    main(opts)

