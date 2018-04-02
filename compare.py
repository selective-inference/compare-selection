import os
from copy import copy
from itertools import product
import time

import numpy as np
import pandas as pd

from utils import data_instances, lagrange_vals
from methods import methods

import knockoff_phenom # more instances

def compare(instance, 
            nsim=50, 
            q=0.2,
            methods=[], 
            verbose=False,
            htmlfile=None,
            method_setup=True):
    
    results = [[] for m in methods]
    
    run_CV = np.any(['CV' in str(m) for m in methods] + ['1se' in str(m) for m in methods])

    if method_setup:
        for method in methods:
            method.setup(instance.sigma)

    for i in range(nsim):

        X, Y, beta = instance.generate()
        l_min, l_1se, l_theory = lagrange_vals(X.copy(), Y.copy(), run_CV=run_CV)
        true_active = np.nonzero(beta)[0]

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
            M = method(X.copy(), Y.copy(), l_theory, l_min, l_1se)
            M.q = q
            selected, active = M.select()
            tic = time.time()
            if active is not None:
                TD = instance.discoveries(selected, true_active)
                FD = len(selected) - TD
                FDP = FD / max(TD + 1. * FD, 1.)
                result.append((TD / (len(true_active)*1.), FD, FDP, tic-toc, len(active)))

            if i > 0:
                df = pd.DataFrame([[m.method_name] + summary(r) for m, r in zip(methods, results)], 
                                  columns=['Method', 'Replicates', 'Full model power', 'SD(Full model power)', 'False discoveries', 'Full model FDR', 'SD(Full model FDR)', 'Time', 'Active'])

                if verbose:
                    print(df[['Replicates', 'Full model power', 'Time']])

                if htmlfile is not None:
                    f = open(htmlfile, 'w')
                    f.write(df.to_html(index=False) + '\n')
                    f.write(instance.params.to_html())
                    f.close()

    big_df = copy(df)
    param = instance.params
    for col in param.columns:
        big_df[col] = param[col][0] 
    big_df['distance_tol'] = instance.distance_tol
    return big_df


def main(opts, clean=False):

    if opts.list_instances:
        print('Instances:\n')
        print(sorted(data_instances.keys()))
    if opts.list_methods:
        print('Methods:\n')
        print(sorted(methods.keys()))
    if opts.list_instances or opts.list_methods:
        return

    if opts.signal_strength is not None:  # looping over signal strengths
        signal_vals = np.atleast_1d(opts.signal_strength)
    else:
        signal_vals = [None]

    new_opts = copy(opts)
    prev_rho = np.nan

    csvfiles = []
    for rho, signal in product(np.atleast_1d(opts.rho),
                               signal_vals):

        # try to save some time on setup of knockoffs

        method_setup = rho != prev_rho 
        prev_rho = rho

        new_opts.signal_strength = signal
        new_opts.rho = rho

        _methods = [methods[n] for n in new_opts.methods]
        _instance = data_instances[new_opts.instance]

        if _instance.signature is None:
            instance = _instance()
        else:
            instance = _instance(**dict([(n, getattr(new_opts, n)) for n in _instance.signature if hasattr(new_opts, n)]))

        if signal is not None: # here is where signal_fac can be ignored
            instance.signal = new_opts.signal_strength

        if opts.csvfile is not None:
            new_opts.csvfile = (os.path.splitext(opts.csvfile)[0] + 
                       "_signal%0.1f_rho%0.2f.csv" % (new_opts.signal_strength,
                                                      new_opts.rho))
        csvfiles.append(new_opts.csvfile)

        results = compare(instance,
                          nsim=new_opts.nsim,
                          methods=_methods,
                          verbose=new_opts.verbose,
                          htmlfile=new_opts.htmlfile,
                          method_setup=method_setup)

        if opts.csvfile is not None:

            f = open(new_opts.csvfile, 'w')
            f.write('# parsed arguments: ' + str(new_opts) + '\n') # comment line indicating arguments used
            f.write(results.to_csv(index=False) + '\n')
            f.close()

            dfs = [pd.read_csv(f, comment='#') for f in csvfiles]
            df = pd.concat(dfs)
            df.to_csv(opts.csvfile, index=False)

    if opts.clean:
        [os.remove(f) for f in csvfiles]

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Compare different LASSO methods in terms of full model FDR and Power.

Try:
    python compare.py --instance indep_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile indep.html --csvfile indep.csv
''')
    parser.add_argument('--instance',
                        dest='instance', help='Which instance to generate data from -- only one choice. To see choices run --list_instances.')
    parser.add_argument('--list_instances',
                        dest='list_instances', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Which methods to use -- choose many. To see choices run --list_methods.', dest='methods')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true')
    parser.add_argument('--nsample', default=800, type=int,
                        dest='n',
                        help='number of data points, n (default 800)')
    parser.add_argument('--nfeature', default=300, type=int,
                        dest='p',
                        help='the number of features, p (default 300)')
    parser.add_argument('--nsignal', default=20, type=int,
                        dest='s',
                        help='the number of nonzero coefs, s (default 20)')
    parser.add_argument('--signal', type=float, nargs='+',
                        dest='signal_strength',
                        help='signal strength to override instance default (default value: None) -- signals are all of this magnitude, randomly placed with random signs')
    parser.add_argument('--signal_fac', default=1.2, type=float,
                        help='Scale applied to theoretical lambda to get signal size. Ignored if --signal is used.')
    parser.add_argument('--rho', nargs='+', type=float,
                        default=0.5,
                        dest='rho',
                        help='Value of AR(1), equicor or mixed param.')
    parser.add_argument('--q', default=0.2, type=float,
                        help='target for FDR (default 0.2)')
    parser.add_argument('--nsim', default=100, type=int,
                        help='How many repetitions?')
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--htmlfile', help='HTML file to store results for one (signal, rho). When looping over (signal, rho) this HTML file tracks the current progress.',
                        dest='htmlfile')
    parser.add_argument('--csvfile', help='CSV file to store results looped over (signal, rho).',
                        dest='csvfile')
    parser.add_argument('--clean', help='Remove individual CSV files after termination?',
                        default=False)

    opts = parser.parse_args()

    main(opts)

