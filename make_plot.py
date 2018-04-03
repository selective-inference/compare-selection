import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from methods import methods

default_methods = [methods[n].method_name for n in ['knockoffs_sigma',
                                                    'knockoffs_sigma_equi',
                                                    'randomized_lasso_half_1se',
                                                    'lee_aggressive']]

def make_plot_rho_fixed(csvfile, 
                        rho,
                        metric='Full model power',
                        methods=default_methods,
                        ylim=None,
                        ax=None):
    df = pd.read_csv(csvfile)

    df = df[df['rho'] == rho]
    signal = sorted(np.unique(df['signal']))

    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()
    for method in methods:
        df_ = df[df['Method'] == method]
        if len(df_) > 0:
            ax.plot(df_['signal'], df_[metric], 'o--', label=method)
            print(method)
    if ylim:
        ax.set_ylim(ylim)
    range_length = max(signal) - min(signal)
    ax.set_xlim([min(signal) - 0.1 * range_length, 
                 max(signal) + 0.1 * range_length])
    ax.set_xlabel('Signal size')
    ax.set_ylabel(metric)
    ax.set_xticks(signal)
    return ax, fig

def make_plot_signal_fixed(csvfile, 
                           signal,
                           metric='Full model power',
                           methods=default_methods,
                           ylim=None,
                           ax=None):
    df = pd.read_csv(csvfile)

    df = df[df['signal'] == signal]
    rho = sorted(np.unique(df['rho']))

    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()
    for method in methods:
        df_ = df[df['Method'] == method]
        print(method)
        print(df_)
        if len(df_) > 0:
            ax.plot(df_['rho'], df_[metric], 'o--', label=method)
            print(method)
    if ylim:
        ax.set_ylim(ylim)
    range_length = max(rho) - min(rho)
    ax.set_xlim([min(rho) - 0.1 * range_length, 
                 max(rho) + 0.1 * range_length])
    ax.set_xlabel(r'AR parameter $\rho$')
    ax.set_xticks(rho)
    ax.set_ylabel(metric)

    return ax, fig

def signal_plot(csvfile, rho=0.25, methods=[]):

    fig = plt.figure(num=1, figsize=(12, 5))
    fig.clf()
    ax = plt.subplot(121)
    ax, fig = make_plot_rho_fixed(csvfile, rho=rho, ylim=(0,1), ax=ax, methods=methods)

    ax = plt.subplot(122)
    ax, fig = make_plot_rho_fixed(csvfile, rho=rho, ylim=(0,1), metric='Full model FDR', 
                                  ax=ax, methods=methods)
    ax.plot([3.5,5], [0.2, 0.2], 'k--', linewidth=2)
    ax.legend(loc='upper right', fontsize='xx-small')
    fig.savefig(csvfile[:-4] + '_rho%0.2f.pdf' % rho)
    fig.savefig(csvfile[:-4] + '_rho%0.2f.png' % rho)

def rho_plot(csvfile, signal=4, methods=[]):

    fig = plt.figure(num=1, figsize=(10, 5))
    fig.clf()
    ax = plt.subplot(121)
    ax, fig = make_plot_signal_fixed(csvfile, signal=signal, ylim=(0,1), ax=ax, methods=methods)

    ax = plt.subplot(122)
    ax, fig = make_plot_signal_fixed(csvfile, signal=signal, ylim=(0,1), metric='Full model FDR',
                                     ax=ax, methods=methods)
    ax.plot([0, 0.75], [0.2, 0.2], 'k--', linewidth=2)
    ax.legend(loc='upper right', fontsize='xx-small')
    fig.savefig(csvfile[:-4] + '_signal%0.1f.pdf' % signal)
    fig.savefig(csvfile[:-4] + '_signal%0.1f.png' % signal)

def main(opts):

    method_names = [methods[n].method_name for n in opts.methods]
    
    if opts.rho:
        for rho in np.atleast_1d(opts.rho):
            signal_plot(opts.csvfile, rho=rho, methods=method_names)
    if opts.signal:
        for signal in np.atleast_1d(opts.signal):
            rho_plot(opts.csvfile, signal=signal, methods=method_names)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Make plots for

Try:
    python make_plot.py --methods lee_theory liu_theory --csvfile indep.csv
''')
    parser.add_argument('--methods', nargs='+', help='Which methods to use -- choose many. To see choices run --list_methods.', dest='methods')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true')
    parser.add_argument('--signal', type=float, 
                        nargs='+',
                        dest='signal',
                        help='Make a plot rho on x-axis for fixed signal')
    parser.add_argument('--rho', nargs='+', type=float,
                        dest='rho',
                        help='Make a plot with signal on x-axis for fixed rho')
    parser.add_argument('--csvfile', help='CSV file to store results looped over (signal, rho).',
                        dest='csvfile')

    opts = parser.parse_args()

    main(opts)



