import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

methods = ['ModelX Knockoffs with Sigma (full)',
           'Randomized LASSO + 1SE, smaller noise (selected)',
           'Randomized LASSO + 1SE, smaller noise (full)',
           'Lee et al. + theory, aggressive (selected)']

def make_plot_rho_fixed(csvfile, 
                        rho,
                        metric='Full model power',
                        methods=methods,
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
    ax.set_xlim([min(signal), max(signal)])
    ax.set_xlabel('Signal size')
    ax.set_ylabel(metric)
    ax.set_xticks(signal)
    return ax, fig

def make_plot_signal_fixed(csvfile, 
                           signal,
                           metric='Full model power',
                           methods=methods,
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
        if len(df_) > 0:
            ax.plot(df_['rho'], df_[metric], 'o--', label=method)
            print(method)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim([min(rho), max(rho)])
    ax.set_xlabel(r'AR parameter $\rho$')
    ax.set_xticks(rho)
    ax.set_ylabel(metric)

    return ax, fig

def signal_plot(csvfile, rho=0.25):

    fig = plt.figure(num=1, figsize=(10, 5))
    fig.clf()
    ax = plt.subplot(121)
    ax, fig = make_plot_rho_fixed(csvfile, rho=rho, ylim=(0,1), ax=ax)

    ax = plt.subplot(122)
    ax, fig = make_plot_rho_fixed(csvfile, rho=rho, ylim=(0,1), metric='Full model FDR', ax=ax)
    ax.plot([3.5,5], [0.2, 0.2], 'k--', linewidth=2)
    ax.legend(loc='upper right', fontsize='xx-small')
    fig.savefig(csvfile[:-4] + '_rho%0.2f.pdf' % rho)

def rho_plot(csvfile, signal=4):

    fig = plt.figure(num=1, figsize=(10, 5))
    fig.clf()
    ax = plt.subplot(121)
    ax, fig = make_plot_signal_fixed(csvfile, signal=signal, ylim=(0,1), ax=ax)

    ax = plt.subplot(122)
    ax, fig = make_plot_signal_fixed(csvfile, signal=signal, ylim=(0,1), metric='Full model FDR', ax=ax)
    ax.plot([0, 0.75], [0.2, 0.2], 'k--', linewidth=2)
    ax.legend(loc='upper right', fontsize='xx-small')
    fig.savefig(csvfile[:-4] + '_signal%0.1f.pdf' % signal)

if __name__ == "__main__":

    for csvfile in ['simresults/AR_n1000_p2000.csv', 'simresults/mixed_n1000_p2000.csv']:
        for rho in [0, 0.25, 0.5]:
            signal_plot(csvfile, rho=rho)
        for signal in [3.5,4,4.5,5]:
            rho_plot(csvfile, signal=signal)


