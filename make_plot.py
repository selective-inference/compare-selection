import os, glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gaussian_methods import methods


def extract_results(df):

    results = [v for v in df.groupby([df.index, 'class_name'])]
    rho_results = [v for v in df.groupby([df.index, 'class_name', 'rho'])]
    signal_results = [v for v in df.groupby([df.index, 'class_name', 'signal'])]

    return results, rho_results, signal_results
    
def plot(df,
         fixed,
         param,
         feature,
         outbase,
         methods=None):

    # results, rho_results, signal_results = extract_results(df)

    methods = methods or np.unique(df['class_name'])

    # plot with rho on x axis
    g_plot = sns.FacetGrid(df, col=fixed, hue='method_name', sharex=True, sharey=True, col_wrap=2, size=5)
    
    def power_plot(param, power, color='r', label='foo'):
        ax = plt.gca()
        ax.plot(param, power, 'o--', color=color, label=label)
        ax.set_xticks(sorted(np.unique(param)))
        ax.set_ylim([0,1])

    rendered_plot = g_plot.map(power_plot, 'rho', 'Full model power')
    rendered_plot.add_legend()
    rendered_plot.savefig(outbase + '.pdf')

    return df

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Make plots for

Try:
    python make_plot.py --methods lee_theory liu_theory --csvbase indep.csv
''')
    parser.add_argument('--methods', nargs='+',
                        dest='methods', 
                        help='Names of methods in plot (i.e. class name). Defaults to all methods.')
    parser.add_argument('--param', 
                        dest='param',
                        default='rho',
                        help='Make a plot with param on x-axis for varying fixed')
    parser.add_argument('--fixed', 
                        dest='fixed',
                        default='signal',
                        help='Which value if fixed for each facet')
    parser.add_argument('--feature', 
                        dest='feature',
                        default='power',
                        help='Variable for y-axis')
    parser.add_argument('--csvbase', help='Basename of csvfile.', dest='csvbase')
    parser.add_argument('--outbase', help='Begginning of name of pdf / png files where results are plotted. Defaults to the base of csvfile.')

    opts = parser.parse_args()

    csvfiles = glob.glob(opts.csvbase + '*signal*rho*csv')
    df = pd.concat([pd.read_csv(f, comment='#') for f in csvfiles])

    df = plot(df,
              opts.fixed,
              opts.param,
              {'power':'Full model power', 'fdr': 'Full model FDR'}[opts.feature],
              opts.outbase,
              methods=opts.methods)



