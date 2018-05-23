import numpy as np, pandas as pd, time

def pvalue_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):

    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)

    active, pvalues = M.generate_pvalues()
    tic = time.time()

    if len(active) > 0:
        value = pd.DataFrame(pvalues, columns=['P-value'])
        value['Time'] = tic-toc
        return value

def pvalue_summary(result):

    value = pd.DataFrame([[np.mean(result['P-value']), 
                           np.std(result['P-value'])]],
                         columns=['P-value',
                                  'SD(P-value)'])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

def FDR_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):
    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    selected, active = M.select()
    tic = time.time()
    true_active = np.nonzero(beta)[0]

    if active is not None:
        TD = instance.discoveries(selected, true_active)
        FD = len(selected) - TD
        FDP = FD / max(TD + 1. * FD, 1.)
        return pd.DataFrame([[TD / (len(true_active)*1.), FD, FDP, tic-toc, len(active)]],
                            columns=['Full model power',
                                     'False discoveries',
                                     'Full model FDP',
                                     'Time',
                                     'Active'])
    else:
        return pd.DataFrame([[0, 0, 0, tic-toc, 0]],
                            columns=['Full model power',
                                     'False discoveries',
                                     'Full model FDP',
                                     'Time',
                                     'Active'])

def FDR_summary(result):

    nresult = result['Full model power'].shape[0]
    value = pd.DataFrame([[nresult,
                           np.mean(result['Full model power']), 
                           np.std(result['Full model power']) / np.sqrt(nresult),
                           np.mean(result['False discoveries']), 
                           np.mean(result['Full model FDP']), 
                           np.std(result['Full model FDP']) / np.sqrt(nresult),
                           np.mean(result['Time']),
                           np.mean(result['Active'])]],
                         columns=['Replicates', 
                                  'Full model power', 
                                  'SD(Full model power)', 
                                  'False discoveries', 
                                  'Full model FDR', 
                                  'SD(Full model FDR)', 
                                  'Time', 
                                  'Active',
                                  ])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value
