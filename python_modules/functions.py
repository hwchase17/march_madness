"""
Custom-written functions to be used in python modules.
"""

import scipy as sp

def spread_to_percent(spread):
    return 1/(1 + 10 ** (-spread / 15))

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll