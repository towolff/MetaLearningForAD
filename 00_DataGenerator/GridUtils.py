import numpy as np
import pandas as pd


def compute_q(p, cos_phi=0.9, mode='inductive'):
    abs_q = p * np.tan(np.arccos(cos_phi))
    # inductive load 'consumes' reactive power
    if mode == 'inductive':
        return abs_q
    # capacitve load 'provides' reactive power
    elif mode == 'capacitve':
        return -1 * abs_q
    else:
        print('ERROR: Illegal mode: %s. Falling back to default (inductive).' % str(mode))
        return abs_q
