#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np
from project2_py.solution_1 import algo_1

def optimize(f, g, c, x0, n, count, prob):
    x_best = x0

    x_best = algo_1(f, g, c, x0, n, count, prob)
    return x_best