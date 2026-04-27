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
from project2_py.solution_algos import algo_1, algo_2

def optimize(f, g, c, x0, n, count, prob):
    x_best = x0

    x_best = algo_1(f, g, c, x0, n, count, prob) #call solution algorithm (I wrote this as a helper method for cleanliness)
    return x_best