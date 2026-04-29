#Generates the AL-vs-QP comparison table in the writeup. For each of the
#three simple problems, runs both algorithms on 500 random seeds (matching
#the autograder's localtest setup) and reports feasibility, median f,
#mean f, and max f

import numpy as np

from project2_py.helpers import Simple1, Simple2, Simple3
from project2_py.project2 import optimize     #algorithm 1: AL + BFGS


def assess(opt_fn, test_class, name, n_trials=500):
    fs, viols, evals = [], [], []
    overrun = 0
    for seed in range(n_trials):
        p = test_class()
        np.random.seed(seed)
        x0 = p.x0()
        xb = opt_fn(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
        if p.count() > p.n:
            overrun += 1
        evals.append(p.count())
        p._reset()
        cb = p.c(xb)
        fb = p.f(xb)
        fs.append(fb)
        viols.append(float(np.max(cb)))

    fs = np.array(fs)
    viols = np.array(viols)
    evals = np.array(evals)
    feas = viols <= 0.0
    ff = fs[feas] if feas.any() else fs

    print(f"{name:}: feas {feas.sum():3d}/{n_trials},"
          f"f median={np.median(ff):+.3e},"
          f"f mean={np.mean(ff):+.3e},"
          f"f max={np.max(ff):+.3e}")


print("Algorithm 1: QP + BFGS (project2.optimize)")
assess(optimize, Simple1, "simple1")
assess(optimize, Simple2, "simple2")
assess(optimize, Simple3, "simple3")