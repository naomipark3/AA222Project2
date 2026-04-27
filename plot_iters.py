#Generates objective-vs-iteration and max-violation-vs-iteration plots for
#simple2 from 3 ICs, using AL+BFGS (algorithm 1). Output:
#simple2_al_f.png, simple2_al_viol.png

import numpy as np
import matplotlib.pyplot as plt

from project2_py.helpers import Simple2
from project2_py.plotting_utils import optimize_with_history

#Tiny floor for log-scale plotting so feasible iterates (where max(0, c) = 0)
#and the converged f (which can be ~1e-13) don't blow up log(0).
LOG_FLOOR = 1e-15

def make_iter_plot(seeds, y_extractor, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))

    #Run AL+BFGS from each IC and plot the per-iteration trace. Lock to the
    #first three default colors so each IC's line and markers stay matched.
    colors = ['C0', 'C1', 'C2']
    for seed, col in zip(seeds, colors):
        p = Simple2()
        p.nolimit()
        np.random.seed(seed)
        x0 = p.x0()
        _, hist = optimize_with_history(p.f, p.g, p.c, x0, p.n, p.count, p.prob)

        y = np.array([y_extractor(fi, vi)
                      for fi, vi in zip(hist['f'], hist['viol'])])
        y = np.maximum(y, LOG_FLOOR)
        iters = np.arange(len(y))

        ax.plot(iters, y, '-', color=col, linewidth=1.2)
        ax.plot(iters[0], y[0], 'o', color=col, markersize=6,
                markerfacecolor='white', markeredgewidth=1.2)
        ax.plot(iters[-1], y[-1], 'x', color=col, markersize=8,
                markeredgewidth=1.5)

    ax.set_yscale('log')
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    #Legend showing start (circle), finish (x), and the three IC colors.
    ax.plot([], [], 'o', color='black', markerfacecolor='white',
            markeredgewidth=1.2, label='Start')
    ax.plot([], [], 'x', color='black',
            markeredgewidth=1.5, label='Finish')
    ax.plot([], [], '-', color='C0', linewidth=1.2, label='initial condition 1')
    ax.plot([], [], '-', color='C1', linewidth=1.2, label='initial condition 2')
    ax.plot([], [], '-', color='C2', linewidth=1.2, label='initial condition 3')
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()
    return fig


#Objective function vs iteration: plot f(x) directly. Floor handles late
#iterations where f drops below 1e-13 (BFGS is essentially at the optimum).
make_iter_plot(
    seeds=[0, 5, 10],
    y_extractor=lambda f, v: f,
    ylabel=r'$f(x)$, i.e. Objective Function',
    title=r'simple2: BFGS with Augmented Lagrangian, $f$ vs iteration',
    filename='simple2_al_f.png',
)

#Max constraint violation vs iteration: plot max(0, max_i c_i(x)). When the
#iterate is feasible, max(0, c) = 0 -> floored to LOG_FLOOR so the curve
#drops to the bottom of the plot, which visually marks feasibility.
make_iter_plot(
    seeds=[0, 5, 10],
    y_extractor=lambda f, v: max(v, 0.0),
    ylabel=r'$\max(0,\, \max_i c_i(x))$, i.e. Maximum Constraint Violation',
    title='simple2: BFGS with Augmented Lagrangian, violation vs iteration',
    filename='simple2_al_viol.png',
)

print('Saved simple2_al_f.png, simple2_al_viol.png')