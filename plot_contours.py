#Generates contour-plus-path figures for simple1 and simple2 using AL+BFGS.
#Output: simple1_al.png, simple2_al.png

import numpy as np
import matplotlib.pyplot as plt
 
from project2_py.helpers import Simple1, Simple2
from project2_py.plotting_utils import (
    optimize_with_history,
    optimize_qp_with_history,
)
 
 
def evaluate_grid(p, xs, ys):
    #Vectorized over the grid by calling f and c pointwise (problems are 2D
    #and grids are small enough that the loop cost doesn't matter).
    X, Y = np.meshgrid(xs, ys)
    F = np.zeros_like(X)
    C = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = np.array([X[i, j], Y[i, j]])
            F[i, j] = p._wrapped_f(pt)
            C[i, j] = float(np.max(np.atleast_1d(p._wrapped_c(pt))))
    return X, Y, F, C
 
 
def plot_problem(test_class, name, seeds, levels, optimizer, algo_label, filename):
    #Build evaluation grid over the [-3, 3]^2 domain required by the spec.
    xs = np.linspace(-3, 3, 200)
    ys = np.linspace(-3, 3, 200)
    p_grid = test_class()
    X, Y, F, C = evaluate_grid(p_grid, xs, ys)
 
    fig, ax = plt.subplots(figsize=(6, 6))
 
    #Shade the infeasible region (where max c(x) > 0) in light gray.
    ax.contourf(X, Y, C, levels=[0.0, C.max() + 1.0], colors=['lightgray'], alpha=0.5)
 
    #Constraint boundary as a thin line.
    ax.contour(X, Y, C, levels=[0.0], colors='gray', linewidths=1.0)
 
    #f contours in black.
    ax.contour(X, Y, F, levels=levels, colors='black', linewidths=0.5, alpha=0.7)
 
    #Run the chosen optimizer from each IC and plot the path. Lock to the
    #first three default colors so each IC's line and markers stay matched.
    colors = ['C0', 'C1', 'C2']
    for seed, col in zip(seeds, colors):
        p = test_class()
        p.nolimit()
        np.random.seed(seed)
        x0 = p.x0()
        _, hist = optimizer(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
        path = np.array(hist['path'])
        ax.plot(path[:, 0], path[:, 1], '-', color=col, linewidth=1.2)
        ax.plot(path[0, 0], path[0, 1], 'o', color=col, markersize=6,
                markerfacecolor='white', markeredgewidth=1.2)
        ax.plot(path[-1, 0], path[-1, 1], 'x', color=col, markersize=8,
                markeredgewidth=1.5)
 
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(f'{name}: {algo_label}')
    ax.set_aspect('equal')
    #Legend showing start (circle), finish (x), infeasible region, and IC paths
    ax.plot([], [], 'o', color='black', markerfacecolor='white',
            markeredgewidth=1.2, label='Start')
    ax.plot([], [], 'x', color='black',
            markeredgewidth=1.5, label='Finish')
    ax.plot([], [], 's', color='lightgray', alpha=0.5,
            label=r'$c(x) > 0$')
    #Initial condition paths (C0, C1, C2)
    ax.plot([], [], '-', color='C0', linewidth=1.2, label='initial condition 1')
    ax.plot([], [], '-', color='C1', linewidth=1.2, label='initial condition 2')
    ax.plot([], [], '-', color='C2', linewidth=1.2, label='initial condition 3')
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
 
 
#simple1's f is roughly linear and spans (-8.6, 9.4): use linear levels.
simple1_levels = np.linspace(-8, 9, 18)
#simple2's f is Rosenbrock-like and spans (0.08, 14400): use log-spaced levels.
simple2_levels = np.logspace(-1, 4, 16)
 
#Algorithm 1: AL + BFGS
plot_problem(Simple1, 'simple1', [0, 5, 10], simple1_levels,
             optimize_with_history, 'BFGS with Augmented Lagrangian', 'simple1_al.png')
plot_problem(Simple2, 'simple2', [0, 5, 10], simple2_levels,
             optimize_with_history, 'BFGS with Augmented Lagrangian', 'simple2_al.png')
 
#Algorithm 2: QP + BFGS
plot_problem(Simple1, 'simple1', [0, 5, 10], simple1_levels,
             optimize_qp_with_history, 'BFGS with Quadratic Penalty', 'simple1_qp.png')
plot_problem(Simple2, 'simple2', [0, 5, 10], simple2_levels,
             optimize_qp_with_history, 'BFGS with Quadratic Penalty', 'simple2_qp.png')
 
print('Saved simple1_al.png, simple2_al.png, simple1_qp.png, simple2_qp.png')