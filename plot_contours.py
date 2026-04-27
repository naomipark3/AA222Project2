#Generates contour-plus-path figures for simple1 and simple2 using AL+BFGS.
#Output: simple1_al.png, simple2_al.png

import numpy as np
import matplotlib.pyplot as plt

from project2_py.helpers import Simple1, Simple2
from project2_py.plotting_utils import optimize_with_history


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


def plot_problem(test_class, name, seeds, levels):
    #Build evaluation grid over the [-3, 3]^2 domain specified in project handout
    xs = np.linspace(-3, 3, 200)
    ys = np.linspace(-3, 3, 200)
    p_grid = test_class()
    X, Y, F, C = evaluate_grid(p_grid, xs, ys)

    fig, ax = plt.subplots(figsize=(6, 6))

    #**Shade the infeasible region (where max c(x) > 0) in light gray. Project handout specifies
    #to plot the FEASIBLE region (c(x) \leq 0) on top of a contour plot
    ax.contourf(X, Y, C, levels=[0.0, C.max() + 1.0], colors=['lightgray'], alpha=0.5)

    #Constraint boundary as a thin line.
    ax.contour(X, Y, C, levels=[0.0], colors='gray', linewidths=1.0)

    #f contours in black.
    ax.contour(X, Y, F, levels=levels, colors='black', linewidths=0.5, alpha=0.7)

    #Run AL+BFGS from each IC and plot the path. Lock to the first three default
    #matplotlib colors so line and markers stay matched per IC.
    colors = ['C0', 'C1', 'C2']
    for seed, col in zip(seeds, colors):
        p = test_class()
        p.nolimit()
        np.random.seed(seed)
        x0 = p.x0()
        _, hist = optimize_with_history(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
        path = np.array(hist['path'])
        ax.plot(path[:, 0], path[:, 1], '-', color=col, linewidth=1.2)
        ax.plot(path[0, 0], path[0, 1], 'o', color=col, markersize=6,
                markerfacecolor='white', markeredgewidth=1.2) #start
        ax.plot(path[-1, 0], path[-1, 1], 'x', color=col, markersize=8,
                markeredgewidth=1.5) #finish

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(f'{name}: BFGS with Augmented Lagrangian')
    ax.set_aspect('equal')
    #Legend showing start (circle) and finish (x)
    ax.plot([], [], 'o', color='black', markerfacecolor='white',
            markeredgewidth=1.2, label='Start')
    ax.plot([], [], 'x', color='black',
            markeredgewidth=1.5, label='Finish')
    ax.plot([], [], 's', color='lightgray', alpha=0.5,
            label=r'$c(x) > 0$')
    #Add initial condition paths (C0, C1, C2) to the legend
    ax.plot([], [], '-', color='C0', linewidth=1.2, label='initial condition 1')
    ax.plot([], [], '-', color='C1', linewidth=1.2, label='initial condition 2')
    ax.plot([], [], '-', color='C2', linewidth=1.2, label='initial condition 3')
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig

#simple1's f is roughly linear and spans (-8.6, 9.4): use linear levels.
fig1 = plot_problem(Simple1, 'simple1', seeds=[0, 5, 10],
                    levels=np.linspace(-8, 9, 18))
fig1.savefig('simple1_al.png', dpi=150)

#simple2's f is Rosenbrock-like and spans (0.08, 14400): use log-spaced levels.
fig2 = plot_problem(Simple2, 'simple2', seeds=[0, 5, 10],
                    levels=np.logspace(-1, 4, 16))
fig2.savefig('simple2_al.png', dpi=150)