#Plotting-only code
#Mirrors `optimize` in project2.py exactly but records the full trajectory
#at every accepted BFGS step plus per-outer transitions, for the writeup
#Use with `problem.nolimit()` to avoid budget interference with the trace.

import numpy as np
from project2_py.bfgs import bfgs_helper, _polish_inward


def optimize_with_history(f, g, c, x0, n, count, prob):
    """
    Same Augmented Lagrangian + BFGS as project2_py.optimize, but returns
    (x_best, history) where history is a dict:
        'path'        : list of np.ndarray, x at each accepted iterate (incl. x0)
        'f'           : list of float, f(x) at each path point (true f, not L_A)
        'viol'        : list of float, max(c(x)) at each path point
        'outer_starts': list of int, indices into path marking start of each outer iter
        'rho'         : list of float, value of rho at each outer iter
        'lam'         : list of np.ndarray, multipliers at each outer iter

    Note: f and c are evaluated at every recorded step for plotting purposes,
    which uses extra budget. Caller should use `problem.nolimit()` to disable
    the eval cap before invoking this function.
    """
    x = np.array(x0, dtype=float).copy()
    m = len(x)

    if count() + 1 > n:
        return x, _empty_history(x, f, c)
    c_x = np.atleast_1d(c(x))
    p = len(c_x)

    if count() + 1 > n:
        return x, _empty_history(x, f, c)
    f_x = f(x)

    #Initialize history with x0
    path = [x.copy()]
    f_hist = [f_x]
    viol_hist = [float(np.max(c_x))]
    outer_starts = []
    rho_hist = []
    lam_hist = []

    margin = 1e-5

    x_best_feas = None
    f_best_feas = np.inf
    x_least_viol = x.copy()
    viol_least = float(np.max(c_x))
    if viol_least <= 0.0:
        x_best_feas = x.copy()
        f_best_feas = f_x

    lam = np.zeros(p)
    rho = 1.0
    rho_growth = 5.0
    rho_max = 1e8
    max_outer = 12
    fd_h = 1e-6
    eval_cost_per_grad = 3 + m

    x_prev = x.copy()

    #Recording callback fires once per accepted BFGS step
    def step_callback(x_new):
        path.append(x_new.copy())
        f_hist.append(float(f(x_new)))
        viol_hist.append(float(np.max(np.atleast_1d(c(x_new)))))

    for outer in range(max_outer):
        remaining = n - count()
        outers_left = max_outer - outer
        if remaining < eval_cost_per_grad + 4:
            break
        if np.isinf(remaining):
            #nolimit() in plotting mode to give each outer plenty of room
            budget = 10000
        else:
            budget = max(remaining // outers_left, int(remaining * 0.40))
            budget = min(budget, remaining - 4)

        outer_starts.append(len(path) - 1)  #index of x at start of this outer
        rho_hist.append(rho)
        lam_hist.append(lam.copy())

        lam_local = lam.copy()
        rho_local = rho

        def f_AL(x_in, _lam=lam_local, _rho=rho_local):
            f_val = f(x_in)
            c_val = np.atleast_1d(c(x_in)) + margin
            psi = np.maximum(0.0, c_val + _lam / _rho)
            return f_val + 0.5 * _rho * float(np.sum(psi * psi))

        def g_AL(x_in, _lam=lam_local, _rho=rho_local):
            grad_f = g(x_in)
            c_val = np.atleast_1d(c(x_in)) + margin
            mults = np.maximum(0.0, _lam + _rho * c_val)
            if not np.any(mults > 0.0):
                return grad_f
            J_T_mults = np.zeros(m)
            for j in range(m):
                if count() + 1 > n:
                    return grad_f + J_T_mults
                e = np.zeros(m)
                e[j] = fd_h
                c_plus = np.atleast_1d(c(x_in + e)) + margin
                J_T_mults[j] = float(np.dot(mults, c_plus - c_val)) / fd_h
            return grad_f + J_T_mults

        x = bfgs_helper(f_AL, g_AL, x, n, count, budget, eval_cost_per_grad,
                       step_callback=step_callback)

        if count() + 1 > n:
            break
        c_x = np.atleast_1d(c(x))
        viol_actual = float(np.max(c_x))
        c_shifted = c_x + margin

        if count() + 1 > n:
            lam = np.maximum(0.0, lam + rho * c_shifted)
            break
        f_x = f(x)

        if viol_actual <= 0.0 and f_x < f_best_feas:
            x_best_feas = x.copy()
            f_best_feas = f_x
        if viol_actual < viol_least:
            x_least_viol = x.copy()
            viol_least = viol_actual

        lam = np.maximum(0.0, lam + rho * c_shifted)

        viol_shifted = float(np.max(c_shifted))
        if outer == 0 or viol_shifted > 1e-6:
            rho = min(rho * rho_growth, rho_max)

        progress = float(np.linalg.norm(x - x_prev))
        if progress < 1e-7 and viol_actual <= 0.0 and outer >= 2:
            break
        x_prev = x.copy()

    #Polish step (also recorded)
    polish_band = 10 * margin
    polish_target = 5 * margin
    polish_max_iter = 8
    candidate = x_best_feas if x_best_feas is not None else x_least_viol
    if count() + (m + 2) * polish_max_iter < n:
        candidate = _polish_inward(candidate, c, count, n, m, fd_h,
                                   polish_band, polish_target, polish_max_iter)
        #Record polished candidate as an extra step (helps visualize the polish jump)
        if not np.allclose(candidate, path[-1]):
            path.append(candidate.copy())
            f_hist.append(float(f(candidate)))
            viol_hist.append(float(np.max(np.atleast_1d(c(candidate)))))

    if count() + 1 <= n:
        c_final = np.atleast_1d(c(candidate))
        viol_final = float(np.max(c_final))
        if viol_final <= 0.0 and count() + 1 <= n:
            f_final = f(candidate)
            if f_final < f_best_feas:
                x_best = candidate
            else:
                x_best = x_best_feas if x_best_feas is not None else x_least_viol
        else:
            x_best = x_best_feas if x_best_feas is not None else x_least_viol
    else:
        x_best = x_best_feas if x_best_feas is not None else x_least_viol

    history = {
        'path': path,
        'f': f_hist,
        'viol': viol_hist,
        'outer_starts': outer_starts,
        'rho': rho_hist,
        'lam': lam_hist,
    }
    return x_best, history


def _empty_history(x, f, c):
    """Fallback for the rare case where we exit before any iteration."""
    return {
        'path': [x.copy()],
        'f': [float(f(x))],
        'viol': [float(np.max(np.atleast_1d(c(x))))],
        'outer_starts': [],
        'rho': [],
        'lam': [],
    }