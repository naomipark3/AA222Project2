import numpy as np
from project2_py.bfgs import bfgs_helper, _polish_inward

def algo_1(f, g, c, x0, n, count, prob):
    """
    Augmented Lagrangian (Kochenderfer & Wheeler, Algorithms for Optimization Section 10.10, pg. 197)
    with BFGS (Section 6.2, Alg. 6.6) as the inner unconstrained solver.
    For inequality constraints c(x) <= 0:
    L_A(x; lam, rho) = f(x) + sum_i [(rho/2) max(0, c_i(x) + lam_i/rho)^2 - lam_i^2/(2 rho)]
    grad L_A(x) = grad f(x) + sum_i max(0, lam_i + rho c_i(x)) * grad c_i(x)
    update: lam_i <- max(0, lam_i + rho c_i(x))
    Three modifications to the algorithm were implemented to pass the tests/handle the autograder:
    (1) Inward feasibility margin (c(x) + margin <= 0): drives the iterate
    strictly inside, so floating-point at the boundary doesn't fail the
    strict `c <= 0` test (e.g. simple2's optimum at (1,1)).
    (2) Two-track best-point tracking: x_best_feas (strictly feasible best)
    and x_least_viol (fallback). Return the strict one if found.
    (3) Polish step at the end: walks x along -grad of the most-active
    constraint until comfortably inside, in case the AL converged near
    the boundary with active mults and BFGS stalled.
    """
    x = np.array(x0, dtype=float).copy()
    m = len(x)
 
    #Probe c-dimension and a baseline f as follows:
    if count() + 1 > n:
        return x
    c_x = np.asarray(c(x)).ravel()
    p = len(c_x)
 
    if count() + 1 > n:
        return x
    f_x = f(x)
 
    #Inward feasibility margin: AL is solved on c(x) + margin <= 0. As a result,
    #the iterate is driven strictly inside the original feasible set. Costs 
    #about a margin worth of f-suboptimality but avoids the boundary FP failure mode.
    #(in reality, cost in objective function scales with margin, but how it scales varies
    #problem to problem)
    margin = 1e-5
 
    #Two-track best-point tracking
    x_best_feas = None
    f_best_feas = np.inf
    x_least_viol = x.copy()
    viol_least = float(np.max(c_x))
 
    if viol_least <= 0.0:
        x_best_feas = x.copy()
        f_best_feas = f_x
 
    #augmented Lagrangian hyperparameters
    lam = np.zeros(p)
    rho = 1.0
    rho_growth = 5.0
    rho_max = 1e8
    max_outer = 12
    fd_h = 1e-6
 
    #Cost model: g_AL = g (2) + c (1) + m forward-diff c-evals = 3 + m
    eval_cost_per_grad = 3 + m
 
    x_prev = x.copy()
 
    for outer in range(max_outer):
        remaining = n - count()
        outers_left = max_outer - outer
        if remaining < eval_cost_per_grad + 4:
            break
 
        #Slightly front-load: max(even split, 40% of remaining), capped to leave probes.
        budget = max(remaining // outers_left, int(remaining * 0.40))
        budget = min(budget, remaining - 4)
 
        #Capture current (lam, rho) by default args
        lam_local = lam.copy()
        rho_local = rho
 
        def f_AL(x_in, _lam=lam_local, _rho=rho_local):
            f_val = f(x_in)
            c_val = np.asarray(c(x_in)).ravel() + margin
            psi = np.maximum(0.0, c_val + _lam / _rho)
            return f_val + 0.5 * _rho * float(np.sum(psi * psi))
 
        def g_AL(x_in, _lam=lam_local, _rho=rho_local):
            grad_f = np.asarray(g(x_in)).ravel()
            c_val = np.asarray(c(x_in)).ravel() + margin
            mults = np.maximum(0.0, _lam + _rho * c_val)
            if not np.any(mults > 0.0):
                return grad_f
            J_T_mults = np.zeros(m)
            for j in range(m):
                if count() + 1 > n:
                    return grad_f + J_T_mults
                e = np.zeros(m)
                e[j] = fd_h
                c_plus = np.asarray(c(x_in + e)).ravel() + margin
                J_T_mults[j] = float(np.dot(mults, c_plus - c_val)) / fd_h
            return grad_f + J_T_mults
 
        x = bfgs_helper(f_AL, g_AL, x, n, count, budget, eval_cost_per_grad)
 
        #Post-inner probes
        if count() + 1 > n:
            break
        c_x = np.asarray(c(x)).ravel()
        viol_actual = float(np.max(c_x))      #we need a strict c <=0 to pass the tests
        c_shifted = c_x + margin              #multiplier update uses this
 
        if count() + 1 > n:
            lam = np.maximum(0.0, lam + rho * c_shifted)
            break
        f_x = f(x)
 
        #Strict-feasibility best update
        if viol_actual <= 0.0 and f_x < f_best_feas:
            x_best_feas = x.copy()
            f_best_feas = f_x
 
        #Least-violation fallback update
        if viol_actual < viol_least:
            x_least_viol = x.copy()
            viol_least = viol_actual
 
        #Multiplier update (uses shifted c, matching the AL formulation)
        lam = np.maximum(0.0, lam + rho * c_shifted)
 
        #Grow rho only if there's still positive shifted violation; otherwise
        #multipliers are doing the work and growing rho just adds ill-conditioning.
        viol_shifted = float(np.max(c_shifted))
        if outer == 0 or viol_shifted > 1e-6:
            rho = min(rho * rho_growth, rho_max)
 
        #Progress-based termination: x stopped moving and we're feasible.
        progress = float(np.linalg.norm(x - x_prev))
        if progress < 1e-7 and viol_actual <= 0.0 and outer >= 2:
            break
        x_prev = x.copy()
 
    #Final polish step: Handles the case where AL converged near c=0 boundary on problems where
    #the unconstrained min lies on the constraint (e.g. simple2 at (1,1) with
    #grad f = 0 AND c = 0). Walks inward along -grad c_active until c <= -5*margin.
    polish_band = 10 * margin
    polish_target = 5 * margin
    polish_max_iter = 8
 
    candidate = x_best_feas if x_best_feas is not None else x_least_viol
    if count() + (m + 2) * polish_max_iter < n:
        candidate = _polish_inward(candidate, c, count, n, m, fd_h, polish_band, polish_target, polish_max_iter)
 
    #Accept polished candidate if strictly feasible AND lower-f than any best so far
    if count() + 1 <= n:
        c_final = np.asarray(c(candidate)).ravel()
        viol_final = float(np.max(c_final))
        if viol_final <= 0.0 and count() + 1 <= n:
            f_final = f(candidate)
            if f_final < f_best_feas:
                return candidate
 
    if x_best_feas is not None:
        return x_best_feas
    return x_least_viol
 
 
def algo_2(f, g, c, x0, n, count, prob, return_history=False):
    """
    BFGS with Quadratic Penalty (considered the dual of augmented lagrangian)
    (Section 10.9, pg. 196, Eq. 10.35 in the textbook):
    P(x; rho) = f(x) + (rho/2) sum_i max(0, c_i(x) + margin)^2
    grad P = grad f + rho sum_i max(0, c_i + margin) grad c_i
 
    Requires rho -> inf to drive the iterate strictly feasible. This makes
    QP slower and worse-conditioned than AL on tight-boundary problems
    (e.g. simple2's degenerate (1,1) corner) but it's a textbook baseline
    and is structurally distinct from AL (no multiplier update).
 
    `return_history=True` returns (x_best, hist) where hist is a list of
    (f_x, viol_actual) tuples per outer, for the f-vs-iter and
    viol-vs-iter plots required by §2.4 of the writeup.
    """
    x = np.array(x0, dtype=float).copy()
    m = len(x)
 
    if count() + 1 > n:
        return (x, []) if return_history else x
    c_x = np.atleast_1d(c(x))
    p = len(c_x)
 
    if count() + 1 > n:
        return (x, []) if return_history else x
    f_x = f(x)
 
    margin = 1e-5  #same inward shift as AL
 
    x_best_feas = None
    f_best_feas = np.inf
    x_least_viol = x.copy()
    viol_least = float(np.max(c_x))
    if viol_least <= 0.0:
        x_best_feas = x.copy()
        f_best_feas = f_x
 
    rho = 1.0
    rho_growth = 10.0  #grow faster than AL's 5x because QP needs larger rho to enforce constraints
    rho_max = 1e10
    max_outer = 12
    fd_h = 1e-6
    eval_cost_per_grad = 3 + m
 
    history = [(f_x, viol_least)]
    x_prev = x.copy()
 
    for outer in range(max_outer):
        remaining = n - count()
        outers_left = max_outer - outer
        if remaining < eval_cost_per_grad + 4:
            break
        budget = max(remaining // outers_left, int(remaining * 0.40))
        budget = min(budget, remaining - 4)
 
        rho_local = rho
 
        def f_QP(x_in, _rho=rho_local):
            f_val = f(x_in)
            c_val = np.atleast_1d(c(x_in)) + margin
            psi = np.maximum(0.0, c_val)
            return f_val + 0.5 * _rho * float(np.sum(psi * psi))
 
        def g_QP(x_in, _rho=rho_local):
            grad_f = g(x_in)
            c_val = np.atleast_1d(c(x_in)) + margin
            mults = _rho * np.maximum(0.0, c_val)  #no lambda term -- pure penalty
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
 
        x = bfgs_helper(f_QP, g_QP, x, n, count, budget, eval_cost_per_grad)
 
        if count() + 1 > n:
            break
        c_x = np.atleast_1d(c(x))
        viol_actual = float(np.max(c_x))
 
        if count() + 1 > n:
            break
        f_x = f(x)
 
        history.append((f_x, viol_actual))
 
        if viol_actual <= 0.0 and f_x < f_best_feas:
            x_best_feas = x.copy()
            f_best_feas = f_x
        if viol_actual < viol_least:
            x_least_viol = x.copy()
            viol_least = viol_actual
 
        #Always grow rho. This is because QP has no multiplier mechanism, only path to convergence is rho -> inf
        rho = min(rho * rho_growth, rho_max)
 
        progress = float(np.linalg.norm(x - x_prev))
        if progress < 1e-7 and viol_actual <= 0.0 and outer >= 2:
            break
        x_prev = x.copy()
 
    #Final polish step (same as AL)
    polish_band = 10 * margin
    polish_target = 5 * margin
    polish_max_iter = 8
    candidate = x_best_feas if x_best_feas is not None else x_least_viol
    if count() + (m + 2) * polish_max_iter < n:
        candidate = _polish_inward(candidate, c, count, n, m, fd_h,
                                   polish_band, polish_target, polish_max_iter)
 
    if count() + 1 <= n:
        c_final = np.atleast_1d(c(candidate))
        viol_final = float(np.max(c_final))
        if viol_final <= 0.0 and count() + 1 <= n:
            f_final = f(candidate)
            if f_final < f_best_feas:
                return (candidate, history) if return_history else candidate
 
    result = x_best_feas if x_best_feas is not None else x_least_viol
    return (result, history) if return_history else result