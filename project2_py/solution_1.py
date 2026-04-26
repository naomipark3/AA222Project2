import numpy as np

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


def _polish_inward(x_in, c, count, n, m, fd_h, polish_band, polish_target, max_iter):
    """Walk x along -grad of the active constraints until viol <= -polish_target.
    Uses forward-difference Jacobian (m c-evals per iter)."""
    x = x_in.copy()
    for _ in range(max_iter):
        if count() + 1 > n:
            return x
        c_x = np.asarray(c(x)).ravel()
        viol = float(np.max(c_x))
        if viol <= -polish_target:
            return x  #comfortably inside
        if viol > polish_band:
            return x  #too far out for this small-step polish to help
 
        active_mask = c_x > -polish_band
        if not np.any(active_mask):
            return x
 
        if count() + m > n:
            return x
        #Sum of grad c_i for i in active (the direction of steepest constraint increase)
        J_active_sum = np.zeros(m)
        for j in range(m):
            if count() + 1 > n:
                return x
            e = np.zeros(m)
            e[j] = fd_h
            c_plus = np.asarray(c(x + e)).ravel()
            J_active_sum[j] = float(np.sum(c_plus[active_mask] - c_x[active_mask])) / fd_h
 
        d_norm = float(np.linalg.norm(J_active_sum))
        if d_norm < 1e-12:
            return x
 
        #Linearization step: c_new ~= c + (grad c).step. Step in direction
        #-J_active_sum to reduce active constraints. Length sized to drive
        #current viol from +viol to -polish_target, with 2x safety factor.
        step_len = 2.0 * (viol + polish_target) / d_norm
        x = x - step_len * J_active_sum / d_norm
    return x

def bfgs_helper(f_local, g_local, x0, n, count, max_evals, eval_cost_per_grad):
    """
    BFGS (Algorithm 6.6 from textbook) with Armijo backtracking. 
    We used this algorithm in Project 1 but it has some modifications:
      - respects local sub-budget `max_evals` AND the global `n`
      - returns the FINAL iterate, not the lowest-f_local iterate (outer AL
        tracks best feasible point itself; f_local is the AL whose minimum
        is generally NOT the best feasible point of the original problem)
      - computes real f(x0) instead of f_val=inf, since the latter auto-accepts
        any first step including overshoots that BFGS can't recover from
        (matters on Rosenbrock-like surfaces).
    """
    m = len(x0)
    x = x0.copy()
    count_start = count()
 
    if count() + eval_cost_per_grad > n:
        return x
    grad = g_local(x)
 
    #Real f(x0): without it, Armijo with f_val=inf auto-accepts any first
    #step including ones that worsen f. On Rosenbrock from x0=(0.1,0.43) this
    #locks BFGS into a bad iterate it can't escape from.
    if count() + 2 > n:
        return x
    f_val = f_local(x)
 
    Q = np.eye(m)
 
    c1 = 1e-4
    rho_ls = 0.5
    max_backtracks = 10
 
    while True:
        evals_used = count() - count_start
        if evals_used + eval_cost_per_grad + 2 > max_evals:
            break
        if count() + eval_cost_per_grad + 2 > n:
            break
 
        d = -Q @ grad
        if grad @ d >= 0:
            Q = np.eye(m)
            d = -grad
 
        #Cap initial step so ||alpha*d|| <= 1; prevents huge first steps when
        #the gradient is large (matters for Rosenbrock and for AL once rho is big).
        d_norm = np.linalg.norm(d)
        alpha = min(1.0, 1.0 / d_norm) if d_norm > 1e-12 else 1.0
        directional_deriv = grad @ d  # negative since d is descent
 
        x_new = x
        f_new = f_val
        accepted = False
        for _ in range(max_backtracks):
            if count() + 2 > n:
                break
            x_new = x + alpha * d
            f_new = f_local(x_new)
            if f_new <= f_val + c1 * alpha * directional_deriv:
                accepted = True
                break
            alpha *= rho_ls
 
        if not accepted:
            #If we've already retried with Q=I and still can't make progress, stop.
            if np.allclose(Q, np.eye(m)):
                break
            Q = np.eye(m)
            continue
 
        if count() + eval_cost_per_grad > n:
            break
 
        grad_new = g_local(x_new)
 
        #BFGS inverse-Hessian update (textbook eq. 6.26)
        delta = x_new - x
        gamma = grad_new - grad
        dg = float(delta @ gamma)
 
        if dg > 1e-10:
            Qg = Q @ gamma
            term1 = np.outer(delta, Qg) + np.outer(Qg, delta)
            term2 = (1.0 + (gamma @ Qg) / dg) * np.outer(delta, delta)
            Q = Q - term1 / dg + term2 / dg
 
        x = x_new
        f_val = f_new
        grad = grad_new
 
    return x