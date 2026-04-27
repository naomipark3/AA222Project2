import numpy as np

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