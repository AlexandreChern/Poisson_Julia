function phi = F_cycle(phi,f,h)
    # Recursive F-cycle multigrid for solving the Poisson equation
    # ∇^2 phi = f on a uniform grid of spacing h

    # pre-smoothing
    phi = smoothing(phi, f, h)

    # compute Residual Errors
    rhs = restriction(r);

    eps = zeros(size(rhs));

    # stop recursion at smallest grid size, otherwise continue recursion
    if smallest_grid_size_is_achieved
        eps = smoothing(eps, rhs, 2 * h)
    else
        eps = F_Cycle(eps, rhs, 2*h)
    end

    # Prolongation and correction
    phi = phi + prolongation(eps);
    # Re-smoothing
    phi = smoothing(phi, f, h)

    # Compute residual Errors
    r = residual(phi, f, h)

    # Restriction 
    rhs = restriction(r)

    # stop recursion at smallest grid size, otherwise continue recursion
    if smallest_grid_size_is_achieved
        eps = smoothing(eps, rhs, 2 * h)
    else
        eps = V_Cycle(eps, rhs, 2*h)
    end

    # Prolongation and correction
    phi = phi + prolongation(eps)

    # Post-smoothing
    phi = smoothing(phi, f, h)
end


function phi = V_cycle(phi, f, h)
    # Recursive V-Cycle Multigrid for solving the Poisson equation 
    # ∇^2 phi = f on a uniform grid of spacing h

    # Pre-smoothing
    phi = smoothing(phi, f, h)

    # Compute Residual Errors
    r = residual(phi, f, h)

    # Restriction 
    rhs = restriction(r)
    eps = zeros(size(rhs))

    # Stop recursion at smallest grid size, otherwise continue recursion
    if smallest_grid_size_is_achieved
        eps = smoothing(eps, rhs, 2*h)
    else
        eps = V_cycle(eps, rhs, 2*h)
    end

    # Prolongation and correction
    phi = phi + prolongation(eps)

    # Post-smoothing
    phi = smoothing(phi, f, h)
end