using LinearAlgebra
using SparseArrays

function poisson1d(n)
    h = 1/(n+1)
    x = [i*h for i in 1:n]
    A = spdiagm(-1=>ones(n-1), 0=>-2*ones(n), 1=>ones(n-1)) / h^2
    f = -ones(n)
    return x, A, f
end

function mg_vcycle(x,A,f,levels, nu1, nu2)
    # compute residual
    r = f - A*x

    # pre-smoothing
    for i = 1:nu1
        x = x + A \ f
        r = f - A*x
    end

    # Restrict residual to coarser grid
    # r_coarse = A[2:2:end, 1:end-1] * r[1:end-1] ## rewrite this
    r_coarse = # TO DO
    A_coarse = # TO DO
    f_coarse = r_coarse

    print(length(f_coarse))
    print(length(x))

    # solve on coarser grid
    if levels == 1
        x_coarse = A \ f_coarse
    else
        x_coarse = zeros(sizeof(f_coarse))
        for i in 1:nu2
            x_coarse = mg_vcycle(x_coarse, A_coarse, f_coarse, levels-1, nu1, nu2)
        end
    end

    # Intepolate correction to finer grid and correct solution
    print(length(kron(ones(2),x_coarse)[1:end-1]))
    x = x + kron(ones(2),x_coarse)[1:end-1]

    # post-smoothing
    for i = 1:nu1
        x = x + A \ (f - A*x)
    end

    return x
end


function full_multigrid(n, nu1, nu2, levels)
    x, A, f = poisson1d(n)

    # Initial guess
    x = zeros(n)

    print(length(x))
    print(length(f))

    # V-cycle
    x = mg_vcycle(x,A,f,levels,nu1,nu2)
    return x
end