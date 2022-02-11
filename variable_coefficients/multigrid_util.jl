function prolongation_matrix(N)
    # SBP preserving
    # N = 2^level + 1
    odata = spzeros(2*N-1,N)
    for i in 1:2*N-1
        if i % 2 == 1
            odata[i,div(i+1,2)] = 1
        else
            odata[i,div(i,2)] = 1/2
            odata[i,div(i,2)+1] = 1/2
        end
    end
    return odata
end

function restriction_matrix(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/2
    odata[end,end-1] = 1/2
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function restriction_matrix_normal(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/4
    odata[end,end-1] = 1/4
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function prolongation_2d(N)
    prolongation_1d = prolongation_matrix(N)
    prolongation_2d = kron(prolongation_1d,prolongation_1d)
    return prolongation_2d
end

function restriction_2d(N)
    # restriction_1d = restriction_matrix_normal(N)
    restriction_1d = restriction_matrix(N)
    restriction_2d = kron(restriction_1d,restriction_1d)
    return restriction_2d
end


function jacobi_brittany!(x,A,b;maxiter=3, ω = 2/3)

    Pinv = Diagonal(1 ./ diag(A))
    P = Diagonal(diag(A))
    Q = A-P

    for j in 1:maxiter
        x[:] = ω * Pinv*(b .- Q*x[:]) + (1 - ω)*x[:]
    end
end

function Two_level_multigrid(A,b;nu=3,NUM_V_CYCLES=1,SBPp=2)
    Nx = Ny = Int(sqrt(length(b)))
    level = Int(log(2,Nx-1))
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = create_A_b(level-1);

    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)
    N_values = Dict(1 => Nx)
    N_values[2] = div(Nx+1,2)

    x = zeros(length(b));
    v_values[1] = x
    
    for cycle_number in 1:NUM_V_CYCLES
        jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        r = b - A*v_values[1];
        f = restriction_2d(Nx) * r;
        v_values[2] = A_2h \ f

        # println("Pass first part")
        e_1 = prolongation_2d(N_values[2]) * v_values[2];
        v_values[1] = v_values[1] + e_1;
        # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
        jacobi_brittany!(v_values[1],A,b;maxiter=nu);
    end
    return (v_values[1],norm(A * v_values[1] - b))
end


function precond_matrix(A, b; m=3, solver="jacobi",SBPp=SBPp)
    #pre and post smoothing 
    N = length(b)
    Nx = Ny = Integer((sqrt(N)))
    level = Integer(log(2,Nx-1))
    IN = sparse(Matrix(I, N, N))
    P = Diagonal(diag(A))
    Pinv = Diagonal(1 ./ diag(A))
    Q = P-A
    L = A - triu(A)
    U = A - tril(A)

    if solver == "jacobi"
        ω = 2/3
        H = ω*Pinv*Q + (1-ω)*IN 
        R = ω*Pinv 
        R0 = ω*Pinv 
    elseif solver == "ssor"
        ω = 1.4  #this is just a guess. Need to compute ω_optimal (from jacobi method)
        B1 = (P + ω*U)\Matrix(-ω*L + (1-ω)*P)
        B2 = (P + ω*L)\Matrix(-ω*U + (1-ω)*P) 
        H = B1*B2
        X = (P+ω*L)\Matrix(IN)
   
        R = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
        R0 = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
    else   
        # wait to be implemented
    end

    for i = 1:m-1
        R += H^i * R0
    end

    # (A_2h, b_2h, x_2h, H1_2h) = get_operators(p, 2*h);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = create_A_b(level-1,SBPp=SBPp)
    I_r = restriction_2d(Nx)
    
    I_p = prolongation_2d(Nx_2h)
    # M = H^m * (R + I_p * (A_2h\Matrix(I_r*(IN - A * R)))) + R
    M = H^m * (R - I_p * (A_2h\Matrix(I_r*(A * R - IN)))) + R
   
    return (M, R, H, I_p, A_2h, I_r, IN)
end


function mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0)
    r = b - A * x;
    # (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b;m=nu,solver="jacobi",p=p)
    # rnew = zeros(length(r))
    z = Two_level_multigrid(A,r;nu=nu,NUM_V_CYCLES=1)[1]
    # z = M*r
    p = z;
    # Ap = A*p;
    num_iter_steps = 0
    norms = [norm(r)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end

    rzold = r'*z

    for step = 1:maxiter
    # for step = 1:100
    # for step = 1:5
        num_iter_steps += 1
        # @show norm(A*p)

        # alpha = r'*z/(p'*A*p)
        alpha = rzold / (p'*A*p)
        # @show alpha

        x .= x .+ alpha * p;

        r .= r .- alpha * A*p
        rs = r' * r
        append!(norms,sqrt(rs))
        if direct_sol != 0 && H_tilde != 0
            sol_error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            # @show error
            append!(errors,sol_error)
        end
        if sqrt(rs) < abstol
            break
        end
        z = Two_level_multigrid(A,r;nu=nu,NUM_V_CYCLES=1)[1]
        # z = M*r
        rznew = r'*z
        beta = rznew/(rzold);
        p = z + beta * p;
        rzold = rznew
    end
    return num_iter_steps, norms, errors
end

A_matrices = Dict()
