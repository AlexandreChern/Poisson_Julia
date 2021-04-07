using SparseArrays
using LinearAlgebra
using Plots
N = 63
h = 1/N
x = range(0,stop=1,length=N+1)
x = x[2:end-1]
u_exact = zeros(N+1)


function Jacobi(N,h,ω,k,nsteps)
    D = spzeros(N-1,N-1)
    L = spzeros(N-1,N-1)
    U = spzeros(N-1,N-1)
    f = spzeros(N-1)

    σ = 0
    for i in 1:N-1
        D[i,i] = 2 + σ*h^2
    end

    for i in 1:N-2
        L[i+1,i] = 1
        U[i,i+1] = 1
    end

    D_inv = sparse(inv(Matrix(D)))

    A = D - L - U

    RJ = D_inv*(L+U)

    # ω = 0.5

    Rω = (1-ω) * sparse(I,N-1,N-1) + ω*RJ

    # v = randn(N-1)
    J = 0:N
    v = sin.(k*J/(N)*π)

    # nsteps = 1000


    for i in 1:nsteps
        # v .= Rω * v + ω.*D_inv*f*h^2
        v_new = copy(v)
        for j = 2:N
            v_new[j] = (1-ω) * v[j] + ω * 1/2 * (v[j-1] + v[j+1])
        end
        v = copy(v_new)
    end

    # @show v
    # @show norm(A*v)
    # plot(x,v)
    return norm(v-u_exact,Inf)
end

iters = 1:2:101

ω = 2/3
err_1 = Jacobi.(N,h,ω,1,iters)
err_3 = Jacobi.(N,h,ω,3,iters)
err_6 = Jacobi.(N,h,ω,6,iters)

fig1 = plot(iters,[err_1,err_3,err_6],ylims=[0,1])
savefig(fig1,"jacobi")


function Gauss_Seidel(N,h,k,nsteps)
    D = spzeros(N-1,N-1)
    L = spzeros(N-1,N-1)
    U = spzeros(N-1,N-1)
    f = spzeros(N-1)

    σ = 0
    for i in 1:N-1
        D[i,i] = 2 + σ*h^2
    end

    for i in 1:N-2
        L[i+1,i] = 1
        U[i,i+1] = 1
    end

    D_inv = sparse(inv(Matrix(D)))

    A = D - L - U

    RG = (D-L)\U

    J = 0:N
    v = sin.(k*J/(N)*π)

    # nsteps = 1000

    for i in 1:nsteps
        # v .= RG * v + h^2 * σ * inv(Matrix(D-L)) * f
        for j = 2:N
            v[j] = 1/2*(v[j-1] + v[j+1])
        end
    end

    return norm(v-u_exact,Inf)
end

err_1 = Gauss_Seidel.(N,h,1,iters)
err_3 = Gauss_Seidel.(N,h,3,iters)
err_6 = Gauss_Seidel.(N,h,6,iters)

fig1 = plot(iters,[err_1,err_3,err_6],ylims=[0,1])
savefig(fig1,"gauss_seidel")


function RB_Gauss_Seidel(N,h,k,nsteps)
    D = spzeros(N-1,N-1)
    L = spzeros(N-1,N-1)
    U = spzeros(N-1,N-1)
    f = spzeros(N-1)

    σ = 0
    for i in 1:N-1
        D[i,i] = 2 + σ*h^2
    end

    for i in 1:N-2
        L[i+1,i] = 1
        U[i,i+1] = 1
    end

    D_inv = sparse(inv(Matrix(D)))

    A = D - L - U

    RG = (D-L)\U

    # ω = 0.5

    # Rω = (1-ω) * sparse(I,N-1,N-1) + ω*RG

    # v = randn(N-1)
    J = 0:N
    v = sin.(k*J/(N)*π)

    # nsteps = 1000

    for i in 1:nsteps
        # v .= RG * v + h^2 * σ * inv(Matrix(D-L)) * f
        for j = 2:N
            if mod(j,2) == 0
                v[j] = 1/2*(v[j-1] + v[j+1])
            end
        end
        for j = 2:N
            if mod(j,2) == 1
                v[j] = 1/2*(v[j-1] + v[j+1])
            end
        end
    end

    # @show v
    # @show norm(A*v)
    # plot(x,v)
    return norm(v-u_exact,Inf)
end

err_1 = RB_Gauss_Seidel.(N,h,1,iters)
err_3 = RB_Gauss_Seidel.(N,h,3,iters)
err_6 = RB_Gauss_Seidel.(N,h,6,iters)

fig1 = plot(iters,[err_1,err_3,err_6])
savefig(fig1,"rg_gauss_seidel")