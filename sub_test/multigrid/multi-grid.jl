using SparseArrays
using LinearAlgebra
using Plots
using Interpolations


L = 2
k = 1

function exact_u(C,k,σ,x)
    return C/(π^2*k^2 + σ) * sin.(k*π*x)
end

function f(C,k,x)
    return C*sin.(k*π*x)
end

function linear_interpolation(v)
    len_v = length(v)
    v_interpolated = zeros(2*len_v+1)
    for i in 1:2*len_v+1
        # println(i)
        if i%2 == 0
            # println("case 1")
            # v_interpolated[i] = (v[div(i,2)] + v[div(i,2)+1])/2
            v_interpolated[i] = v[div(i,2)]
        elseif i == 1 
            # println("case 2")
            # v_interpolated[i] = (v[div((i+1),2)])
            v_interpolated[i] = (v[1])/2
        elseif i == 2*len_v + 1
            v_interpolated[i] = (v[end])/2
        else
            v_interpolated[i] = (v[div(i-1,2)] + v[div(i+1,2)]) / 2
        end
    end
    return v_interpolated
end

function weighting(f)
    len_f = length(f)
    f_weighted = zeros(div(len_f-1,2))
    for i in 1:length(f_weighted)
        f_weighted[i] = (f[2*i-1] + 2*f[2*i] + f[2*i+1])/4
    end
    return f_weighted
end

# for one vector, linear interpolation and weighting are not necessary reverse operation

function Jacobi_iter(ω,v,f)
    N = length(v)
    # h = v[2] - v[1]
    h = 1/(N+1)
    v_new = copy(v)
    for j = 2:N-1
        v_new[j] = (1-ω) * v[j] + ω * 1/2 * (v[j-1] + v[j+1] + h^2*f[j])
    end
    return v_new
end

function A(v)
    # h = v[2] - v[1]
    h = 1/(length(v)+1)
    v_new = similar(v)
    v_new[1] = (2*v[1] - v[2]) / h^2
    v_new[end] = (2*v[end] - v[end-1]) / h^2
    for i in 2:length(v_new)-1
        v_new[i] = (- v[i-1] + 2v[i] - v[i+1]) / h^2
    end
    return v_new
end

function A_matrix(N)
    A = spzeros(N,N)
    h = 1/(N+1)
    for i in 1:N
        A[i,i] = 2
    end
    for i in 1:N-1
        A[i,i+1] = -1
        A[i+1,i] = -1
    end
    return A ./ h^2
end

function multi_grid(L,iter_times)
    ω = 2/3
    N = 2^7
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
    C = 1
    # iter_times = 10
    v = 1/2*(sin.(16*x*π/N) + sin.(40*x*π/N))
    # v = similar(x)
    rhs = C*sin.(k*π*x)
    v_values = Dict(1 => v)
    rhs_values = Dict(1 => rhs)
    # @show rhs_values[1]
    for i in 1:L
        @show i
        if i != L
            for _ in 1:iter_times
                v = Jacobi_iter(ω,v,rhs_values[i])
            end
            v_values[i] = v
            rhs = weighting(rhs_values[i] - A(v_values[i]))
            # rhs = weighting(rhs_values[i] - A_matrix(N-1)*(v_values[i]))
            rhs_values[i+1] = rhs
            N = div(N,2)
            v = zeros(N-1)
        else
            v_values[i] = A_matrix(N-1) \ rhs_values[i]
            # v_values[i] = Jacobi_iter(ω,v,rhs_values[i])
        end
        @show v_values[i]
    end
    println("Pass first part")
    for i in 1:length(v_values)
        # @show length(v_values[i])
    end
    for i in 1:L-1
        j = L - i
        # @show j
        # @show v_values[j]
        # @show v_values[j+1]
        v_values[j] = v_values[j] + linear_interpolation(v_values[j+1])
        v = v_values[j]
        for i in 1:iter_times
            v = Jacobi_iter(ω,v,rhs_values[j])
        end
        v_values[j] = v
    end
    return v_values[1]
end