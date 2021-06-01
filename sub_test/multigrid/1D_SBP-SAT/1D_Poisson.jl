# Solving -U_xx = π^2 * sin(πx)
# u0 = 0, x = 0 (Dirichlet Boundary Condition)
# u_x = -π, x = 1 (Neumann Boundary Condition)
# exact solution u = sin(π*x)




using LinearAlgebra
using SparseArrays
using Plots

include("diagonal_sbp.jl")

global σ=0

function e(i,n)
    a = spzeros(n)
    a[i] = 1
    return a
end


function exact_u(x)
    return sin.(π*x)
end

function Jacobi_iter(ω,v,f)
    N = length(v)
    # h = v[2] - v[1]
    h = 1/(N+1)
    v_new = copy(v)
    for j = 2:N-1
        v_new[j] = (1-ω) * v[j] + ω * 1/(2 + σ*h^2) * (v[j-1] + v[j+1] + h^2*f[j])
    end
    v_new[1] = (1-ω) * v[1] + ω * 1/(2 + σ*h^2) * (v[2] + h^2*f[1]) 
    v_new[end] = (1-ω) * v[end] +  ω * 1/(2 + σ*h^2) * (v[end-1] + h^2*f[end])
    return v_new
end

# function Jacobi_iter_SBP(v,A,b)
#     v_new = copy(v)
#     v_new = A*v + b
#     return v_new
# end

N_list = [2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10]

h_list = 1 ./ N_list

p = 2
i = 3

n = N_list[i]


(D1, HI1, H1, r1) = diagonal_sbp_D1(p,n,xc=(0,1))
(D2, S0, SN, HI2, H2, r2) = diagonal_sbp_D2(p,n,xc=(0,1))


analy_sol = sin.(r1*π)

e0 = e(1,n+1)
en = e(n+1,n+1)

BS = SN - S0

α = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
γ = 2 # γ > 1 for stability
σ₁ = -γ/(α[p]*h_list[i]) # For left boundary Dirichlet condition
σ₂ = -γ/(α[p]*h_list[i]) # For right boundary Neumann condition
β = 1 # For left boundary Neumann condition

A = D2 + β*HI1*BS'*e0*e0' + σ₁*HI1*e0*e0' + σ₂*HI1*en*en'*D1

b = - π^2 * sin.(r1*π) - σ₂*HI1*en*π

direct_sol = A \ Array(b)

plot(analy_sol)
plot!(direct_sol)



function Linear_Operators(n,p)
    (D1, HI1, H1, r1) = diagonal_sbp_D1(p,n,xc=(0,1))
    (D2, S0, SN, HI2, H2, r2) = diagonal_sbp_D2(p,n,xc=(0,1))

    e0 = e(1,n+1)
    en = e(n+1,n+1)

    BS = SN - S0

    α = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
    γ = 2 # γ > 1 for stability
    σ₁ = -γ/(α[p]*h_list[i]) # For left boundary Dirichlet condition
    σ₂ = -γ/(α[p]*h_list[i]) # For right boundary Neumann condition
    β = 1 # For left boundary Neumann condition

    A = D2 + β*HI1*BS'*e0*e0' + σ₁*HI1*e0*e0' + σ₂*HI1*en*en'*D1
    b = - π^2 * sin.(r1*π) - σ₂*HI1*en*π
    return (A,b,D1,D2)
end



function linear_interpolation(v)
    v_out = zeros(2*length(v)-1)
    v_out[1] = v[1]
    v_out[end] = v[end]
    for i in 2:length(v_out)-1
        if mod(i,2) == 1
            v_out[i] = v[div(i+1,2)]
        else
            v_out[i] = (v[div(i,2)] + v[div(i,2)+1]) / 2
            # v_out[i] = 0
        end
    end
    return v_out
end

function linear_interpolation_matrix(n)
    v_out = spzeros(2*n-1,n)
    for i in 1:2*n-1
        if mod(i,2) == 1
            v_out[i,div(i+1,2)] = 1
        else
            v_out[i,div(i,2):div(i,2)+1] = [1 1] /2
        end
    end
    return v_out
end




function restriction(v)
    v_out = zeros(div(length(v)+1,2))
    v_out[1] = (v[1] + v[2]) / 2
    v_out[end] = (v[end-1] + v[end]) / 2
    for i in 2:length(v_out) - 1
        v_out[i] = (v[i*2-2] + 2*v[i*2-1] + v[i*2]) / 4
    end
    return v_out
end

function restriction_matrix(n)
    v_out = spzeros(div(n+1,2),n)
    v_out[1,1:2] = [1 1] / 2
    v_out[end,end-1:end] = [1 1] / 2
    for i in 2:div(n+1,2) - 1
        v_out[i,2*i-2] = 1/4
        v_out[i,2*i-1] = 1/2
        v_out[i,2*i] = 1/4
    end
    return v_out 
end


function V_cycle()
    ω = 2/3
    N = 2^6
    L = 3
    iter_times = 100
    x = range(0,stop=1,step=1/N)
    v = zeros(N+1)
    (A,b,D1,D2) = Linear_Operators(N,2)
    v_values = Dict(1=>v)
    rhs_values = Dict(1=>b)
    A_values = Dict(1=>A)
    for i in 1:L
        @show i
        if i != L
            (A,b,D1,D2) = Linear_Operators(N,2)
            A_values[i] = A
            for _ in 1:iter_times
                v = Jacobi_iter(ω,v,rhs_values[i])
                # v = Jacobi_iter_SBP(v,A_values[i],rhs_values[i])
            end
            v_values[i] = v
            @show v
            rhs = restriction(rhs_values[i] - A_values[i] * v_values[i]00)
            println()
            @show rhs
            rhs_values[i+1] = rhs
            N = div(N,2)
            v = zeros(N+1)
        else
            (A,b,D1,D2) = Linear_Operators(N,2)
            # @show size(A)
            # @show rhs_values[i]
            # v_values[i] = A \ Vector(rhs_values[i])
            v_values[i] = A \ Vector(b)
        end
        @show v_values[i]
        println()
    end
    println("Pass first part")

    for i in 1:L-1
        j = L - I
        @show v_values[j]
        v_values[j] = v_values[j] + linear_interpolation(v_values[j+1])
        v = v_values[j]
        @show v_values[j]
        for _ in 1:iter_times
            v = Jacobi_iter(ω,v,rhs_values[j])
            # v = Jacobi_iter_SBP(v,A_values[j])
        end
        v_values[j] = v
        @show j
        @show v_values[j]
    end
    return v_values[1], exact_u(x)
end

function plot_results(results)
    plot(results[1])
    plot!(results[2])
end
