# Solving -U_xx = π^2 * sin(πx)
# u0 = 0, x = 0 (Dirichlet Boundary Condition)
# u_x = -π, x = 1 (Neumann Boundary Condition)



using LinearAlgebra
using SparseArrays
using Plots

include("diagonal_sbp.jl")


function e(i,n)
    a = spzeros(n)
    a[i] = 1
    return a
end


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
    return (A,b,D1)
end


function Linear_system()
    
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




function restriction(v)
    v_out = zeros(div(length(v)+1,2))
    v_out[1] = (v[1] + v[2]) / 2
    v_out[end] = (v[end-1] + v[end]) / 2
    for i in 2:length(v_out) - 1
        v_out[i] = (v[i*2-2] + 2*v[i*2-1] + v[i*2]) / 4
    end
    return v_out
end