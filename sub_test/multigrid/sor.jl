using Plots: length
using Base: sign_mask
using Plots
using LinearAlgebra

A = [4 -1 -6 0;
-5 -4 10 8;
0 9 4 -2;
1 0 -7 5]

b = [2;21;-12;-6]

ω = 1/2

ϕ = zeros(length(b))

function sor(A,b,ϕ,ω)
    (m,n) = size(A)
    iter_times = 0
    while norm(A*ϕ-b) >= 1e-15
    # for _ in 1:100
        for i in 1:n
            σ = 0
            for j in 1:n
                if j != i
                    σ = A[i,j]*ϕ[j] + σ
                end
            end
            ϕ[i] = (1-ω) * ϕ[i] + ω/A[i,i]*(b[i] - σ)
        end
        iter_times += 1
        if iter_times > 10000
            break
        end
    end
    return (ϕ,iter_times)
end


function Jacobi(A,b,ϕ,ω)
    (m,n) = size(A)
    iter_times = 0
    while norm(A*ϕ-b) >= 1e-16
    # for _ in 1:10
        for i in 1:n
            σ = 0
            for j in 1:n
                if j != i
                    σ = A[i,j]*ϕ[j] + σ
                end
            end
            # ϕ[i] = (1-ω) * ϕ[i] + ω/A[i,i]*(b[i] - σ)
            ϕ[i] = 1/A[i,i]*(b[i]-σ)
        end
        iter_times += 1
        if iter_times >= 10000
            break
        end
    end
    return (ϕ,iter_times)
end

A = [4 -1 -6 0;
-5 -4 10 8;
0 9 4 -2;
1 0 -7 5]

b = [2;21;-12;-6]

ω = 1/2

ϕ = zeros(length(b))
sor(A,b,ϕ,1/3)
ϕ = zeros(length(b))
sor(A,b,ϕ,ω)
ϕ = zeros(length(b))
sor(A,b,ϕ,1)

ϕ = zeros(length(b))
Jacobi(A,b,ϕ,ω)