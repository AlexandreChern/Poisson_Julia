using SparseArrays
using LinearAlgebra
using Plots
using Interpolations


L = 2
k = 2

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
    h = v[2] - v[1]
    v_new = copy(v)
    for j = 2:N
        v_new[j] = (1-ω) * v[j] + ω * 1/2 * (v[j-1] + v[j+1] + h^2*f)
    end
    return v_new
end


function multi_grid()

end