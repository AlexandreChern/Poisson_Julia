using LinearAlgebra
using SparseArrays

function linear_interpolation(v)
    len_v = length(v)
    v_interpolated = zeros(2*len_v+1)
    for i in 1:2*len_v+1
        if i%2 == 0
            v_interpolated[i] = (v[div(i,2)])
        elseif i == 1 
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
    for i in eachindex(f_weighted)
        f_weighted[i] = (f[2*i-1] + 2*f[2*i] + f[2*i+1])/4
    end
    return f_weighted
end