using CUDA
using Distributed
using DistributedData
using Adapt


struct coef_GPU_sbp_sat{T}
    BS::T
    grid::T
    sat::T
end

struct coef_GPU_sbp_sat_test{CuArray}
    BS::CuArray
    grid::CuArray
    sat::CuArray
end

Adapt.@adapt_structure coef_GPU_sbp_sat
Adapt.@adapt_structure coef_GPU_sbp_sat_test


