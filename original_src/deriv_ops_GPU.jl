using CUDA
using Printf
using StaticArrays
using GPUifyLoops: @unroll



function D2x_GPU(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
end