using CUDA
using Distributed

gpus = Int(length(devices()))
addprocs(gpus)
@everywhere using CUDA
@everywhere using DistributedData
@everywhere using LinearAlgebra

@everywhere include("assembling_matrix.jl")
@everywhere include("laplacian.jl")

gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
    remotecall_wait(device!, p, d)
    p
end

Random.seed!(0)

@everywhere level = 4 # 2^3 +1 points in each direction
idata_cpu = randn(2^level+1,2^level+1)

@everywhere (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level)

@everywhere coef_p2 = coef_GPU_sbp_sat(CuArray([2. 0]), # The 0 here is only to make them identical arrays
    CuArray([1. -2 1]),
    CuArray([1. -2 1]),
    CuArray([3/2 -2 1/2]),
    CuArray([Nx Ny 1/(Nx-1) 1/(Ny-1)]),
    CuArray([13/(1/(Nx-1)) 13/(1/(Ny-1)) 1 1 -1]))  
coef_p2_D = cudaconvert(coef_p2)

odata_H_tilde_D2 = reshape(H_tilde*-D2*idata_cpu[:],Nx,Ny)
odata_H_tilde_A = reshape(A*idata_cpu[:],Nx,Ny)


odata_cpu = reshape(A*idata_cpu[:],Nx,Ny)

idata_GPU = CuArray(idata_cpu)
odata_GPU = CuArray(zeros(size(idata_GPU)))


