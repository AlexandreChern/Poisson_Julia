using CUDA
using Distributed
using DistributedData
using Adapt

gpus = Int(length(devices()))
addprocs(gpus)

gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
    remotecall_wait(device!, p, d)
    p
end

@everywhere include("GPU_struct.jl")

@everywhere coef_eg = coef_GPU_sbp_sat_test(CuArray([3/2 -2 1/2]),
    CuArray([5 5 1/(5-1) 1/(5-1)]),
    CuArray([13/(1/(5-1)) 13/(1/(5-1)) 1 1 -1])
)

@everywhere coef_eg2 = coef_GPU_sbp_sat(CuArray([3/2 -2 1/2]),
    CuArray([5 5 1/(5-1) 1/(5-1)]),
    CuArray([13/(1/(5-1)) 13/(1/(5-1)) 1 1 -1])
)

save_at(2,:coef_eg_3, :(coef_GPU_sbp_sat_test(CuArray([3/2 -2 1/2]),
CuArray([5 5 1/(5-1) 1/(5-1)]),
CuArray([13/(1/(5-1)) 13/(1/(5-1)) 1 1 -1])
)))

save_at(3,:coef_eg_3, :(coef_GPU_sbp_sat_test(CuArray([3/2 -2 1/2]),
CuArray([5 5 1/(5-1) 1/(5-1)]),
CuArray([13/(1/(5-1)) 13/(1/(5-1)) 1 1 -1])
)))

save_at(3,:coef_eg_4, :(cudaconvert(coef_GPU_sbp_sat_test(CuArray([3/2 -2 1/2]),
CuArray([5 5 1/(5-1) 1/(5-1)]),
CuArray([13/(1/(5-1)) 13/(1/(5-1)) 1 1 -1])
))))

@everywhere gpu_test_array = CuArray(randn(3,3))

isbits(coef_eg) # Should return False
isbits(coef_eg2) # Should return False

coef_eg_D = cudaconvert(coef_eg)
coef_eg2_D = cudaconvert(coef_eg2)

isbits(coef_eg_D) # Should return True
isbits(coef_eg2_D) # Should return True


@fetchfrom 2 coef_eg
@fetchfrom 2 coef_eg2

@everywhere coef_eg_D = cudaconvert(coef_eg)
@everywhere coef_eg2_D = cudaconvert(coef_eg2)
@fetchfrom 2 isbits(coef_eg_D) # UndefVarError T not defined
@fetchfrom 2 coef_eg2_D # Same UndefVarError T not defined



save_at(3,:coef_eg_D2, :(cudaconvert(coef_eg)))

get_val_from(3,:coef_eg_D2)