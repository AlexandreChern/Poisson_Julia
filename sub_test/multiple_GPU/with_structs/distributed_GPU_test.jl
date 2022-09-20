using Distributed
using CUDA
using DistributedData


gpus = Int(length(devices()))
addprocs(gpus)
@everywhere using CUDA
@everywhere using DistributedData
@everywhere using LinearAlgebra
@everywhere include("laplacian.jl")

gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
    remotecall_wait(device!, p, d)
    p
end

gpu_devices = Dict(enumerate(devices()))
gpu_processes = Dict(zip(workers(), CUDA.devices()))

@everywhere dim_x = dim_y = 16
@everywhere dims = (dim_x,dim_y)

cu_a = remotecall(CuArray,gpuworkers[1],randn(dims))
cu_b = remotecall(CuArray,gpuworkers[1],randn(dims))

cu_c = remotecall(CuArray,gpuworkers[2],zeros(dims))
cu_d = remotecall(CuArray,gpuworkers[2],zeros(dims))

fetch_a = remotecall_fetch(cu_a -> fetch(cu_a),2,cu_a)
fetch_b = remotecall_fetch(cu_b -> fetch(cu_b),3,cu_b)

# cu_sum = remotecall_fetch(sum,gpuworkers[1],cu_a,cu_b)
# remotecall_fetch(cu_sum->fetch(cu_sum),2)

# work_gpu_dict = Dict(zip(workers(), CUDA.devices()))
# @sync begin
#     @spawnat gpuworkers[1] begin
#         # device!(0)
#         work_gpu_dict[gpuworkers[1]]
#         cu_a .= cu_b
#     end
# end

# asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
#     device!(d)
#     for _ in 1:100
#         sum(cu_a)
#     end
# end

save_at(2,:cu_a,:(CuArray(randn(dims))))
cu_a_local = get_val_from(2,:(cu_a[1:10,1:10]))
save_at(2,:cu_b,:(CuArray(randn(dims))))
cu_b_local = get_val_from(2,:(cu_b[1:10,1:10]))
save_at(2,:cu_c,:(cu_a + cu_b))
cu_c_local = get_val_from(2,:(cu_c[1:10,1:10]))
#how to perform cu_c = cu_a + cu_b and save on process 2


get_val_from(2,:(device(cu_c)))
save_at(2,:cu_out1,:(CuArray(randn(dims))))


save_at(3,:cu_d,:(CuArray(randn(dims))))
cu_d_local = get_val_from(3,:(cu_d[1:10,1:10]))
save_at(3,:cu_e,:(CuArray(randn(dims))))
cu_e_local = get_val_from(3,:(cu_e[1:10,1:10]))
save_at(3,:cu_f,:(cu_d + cu_e))
cu_f_local = get_val_from(3,:(cu_f[1:10,1:10]))




get_val_from(3,:(device(cu_f)))
save_at(3,:cu_out2,:(CuArray(randn(dims))))


@everywhere function test(cu_b, cu_a)
    for _ in 1:100000
        cu_b .= cu_a
    end
end

@everywhere function func_2(cu_out, cu_var_1,cu_var_2)
    for _ in 1:2000
        cu_out .= cu_var_1 .* cu_var_2
    end
end



let 
    _cu_a_local = CuArray(randn(dims))
    _cu_b_local = CuArray(randn(dims))
    _cu_c_local = CuArray(randn(dims))

    @elapsed func_2(_cu_c_local,_cu_b_local,_cu_a_local)
end

# @elapsed get_val_from(2,:(test(cu_b,cu_a)))
# @elapsed get_val_from(3,:(test(cu_e,cu_d)))

let 
    t_elapsed = @elapsed @sync begin
        @spawnat 2 begin
            device!(0)
            get_val_from(2,:(func_2(cu_out1,cu_b,cu_a)))
        end
        @spawnat 3 begin
            device!(1)
            get_val_from(3,:(func_2(cu_out2,cu_e,cu_d)))
        end
    end
    @show t_elapsed


    _cu_a_local = get_val_from(2,:(cu_a[1:10,1:10]))
    _cu_b_local = get_val_from(2,:(cu_b[1:10,1:10]))
    _cu_out1_local = get_val_from(2,:(cu_out1[1:10,1:10]))

    _cu_d_local = get_val_from(3,:(cu_d[1:10,1:10]))
    _cu_e_local = get_val_from(3,:(cu_e[1:10,1:10]))
    _cu_out2_local = get_val_from(3,:(cu_out2[1:10,1:10]))

    @show norm(_cu_out1_local - _cu_a_local .* _cu_b_local)
    @show norm(_cu_out2_local - _cu_d_local .* _cu_e_local)
    
end


@elapsed @sync begin
    @spawnat 2 begin
        device!(0)
        get_val_from(2,:(func_2(cu_out1,cu_b,cu_a)))
    end
    @spawnat 3 begin
        device!(1)
        get_val_from(3,:(func_2(cu_out2,cu_e,cu_d)))
    end
    nothing
end


@elapsed @sync begin
    Threads.@spawn begin
        device!(0)
        for _ in 1:100
            sin.(CuArray(randn(1024,1024)))
        end
        synchronize()
    end
    Threads.@spawn begin
        device!(1)
        for _ in 1:100
            sin.(CuArray(randn(1024,1024)))
        end
        synchronize()
    end
end

