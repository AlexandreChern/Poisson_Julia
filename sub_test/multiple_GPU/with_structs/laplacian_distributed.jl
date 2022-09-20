using Distributed
using CUDA
using DistributedData


if length(workers()) <= 2
    gpus = Int(length(devices()))
    addprocs(gpus)
    @everywhere using CUDA
    @everywhere using DistributedData
    @everywhere using LinearAlgebra
    # @everywhere include("laplacian.jl")

    gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
        remotecall_wait(device!, p, d)
        p
    end
end

@everywhere include("laplacian.jl")
@everywhere using Random
@everywhere Random.seed!(0)


gpu_devices = Dict(enumerate(devices()))
gpu_processes = Dict(zip(workers(), CUDA.devices()))

@everywhere dim_x = dim_y = 2^10
# @everywhere dim_x = dim_y = 16384
# @everywhere dim_x = dim_y = 2^15 # 32768
@everywhere dims = (dim_x,dim_y)


# allocating cu_a, cu_b, cu_c on GPU 0

fetch(save_at(2,:cu_a,:(CuArray(randn(dims)))))
cu_a_local = get_val_from(2,:(cu_a[1:10,1:10]))
fetch(save_at(2,:cu_b,:(CuArray(randn(dims)))))
cu_b_local = get_val_from(2,:(cu_b[1:10,1:10]))
fetch(save_at(2,:cu_c,:(cu_a + cu_b)))
cu_c_local = get_val_from(2,:(cu_c[1:10,1:10]))


cu_c_local ≈ cu_a_local + cu_b_local

save_at(2,:cu_out1,:(CuArray(randn(dims))))

# Show memory allocated on worker 2
@fetchfrom 2 InteractiveUtils.varinfo()


# allocating cu_d, cu_e, cu_f on GPU 1
fetch(save_at(3,:cu_d,:(CuArray(randn(dims)))))
cu_d_local = get_val_from(3,:(cu_d[1:10,1:10]))
fetch(save_at(3,:cu_e,:(CuArray(randn(dims)))))
cu_e_local = get_val_from(3,:(cu_e[1:10,1:10]))
fetch(save_at(3,:cu_f,:(cu_d + cu_e)))
cu_f_local = get_val_from(3,:(cu_f[1:10,1:10]))

cu_f_local ≈ cu_d_local + cu_e_local

save_at(3,:cu_out2,:(CuArray(randn(dims))))

# Show memory allocated on worker 3
@fetchfrom 3 InteractiveUtils.varinfo()

@sync begin
    @spawnat 2 begin
        # device!(0)
        # copy_GPU(cu_b,cu_c)
        laplacian_GPU(cu_b,cu_c)
    end
    @spawnat 3 begin
        # device!(1)
        # copy_GPU(cu_e,cu_f)
        laplacian_GPU(cu_e,cu_f)
    end
    nothing
end

CUDA.allowscalar(false)
begin
   cu_c_right = get_val_from(2,:(cu_c[:,end-1]))
   cu_f_left = get_val_from(3,:(cu_f[:,2]))
end


@sync begin
    @spawnat 2 begin
        cu_f_left = get_val_from(3,:(cu_f[:,2]))
        copyto!(cu_c[:,end],cu_f_left)
    end

    @spawnat 3 begin
        cu_c_right = get_val_from(2,:(cu_c[:,end-1]))
        copyto!(cu_f[:,1], cu_c_right)
    end
end


# get_val_from(2,:(cu_c[1:10,1:10]))

get_val_from(2,:cu_c)
get_val_from(3,:cu_f)

function distributed_laplacian()
    @async begin
        @spawnat 2 begin
            # device!(0)
            # copy_GPU(cu_b,cu_c)
            laplacian_GPU(cu_b,cu_c)
        end
        @spawnat 3 begin
            # device!(1)
            # copy_GPU(cu_e,cu_f)
            laplacian_GPU(cu_e,cu_f)
        end
        nothing
    end
    synchronize()
    @async begin
        @spawnat 2 begin
            cu_f_left = get_val_from(3,:(cu_f[:,2]))
            # cu_c[:,end] .= cu_f_left
            copyto!(cu_c[:,end],cu_f_left)
        end
    
        @spawnat 3 begin
            cu_c_right = get_val_from(2,:(cu_c[:,end-1]))
            # cu_f[:,1] .= cu_c_right
            copyto!(cu_f[:,1],cu_c_right)
        end
    end
    synchronize()
end

@elapsed for _ in 1:100
    distributed_laplacian()
end


function distributed_laplacian_v2()
    t1 = @task begin
        @spawnat 2 begin
            # device!(0)
            # copy_GPU(cu_b,cu_c)
            laplacian_GPU(cu_b,cu_c)
            boundary(cu_b,cu_c;orientation=1)
        end
        @spawnat 3 begin
            # device!(1)
            # copy_GPU(cu_e,cu_f)
            laplacian_GPU(cu_e,cu_f)
        end
        nothing
    end
    # @show istaskstarted(t1)
    schedule(t1)
    # wait(t1)
    # @show istaskdone(t1)
    synchronize()
    t2 = @task begin
        @spawnat 2 begin
            cu_f_left = get_val_from(3,:(cu_f[:,2]))
            # cu_c[:,end] .= cu_f_left
            copyto!(cu_c[:,end],cu_f_left)
        end
    
        @spawnat 3 begin
            cu_c_right = get_val_from(2,:(cu_c[:,end-1]))
            # cu_f[:,1] .= cu_c_right
            copyto!(cu_f[:,1],cu_c_right)
        end
    end
    # if istaskdone(t1)
    #     schedule(t2)
    #     wait(t2)
    # end
    schedule(t2)
    # wait(t1)
    # wait(t2)
    synchronize()
end

@elapsed for _ in 1:100
    distributed_laplacian_v2()
end



device!(0)
cu_loc1 = CuArray(randn(dims));
cu_loc2 = CuArray(randn(dims));

@elapsed for _ in 1:100
    laplacian_GPU(cu_loc1,cu_loc2);
end


@elapsed @spawnat 3 begin
    for _ in 1:100
        laplacian_GPU(cu_e,cu_f)
    end
end