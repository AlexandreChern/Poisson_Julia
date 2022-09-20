using CUDA
using Distributed

gpus = Int(length(devices()))
addprocs(gpus)
@everywhere using CUDA
@everywhere using DistributedData
@everywhere using Distributed
@everywhere using LinearAlgebra
@everywhere using Adapt
@everywhere using Random

@everywhere include("assembling_matrix.jl")
@everywhere include("laplacian.jl")

gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
    remotecall_wait(device!, p, d)
    p
end

gpu_devices = Dict(enumerate(devices()))
gpu_processes = Dict(zip(workers(), CUDA.devices()))
@everywhere workers_dict = Dict(zip(workers(),1:length(workers())))


Random.seed!(0)

@everywhere level = 4 # 2^3 +1 points in each direction
idata_cpu = randn(2^level+1,2^level+1)

@everywhere (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level)

@everywhere coef_p2 = coef_GPU_sbp_sat_v4(CuArray([2. 0]), # The 0 here is only to make them identical arrays
    CuArray([1. -2 1]),
    CuArray([1. -2 1]),
    CuArray([3/2 -2 1/2]),
    CuArray([Nx Ny 1/(Nx-1) 1/(Ny-1)]),
    CuArray([13/(1/(Nx-1)) 13/(1/(Ny-1)) 1 1 -1]))  
coef_p2_D = cudaconvert(coef_p2)
# @everywhere coef_p2_D
@everywhere coef_p2_D = cudaconvert(coef_p2)


odata_H_tilde_D2 = reshape(H_tilde*-D2*idata_cpu[:],Nx,Ny)
odata_H_tilde_A = reshape(A*idata_cpu[:],Nx,Ny)


odata_cpu = reshape(A*idata_cpu[:],Nx,Ny)

idata_GPU = CuArray(idata_cpu)
odata_GPU = CuArray(zeros(size(idata_GPU)))


@everywhere num_blocks = length(devices())
@everywhere sub_block_width = div(Ny,num_blocks)
# @everywhere y_indeces = 1:sub_block_width:Ny
# y_indeces = [1:9,9:17]
@everywhere y_indeces = Vector{UnitRange{Int64}}(undef,num_blocks)
for i in 1:length(y_indeces)
    if i == 1
        y_indeces[i] = 1:sub_block_width+1
    elseif i == length(y_indeces)
        y_indeces[i] = sub_block_width*(i-1):Ny
    else
        y_indeces[i] = (i-1)*sub_block_width:i*sub_block_width+1
    end
end
@everywhere y_indeces

@sync begin
    for (proc,dev) in gpu_processes
        # @spawnat proc begin
        #     @show proc, dev
        #     device!(dev)
        #     idata_GPU_proc = CuArray(zeros(Nx,sub_block_width))
        # end
        save_at(proc,:idata_GPU_proc,:(CuArray(zeros(Nx,sub_block_width))))
    end
end

@sync begin
    for (proc,dev) in gpu_processes
        # @spawnat proc begin
            if proc == workers()[1]
                save_at(proc,:idata_GPU_proc,:(CuArray(zeros(Nx,sub_block_width+1))))
                save_at(proc,:odata_GPU_proc,:(CuArray(zeros(Nx,sub_block_width+1))))
            elseif proc == workers()[end]
                save_at(proc,:idata_GPU_proc,:(CuArray(zeros(Nx,Ny-sub_block_width*(num_blocks-1)+1))))
                save_at(proc,:odata_GPU_proc,:(CuArray(zeros(Nx,Ny-sub_block_width*(num_blocks-1)+1))))
            else
                save_at(proc,:idata_GPU_proc,:(CuArray(zeros(Nx,sub_block_width+2))))
                save_at(proc,:odata_GPU_proc,:(CuArray(zeros(Nx,sub_block_width+2))))
            end
        # end
    end
end

# Copying input data to different processes
@sync begin
    for (proc,dev) in gpu_processes
        @show proc
        # worker_index = workers_dict[proc]
        @spawnat proc begin
            if myid() == workers()[1]
                worker_index = workers_dict[myid()]
                @show worker_index
                idata_GPU_temp = get_val_from(1,:(idata_GPU[:,$(y_indeces[worker_index])]))
                @show size(idata_GPU_temp)
                copyto!(idata_GPU_proc,idata_GPU_temp)
            end
        end
    end
end

# Creating boundaries data on different processes
@async begin 
    for (proc,dev) in gpu_processes
        if proc == workers()[1]
            save_at(proc,:odata_boundaries,[CuArray(zeros(Nx,length(y_indeces[(workers_dict[proc])]))),
                                            CuArray(zeros(3,length(y_indeces[(workers_dict[proc])]))),
                                            CuArray(zeros(3,length(y_indeces[(workers_dict[proc])])))])
        elseif proc == workers()[end]
            save_at(proc,:odata_boundaries,[CuArray(zeros(3,length(y_indeces[(workers_dict[proc])]))),
                                            CuArray(zeros(Nx,length(y_indeces[(workers_dict[proc])]))),
                                            CuArray(zeros(3,length(y_indeces[(workers_dict[proc])])))])
        else
            save_at(proc,:odata_boundaries,[CuArray(zeros(3,length(y_indeces[workers_dict[proc]]))),
                                            CuArray(zeros(3,length(y_indeces[workers_dict[proc]])))])
        end
    end
end


@sync begin
    for (proc,dev) in gpu_processes
        @spawnat proc begin
            # laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,coef_p2_D)
            # @show coef_p2_D
            laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,coef_p2_D)
        end
    end
end


# @sync begin
#     for (proc,dev) in gpu_processes
#         @spawnat proc 
#     end
# end
