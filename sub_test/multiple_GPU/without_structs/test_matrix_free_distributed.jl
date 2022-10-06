using CUDA
using Distributed

gpus = Int(length(devices()))
if length(gpus) == 1
    addprocs(gpus)
end
@everywhere using CUDA
@everywhere using DistributedData
@everywhere using Distributed
@everywhere using LinearAlgebra
@everywhere using Adapt
@everywhere using Random

@everywhere include("assembling_matrix.jl")
@everywhere include("laplacian.jl")

println("Finish loading packages everywhere!")
gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
    remotecall_wait(device!, p, d)
    p
end

gpu_devices = Dict(enumerate(devices()))
gpu_processes = Dict(zip(workers(), CUDA.devices()))
@everywhere workers_dict = Dict(zip(workers(),1:length(workers())))


Random.seed!(0)

@everywhere level = 12 # 2^3 +1 points in each direction
idata_cpu = randn(2^level+1,2^level+1)

println("Starting Assembling Matrix in the main process")
(A,D2,b,H_tilde,Nx,Ny) = Assembling_matrix(level)

A_d = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
b_d = CuArray(b)
out_d = similar(b_d)
out_c = similar(b)

@everywhere Nx,Ny

for worker in workers()
    @fetchfrom worker Nx,Ny
end


@everywhere hx = 1/(Nx-1)
@everywhere hy = 1/(Ny-1)

# @everywhere coef_p2 = coef_GPU_sbp_sat_v4(CuArray([2. 0]), # The 0 here is only to make them identical arrays
#     CuArray([1. -2 1]),
#     CuArray([1. -2 1]),
#     CuArray([3/2 -2 1/2]),
#     CuArray([Nx Ny 1/(Nx-1) 1/(Ny-1)]),
#     CuArray([13/(1/(Nx-1)) 13/(1/(Nx-1)) 1 1 -1]))  
# coef_p2_D = cudaconvert(coef_p2)
# # @everywhere coef_p2_D
# @everywhere coef_p2_D = cudaconvert(coef_p2)


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

# @sync begin
#     for (proc,dev) in gpu_processes
#         # @spawnat proc begin
#         #     @show proc, dev
#         #     device!(dev)
#         #     idata_GPU_proc = CuArray(zeros(Nx,sub_block_width))
#         # end
#         save_at(proc,:idata_GPU_proc,:(CuArray(zeros(Nx,sub_block_width))))
#     end
# end

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

# sum_2 = @fetchfrom 2 sum(idata_GPU_proc[:,1:end-1])
# sum_3 = @fetchfrom 3 sum(idata_GPU_proc[:,2:end])

# sum_1 = sum(idata_GPU)
# @assert sum_1 ≈ sum_2 + sum_3

# Creating boundaries data on host process
odata_boundaries_local = [CuArray(zeros(Nx,3)),CuArray(zeros(3,Ny)),CuArray(zeros(Nx,3)),CuArray(zeros(3,Ny))]

# Creating boundaries data on different processes
@async begin 
    for (proc,dev) in gpu_processes
        if proc == workers()[1]
            save_at(proc,:odata_boundaries,[CuArray(zeros(Nx,3)),
                                            CuArray(zeros(3,length(y_indeces[(workers_dict[proc])]))),
                                            CuArray(zeros(3,length(y_indeces[(workers_dict[proc])])))])
        elseif proc == workers()[end]
            save_at(proc,:odata_boundaries,[CuArray(zeros(3,length(y_indeces[(workers_dict[proc])]))),
                                            CuArray(zeros(Nx,3)),
                                            CuArray(zeros(3,length(y_indeces[(workers_dict[proc])])))])
        else
            save_at(proc,:odata_boundaries,[CuArray(zeros(3,length(y_indeces[workers_dict[proc]]))),
                                            CuArray(zeros(3,length(y_indeces[workers_dict[proc]])))])
        end
    end
end


@everywhere REPEAT_TIMES = 10000
elapsed_multi_GPUs = @elapsed @sync begin
    for (proc,dev) in gpu_processes
        # @spawnat proc begin
        #     # laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,coef_p2_D)
        #     # @show coef_p2_D
        #     laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,Nx,Ny,hx,hy)
        # end
        if proc == workers()[1]
            @spawnat proc begin
                for _ in 1:REPEAT_TIMES
                    laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,Nx,Ny,hx,hy)
                    # # Boundary calculation
                    boundary((@view idata_GPU_proc[:,1:3]),odata_boundaries[1],Nx,Ny,hx,hy;orientation=1,type=1)
                    boundary((@view idata_GPU_proc[1:3,:]),odata_boundaries[2],Nx,Ny,hx,hy;orientation=2,type=1)
                    boundary((@view idata_GPU_proc[end-2:end,:]),odata_boundaries[3],Nx,Ny,hx,hy;orientation=4,type=1)
                    # # adding boundary data into odata_GPU_proc
                    # # @inbounds odata_GPU_proc[:,1:3] .+= odata_boundaries[1][:,1:3]
                    copyto!((@view odata_GPU_proc[:,1:3]), (@view odata_GPU_proc[:,1:3]) .+ (@view odata_boundaries[1][:,1:3]))
                    # # @inbounds odata_GPU_proc[1,:] .+= odata_boundaries[2][1,:] # this line of the code is extremely slow
                    copyto!((@view odata_GPU_proc[1,:]), (@view odata_GPU_proc[1,:]) .+ (@view odata_boundaries[2][1,:]))
                    # # @inbounds odata_GPU_proc[end,:] .+= odata_boundaries[3][end,:]
                    copyto!((@view odata_GPU_proc[end,:]), (@view odata_GPU_proc[end,:]) .+ (@view odata_boundaries[3][end,:]))
                end
            end
        elseif proc == workers()[end]
            @spawnat proc begin
                for _ in 1:REPEAT_TIMES
                    laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,Nx,Ny,hx,hy)  
                    # # Boundary calculation          
                    boundary((@view idata_GPU_proc[1:3,:]),odata_boundaries[1],Nx,Ny,hx,hy;orientation=2,type=3)
                    boundary((@view idata_GPU_proc[:,end-2:end]),odata_boundaries[2],Nx,Ny,hx,hy;orientation=3,type=3)
                    boundary((@view idata_GPU_proc[end-2:end,:]),odata_boundaries[3],Nx,Ny,hx,hy;orientation=4,type=3)
                    # # adding boundary data into odata_GPU_proc
                    # # odata_GPU_proc[:,end-2:end] .+= odata_boundaries[2][:,end-2:end]

                    copyto!((@view odata_GPU_proc[:,end-2:end]),(@view odata_GPU_proc[:,end-2:end]) .+ (@view odata_boundaries[2][:,end-2:end]))
                    # # odata_GPU_proc[1,:] .+= odata_boundaries[1][1,:] # this line of code is extremely slow
                    copyto!((@view odata_GPU_proc[1,:]), (@view odata_GPU_proc[1,:]) .+ (@view odata_boundaries[1][1,:]))
                    copyto!((@view odata_GPU_proc[end,:]), (@view odata_GPU_proc[end,:]) .+ (@view odata_boundaries[3][end,:]))
                end
            end
        else
            @spawnat proc begin
                for _ in 1:REPEAT_TIMES
                    laplacian_GPU_v2(idata_GPU_proc,odata_GPU_proc,Nx,Ny,hx,hy)            
                    boundary(idata_GPU_proc,odata_boundaries[1],Nx,Ny,hx,hy;orientation=2,type=2)
                    boundary(idata_GPU_proc,odata_boundaries[2],Nx,Ny,hx,hy;orientation=4,type=2)
                    copyto!((@view odata_GPU_proc[1,:]), (@view odata_GPU_proc[1,:]) .+ (@view odata_boundaries[1][1,:]))
                    copyto!((@view odata_GPU_proc[end,:]), (@view odata_GPU_proc[end,:]) .+ (@view odata_boundaries[2][end,:]))
                end
            end
        end
    end
end

@elapsed for _ in 1:REPEAT_TIMES
    size(idata_GPU)
end

elapsed_single_GPU = @elapsed for _ in 1:REPEAT_TIMES
    laplacian_GPU_v2(idata_GPU,odata_GPU,Nx,Ny,hx,hy)
    boundary(idata_GPU,odata_boundaries_local[1],Nx,Ny,hx,hy;orientation=1,type=1)
    boundary(idata_GPU,odata_boundaries_local[2],Nx,Ny,hx,hy;orientation=2,type=2)
    boundary(idata_GPU,odata_boundaries_local[3],Nx,Ny,hx,hy;orientation=3,type=3)
    boundary(idata_GPU,odata_boundaries_local[4],Nx,Ny,hx,hy;orientation=4,type=2)
    copyto!((@view odata_GPU[:,1:3]),(@view odata_GPU[:,1:3]) .+= odata_boundaries_local[1])
    copyto!((@view odata_GPU[1:3,:]),(@view odata_GPU[1:3,:]) .+= odata_boundaries_local[2]) 
    copyto!((@view odata_GPU[:,end-2:end]),(@view odata_GPU[:,end-2:end]) .+= odata_boundaries_local[3])
    copyto!((@view odata_GPU[end-2:end,:]),(@view odata_GPU[end-2:end,:]) .+= odata_boundaries_local[4]) 
end

@fetchfrom 2 device(odata_GPU_proc)
@fetchfrom 2 device.(odata_boundaries)


@fetchfrom 2 odata_GPU_proc
@fetchfrom 2 odata_boundaries[1]
@fetchfrom 2 odata_boundaries[2]
@fetchfrom 2 odata_boundaries[3]


@fetchfrom 3 device(odata_GPU_proc)
@fetchfrom 3 device.(odata_boundaries)
@fetchfrom 3 odata_GPU_proc
@fetchfrom 3 odata_boundaries[1]
@fetchfrom 3 odata_boundaries[2]
@fetchfrom 3 odata_boundaries[3]
# @sync begin
#     for (proc,dev) in gpu_processes
#         @spawnat proc 
#     end
# end

sum_1 = sum(odata_H_tilde_A)

sum_3 = @fetchfrom 3 sum(odata_GPU_proc)
sum_2 = @fetchfrom 2 sum(odata_GPU_proc)

@assert sum_1 ≈ sum_2 + sum_3

odata_GPU_proc_3 = @fetchfrom 3 odata_GPU_proc[:,2:end]
odata_GPU_proc_2 = @fetchfrom 2 odata_GPU_proc[:,1:end-1]

odata_GPU_cat = hcat(odata_GPU_proc_2,odata_GPU_proc_3)

diff = Array(odata_GPU_cat) .- odata_H_tilde_A
extrema(diff)

findall(diff .== maximum(diff))
findall(diff .== minimum(diff))