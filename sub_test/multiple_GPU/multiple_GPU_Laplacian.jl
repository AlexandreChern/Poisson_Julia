using CUDA
using Random
devices()
include("split_matrix_free.jl")

gpus = Int(length(devices()))

dims = (2048,2048)
# dims = (1024,1024)
# dims = (16,16)

# dims = (4096,4096)s
# dims = (4096,8192)


Random.seed!(0)
device!(0)
idata = CuArray(randn(Float64,dims) .* 100)
odata = CuArray(randn(Float64,dims) .* 100)


# CuArray doesn't support unified memory yet,
# so allocate our own buffers

@show sizeof(idata)

# Splitting the data into two gpus

dims1 = (dims[1],div(dims[2],2)+1)
dims2 = (dims[1],dims[2]-div(dims[2],2)+1)

# device!(1)

# buf_gpu1 = Mem.alloc(Mem.Unified,dims1[1]*dims[2]*8)
# d_gpu1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims1)
# finalizer(d_gpu1) do _
#     Mem.free(buf_gpu1)
# end


# buf_gpu2 = Mem.alloc(Mem.Unified, dims2[1]*dims2[2]*8)
# d_gpu2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims2)
# finalizer(d_gpu2) do _
#     Mem.free(buf_gpu2)
# end


# buf_gpu1 = Mem.alloc(Mem.Unified,dims1[1]*dims[2]*8)
# d_gpu_out1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims1)
# finalizer(d_gpu_out1) do _
#     Mem.free(buf_gpu1)
# end


# buf_gpu2 = Mem.alloc(Mem.Unified, dims2[1]*dims2[2]*8)
# d_gpu_out2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims2)
# finalizer(d_gpu_out2) do _
#     Mem.free(buf_gpu2)
# end

device!(0)

buf_gpu1 = Mem.alloc(Mem.DeviceBuffer,dims1[1]*dims[2]*8)
d_gpu1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims1)
finalizer(d_gpu1) do _
    Mem.free(buf_gpu1)
end


buf_gpu_out1 = Mem.alloc(Mem.DeviceBuffer,dims1[1]*dims[2]*8)
d_gpu_out1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu_out1), dims1)
finalizer(d_gpu_out1) do _
    Mem.free(buf_gpu_out1)
end

device!(1)
buf_gpu2 = Mem.alloc(Mem.DeviceBuffer, dims2[1]*dims2[2]*8)
d_gpu2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims2)
finalizer(d_gpu2) do _
    Mem.free(buf_gpu2)
end


buf_gpu_out2 = Mem.alloc(Mem.DeviceBuffer, dims2[1]*dims2[2]*8)
d_gpu_out2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu_out2), dims2)
finalizer(d_gpu_out2) do _
    Mem.free(buf_gpu_out2)
end

# d_gpu1 = CuArray(zeros(dims1))
# d_gpu2 = CuArray(zeros(dims2))
# d_gpu_out1 = copy(d_gpu1)
# d_gpu_out2 = copy(d_gpu2)


# copyto!(d_gpu_out1,d_gpu1)
# copyto!(view(d_gpu1,:,1:dims1[2]),view(idata,:,1:dims1[2]))
# copyto!(view(d_gpu2,:,1:dims2[2]),view(idata,:,dims1[2]-1:dims1[2]-2+dims2[2]))
copyto!(d_gpu1,view(idata,:,1:dims1[2]))
copyto!(d_gpu2,view(idata,:,dims1[2]-1:dims1[2]-2+dims2[2]))

# display(d_gpu1)
# display(d_gpu2)

Nx,Ny = size(idata)
h = 1/(Nx-1)
TILE_DIM_1 = 16
TILE_DIM_2 = 16
griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
blockdim_2d = (TILE_DIM_1,TILE_DIM_2)

alpha1 = alpha2 = -13/h
alpha3 = alpha4 = -1
beta = 1

idata_lists = [d_gpu1, d_gpu2]


odata_lists = [d_gpu_out1, d_gpu_out2]

gpu = 1
type = gpu
_Nx,_Ny = size(idata_lists[gpu])
# for _ in 1:100
t1 = @elapsed  for (gpu, dev) in enumerate(devices())
    device!(dev)
    # idata_lists[gpu] .+= gpu
    # type = gpu
    # _Nx,_Ny = size(idata_lists[gpu])
    for _ in 1:10000
        @cuda threads=blockdim_2d blocks=griddim_2d matrix_free_A(idata_lists[gpu],odata_lists[gpu],_Nx,_Ny,type,Val(TILE_DIM_1), Val(TILE_DIM_2))
        # synchronize()
        # odata_lists[gpu] .= idata_lists[gpu] .+ gpu
    end
    synchronize()
end
# end



### Testing Memory Performance

t2 = @elapsed  for (gpu, dev) in enumerate(devices())
    device!(dev)
    for _ in 1:10000
      copyto!(odata_lists[gpu],idata_lists[gpu])
    end
    synchronize()
end


Nx,Ny = size(idata)
t3 = @elapsed for _ in 1:10000
    device!(0)
    @cuda threads=blockdim_2d blocks=griddim_2d matrix_free_A(idata,odata,Nx,Ny,Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
end


t4 = @elapsed begin 
    device!(0)
    for _ in 1:200000
        copyto!(odata,idata)
    end
end

t_4a = @elapsed begin
    device!(0)
    for _ in 1:200000
        copyto!(d_gpu_out1,d_gpu1)
    end
end

t_4b = @elapsed begin
    device!(1)
    for _ in 1:200000
        copyto!(d_gpu_out2,d_gpu2)
    end
end


t5 = @elapsed @sync begin
    @async begin
        device!(0)
        for _ in 1:200000
            # copyto!(odata_lists[1],idata_lists[1])
            copyto!(d_gpu_out1,d_gpu1)
        end
    end
    @async begin
        device!(1)
        for _ in 1:200000
            # copyto!(odata_lists[2],idata_lists[2])
            copyto!(d_gpu_out2,d_gpu2)
        end
    end
end



t6 = @elapsed @sync begin
    @async begin
        device!(0)
        for _ in 1:10000
            # copyto!(odata_lists[1],idata_lists[1])
            copyto!(d_gpu_out1,d_gpu1)
        end
    end
    # @async begin
    #     device!(1)
    #     for _ in 1:10000
    #         # copyto!(odata_lists[2],idata_lists[2])
    #         copyto!(d_gpu_out2,d_gpu2)
    #     end
    # end
end

t7= @elapsed begin
    for _ in 1:10000
        copyto!(odata,idata)
    end
end

Through_put6 = 2*sizeof(d_gpu2) * 10000 / (1024^3 * t6)
Through_put5 = 2*(sizeof(d_gpu2) + sizeof(d_gpu1)) * 10000 / (1024^3 * t5)
Through_put7 = 2*sizeof(idata) * 10000 / (1024^3 * t7)



idata_gpu = CuArray(randn(2048,2048))
odata_gpu = CuArray(randn(2048,2048))

t8 = @elapsed begin
    for _ in 1:200000
        copyto!(odata,idata)
    end
end 


Through_put8 = (sizeof(idata_gpu) + sizeof(odata_gpu))*200000/(1024^3*t8)