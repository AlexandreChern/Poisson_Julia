using CUDA
using Random
devices()
include("split_matrix_free.jl")

gpus = Int(length(devices()))

dims = (8,8)

Random.seed!(0)
idata = randn(Float64,dims) .* 100


# CuArray doesn't support unified memory yet,
# so allocate our own buffers

@show sizeof(idata)

# Splitting the data into two gpus

dims1 = (dims[1],div(dims[2],2)+1)
dims2 = (dims[1],dims[2]-div(dims[2],2)+1)

# device!(1)

buf_gpu1 = Mem.alloc(Mem.Unified,dims1[1]*dims[2]*8)
d_gpu1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims1)
finalizer(d_gpu1) do _
    Mem.free(buf_gpu1)
end


buf_gpu2 = Mem.alloc(Mem.Unified, dims2[1]*dims2[2]*8)
d_gpu2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims2)
finalizer(d_gpu2) do _
    Mem.free(buf_gpu2)
end

# d_gpu_out_1 = copy(d_gpu1)
# d_gpu_out_2 = copy(d_gpu2)

buf_gpu1 = Mem.alloc(Mem.Unified,dims1[1]*dims[2]*8)
d_gpu_out1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims1)
finalizer(d_gpu_out1) do _
    Mem.free(buf_gpu1)
end


buf_gpu2 = Mem.alloc(Mem.Unified, dims2[1]*dims2[2]*8)
d_gpu_out2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims2)
finalizer(d_gpu_out2) do _
    Mem.free(buf_gpu2)
end


# copyto!(d_gpu_out1,d_gpu1)
copyto!(view(d_gpu1,:,1:dims1[2]),view(idata,:,1:dims1[2]))
copyto!(view(d_gpu2,:,1:dims2[2]),view(idata,:,dims1[2]-1:dims1[2]-2+dims2[2]))

display(d_gpu1)
display(d_gpu2)

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

for (gpu, dev) in enumerate(devices())
    device!(dev)
    idata_lists[gpu] .+= gpu
    type = gpu
    _Nx,_Ny = size(idata_lists[gpu])
    @cuda threads=blockdim_2d blocks=griddim_2d matrix_free_A(idata_lists[gpu],odata_lists[gpu],_Nx,_Ny,type,Val(TILE_DIM_1), Val(TILE_DIM_2))
end