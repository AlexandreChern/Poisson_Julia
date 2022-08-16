using CUDA
devices()
include("split_matrix_free.jl")

gpus = Int(length(devices()))

dims = (8,8)

idata = randn(Float64,dims) .* 100


# CuArray doesn't support unified memory yet,
# so allocate our own buffers


buf_gpu1 = Mem.alloc(Mem.Unified, sizeof(a))
d_gpu1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims)
finalizer(d_gpu1) do _
    Mem.free(buf_gpu1)
end


buf_gpu2 = Mem.alloc(Mem.Unified, sizeof(a))
d_gpu2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims)
finalizer(d_gpu2) do _
    Mem.free(buf_gpu2)
end

d_gpu_out_1 = copy(d_gpu1)
d_gpu_out_2 = copy(d_gpu2)

copyto!(view(d_gpu1,:,1:4),view(a,:,1:4))
copyto!(view(d_gpu2,:,5:8),view(a,:,5:8))

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


odata_lists = [d_gpu_out_1, d_gpu_out_2]

for (gpu, dev) in enumerate(devices())
    device!(dev)
    idata_lists[gpu] .+= gpu
    @cuda threads=blockdim_2d blocks=griddim_2d matrix_free_A(idata_lists[gpu],odata_lists[gpu],Nx,Ny,Val(TILE_DIM_1), Val(TILE_DIM_2))
end