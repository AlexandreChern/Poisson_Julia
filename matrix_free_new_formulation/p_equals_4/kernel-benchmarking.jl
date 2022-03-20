include("Poisson.jl")
include("matrix-free-p4.jl")
include("matrix-free-p2.jl")
include("matrix-free-p4-GPU.jl")


repetitions = 10000

level = 13
i = j = level

h_list_x = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13, 1/2^14]
h_list_y = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13, 1/2^14]

m_list = 1 ./h_list_x;
n_list = 1 ./h_list_y;

N_x = Integer(m_list[i]);
N_y = Integer(n_list[j]);

Nx = N_x + 1
Ny = N_y + 1
hx = h_list_x[i]
hy = h_list_y[j]


idata = randn(Nx,Ny)
odata = zeros(Nx,Ny)

idata_GPU = CuArray(idata)
odata_GPU = CuArray(zeros(Nx,Ny))


odata_GPU_NSWE_tmp = odata_GPU_NSWE(
    CuArray(zeros(4,Nx)),
    CuArray(zeros(4,Nx)),
    CuArray(zeros(Ny,4)),
    CuArray(zeros(Ny,4)),
    CuArray(zeros(4,Nx)),
    CuArray(zeros(4,Nx)),
    CuArray(zeros(Ny,4)),
    CuArray(zeros(Ny,4))
)


hx=hy = 1/(Nx-1)
TILE_DIM_1 = 32
TILE_DIM_2 = 32
griddim_2D = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
blockdim_2D = (TILE_DIM_1,TILE_DIM_2)

TILE_DIM = 256
griddim_1D = div(Nx+TILE_DIM-1,TILE_DIM)
blockdim_1D = TILE_DIM

grid_info_GPU = grid_info(
    griddim_1D,
    blockdim_1D,
    griddim_2D,
    blockdim_2D
)

coeff = cudaconvert(CuArray(Array([1.0,2.0,1.0])))

# Precomipling kernels
@cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p4_GPU_kernel_v3(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2
@cuda threads=blockdim_2D blocks=griddim_2D D2_split_dev_p2(idata_GPU,odata_GPU,coeff,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
@cuda threads=blockdim_2D blocks=griddim_2D D2_split_dev_p2_v2(idata_GPU,odata_GPU,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
@cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p2_GPU_kernel(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2

# Start Benchmarking

time_D2_p4 = @elapsed for _ in 1:repetitions
    @cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p4_GPU_kernel_v3(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2
end


time_D2_p2 = @elapsed for _ in 1:repetitions
    @cuda threads=blockdim_2D blocks=griddim_2D D2_split_dev_p2(idata_GPU,odata_GPU,coeff,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
end

time_D2_p2_v2 = @elapsed for _ in 1:repetitions
    @cuda threads=blockdim_2D blocks=griddim_2D D2_split_dev_p2_v2(idata_GPU,odata_GPU,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
end

time_D2_p2_v3 = @elapsed for _ in 1:repetitions
    @cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p2_GPU_kernel(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2
end

time_D2_p4 = @elapsed for _ in 1:repetitions
    @cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p4_GPU_kernel_v3(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2
end




through_put_matrix_free_p4 = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_p4)
through_put_matrix_free_p2 = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_p2)
through_put_matrix_free_p2_v2 = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_p2_v2)
through_put_matrix_free_p2_v3 = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_p2_v3)




@show time_D2_p4
@show time_D2_p2
@show time_D2_p2_v2
@show time_D2_p2_v3


@show through_put_matrix_free_p4
@show through_put_matrix_free_p2
@show through_put_matrix_free_p2_v2
@show through_put_matrix_free_p2_v3