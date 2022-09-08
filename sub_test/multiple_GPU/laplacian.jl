using CUDA
using Adapt

##################################################################################
#   Orientation in 2D:     #       Types of sub-domains
#   West: 1, East: 3       #       Leftmost: 1  Interior: 2 Rightmost: 3
#   South: 2, North: 4     #       
#                          #
#   ###### 4 ######        #       #########################################
#   #             #        #       #     |     |     |               |     #
#   #             #        #       #     |     |     |               |     #
#   1             3        #       #  1  |  2  | ... |               |  3  #
#   #             #        #       #     |     |     |               |     #
#   #             #        #       #     |     |     |               |     #
#   ###### 2 ######        #       #########################################
#                          #
##################################################################################

function copy_kernel(idata,odata,Nx,Ny)
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy

    if 1 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = idata[i,j]
    end
    return nothing
end


function copy_GPU(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d copy_kernel(idata,odata,Nx,Ny)
end


function laplacian_GPU(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d laplacian_kernel(idata,odata,Nx,Ny)
end

function laplacian_kernel(idata,odata,Nx,Ny)
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy

    if 1 <= i <= Nx && 1 <= j <= Ny
        if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
            odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
        else
            # odata[i,j] += 1
            odata[i,j] = 0
        end
    end
    return nothing
end


function laplacian_GPU_v2(idata,odata,coef_D)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d laplacian_kernel_v2(idata,odata,coef_D)
end

function laplacian_kernel_v2(idata,odata,coef_D)
    tidx = threadIdx().x
    tidy = threadIdx().y
    Nx = coef_D.grid[1]
    Ny = coef_D.grid[2]
    hx = coef_D.grid[3]
    hy = coef_D.grid[4]
    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy

    if 1 <= i <= Nx && 1 <= j <= Ny
        if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
            odata[i,j] = -(idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) # - representing -D2
        else
            # odata[i,j] += 1
            odata[i,j] = 0
        end
    end
    return nothing
end


# function boundary(idata,odata;orientation=1,type=1)
#     Nx,Ny = size(idata)
#     h = 1/(Nx-1)
#     TILE_DIM_1D = 16
#     blockdim_1D = TILE_DIM_1D
#     griddim_1D = div(Nx,TILE_DIM_1D) + 1
#     boundary_kernel = boundary_kernels[orientation,type]
#     @cuda threads=blockdim_1D blocks=griddim_1D boundary_kernel_1_1(idata,odata,Nx,Ny)
# end


# function boundary_kernel_1_1(idata,odata,Nx,Ny)
#     tidx = threadIdx().x
#     i = (blockIdx().x - 1) * blockDim().x + tidx

#     if 1 <= i <= Nx
#         # odata[i,1] .= 1
#         # odata[i,2] .= 2
#         odata[i,1] = 1
#         # odata[i,2] = 2
#     end
#     return nothing
# end

function boundary_kernel_1_1(idata,odata,coef_D)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    Nx = coef_D.grid[1]
    h = coef_D.grid[3]
    tau_W = coef_D.sat[1]
    beta = coef_D.sat[5]
    if 2 <= idx <= Nx - 1
        odata[idx,1] = (-(idata[idx-1,1] - 2*idata[idx,1] + idata[idx+1,1] + idata[idx,1] - 2*idata[idx,2] + idata[idx,3]) + 2 * beta * (1.5 * idata[idx,1]) + 2 * tau_W * idata[idx,1] * h) / 2
        # (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        odata[idx,2] = (2 * beta * (-1 * idata[idx,1]))
        odata[idx,3] = (0.5 * beta * idata[idx,1])
    end
    sync_threads()
    if idx == 1 || idx == Nx
        odata[idx,1] =  (2 * beta * (1.5 * idata[idx,1]) + 2 * tau_W * (idata[idx,1]) * h )/ 4
        odata[idx,2] = (2 * beta * (-1 * idata[idx,1])) / 2
        odata[idx,3] = (0.5 * beta * idata[idx,1]) / 2
    end
    return nothing
end


function boundary_kernel_2_1(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_3_1(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_4_1(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_1_2(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_2_2(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_3_2(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_4_2(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_1_3(idata,odata,coef_D)
    return nothing
end

function boundary_kernel_2_3(idata,odata,coef_D)
    return nothing
end


function boundary_kernel_3_3(idata,odata,coef_D)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    Nx = coef_D.grid[1]
    h = coef_D.grid[3]
    tau_E = coef_D.sat[2]
    beta = coef_D.sat[5]
    if 2 <= idx <= Nx - 1
        odata[idx,end] = (-(idata[idx-1,end] - 2*idata[idx,end] + idata[idx+1,end] + idata[idx,end] - 2*idata[idx,end-1] + idata[idx,end-2]) + 2 * beta * (1.5 * idata[idx,end]) + 2 * tau_E * idata[idx,end] * h) / 2
        # (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        odata[idx,end-1] = (2 * beta * (-1 * idata[idx,end]))
        odata[idx,end-2] = (0.5 * beta * idata[idx,end])
    end
    sync_threads()
    if idx == 1 || idx == Nx
        odata[idx,end] =  (2 * beta * (1.5 * idata[idx,end]) + 2 * tau_E * (idata[idx,end]) * h )/ 4
        odata[idx,end-1] = (2 * beta * (-1 * idata[idx,end])) / 2
        odata[idx,end-2] = (0.5 * beta * idata[idx,end]) / 2
    end
    return nothing
end

function boundary_kernel_4_3(idata,odata,coef_D)
    return nothing
end

boundary_kernels = [boundary_kernel_1_1 boundary_kernel_2_1 boundary_kernel_3_1 boundary_kernel_4_1;
boundary_kernel_2_1 boundary_kernel_2_2 boundary_kernel_3_2 boundary_kernel_4_2;
boundary_kernel_3_1 boundary_kernel_3_2 boundary_kernel_3_3 boundary_kernel_4_3
]


function boundary(idata,odata,coef_D;orientation=1,type=1)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1D = 16
    blockdim_1D = TILE_DIM_1D
    griddim_1D = div(Nx,TILE_DIM_1D) + 1
    boundary_kernel = boundary_kernels[type,orientation]
    @cuda threads=blockdim_1D blocks=griddim_1D boundary_kernel(idata,odata,coef_D)
end

struct coef_GPU{T}
    bhinv::T
    d::T
    bd::T
    BS::T
end

struct coef_GPU_v2{T}
    bhinv::T
    d::T
    bd::T
    BS::T
    grid::T
end

struct coef_GPU_sbp_sat{T}
    bhinv::T
    d::T
    bd::T
    BS::T
    grid::T # Nx, Ny, hx, hy
    sat::T # tau_W, tau_E, tau_N, tau_S, beta
end

Adapt.@adapt_structure coef_GPU
Adapt.@adapt_structure coef_GPU_v2
Adapt.Adapt.@adapt_structure coef_GPU_sbp_sat

coef_p4 = coef_GPU(CuArray([48/17 48/59 48/43 48/49]),
                CuArray([-1/12 4/3 -5/2 4/3 -1/12]),
                CuArray([ 2    -5       4     -1       0      0;
                    1    -2       1      0       0      0;
                    -4/43 59/43 -110/43  59/43   -4/43   0;
                    -1/49  0      59/49 -118/49  64/49  -4/49]),
                CuArray([11/6 -3 3/2 -1/3]))

coef_p2 = coef_GPU_v2(CuArray([2. 0]), # The 0 here is only to make them identical arrays
    CuArray([1. -2 1]),
    CuArray([1. -2 1]),
    CuArray([3/2 -2 1/2]),
    CuArray([9 9 1/8 1/8]))   

coef_p2_D = cudaconvert(coef_p2)

coef_p2 = coef_GPU_sbp_sat(CuArray([2. 0]), # The 0 here is only to make them identical arrays
    CuArray([1. -2 1]),
    CuArray([1. -2 1]),
    CuArray([3/2 -2 1/2]),
    CuArray([9 9 1/8 1/8]),
    CuArray([13/(1/8) 13/(1/8) 1 1 -1]))  
coef_p2_D = cudaconvert(coef_p2)

# boundary_1 = CuArray(randn(10,3))
# boundary_1_out = CuArray(zeros(10,3))

# boundary(boundary_1,view(boundary_1_out,:,1);orientation=1)