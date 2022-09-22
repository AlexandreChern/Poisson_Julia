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


function copy_GPU(idata,odata,Nx,Ny)
    # Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d copy_kernel(idata,odata,Nx,Ny)
end


function laplacian_GPU(idata,odata,Nx,Ny,hx,hy)
    # Nx,Ny = size(idata)
    # h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d laplacian_kernel(idata,odata,Nx,Ny,hx,hy)
end

function laplacian_kernel(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy

    Ny = size(idata)[2]

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


function laplacian_GPU_v2(idata,odata,Nx,Ny,hx,hy)
    # Nx,Ny = size(idata)
    # h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d laplacian_kernel_v2(idata,odata,Nx,Ny,hx,hy)
end

function laplacian_kernel_v2(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    tidy = threadIdx().y
    # Nx = coef_D.grid[1]
    # # Ny = coef_D.grid[2]
    # Ny = size(odata)[2]
    # hx = coef_D.grid[3]
    # hy = coef_D.grid[4]
    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy
    Ny = size(idata)[2]


    if 1 <= i <= Nx && 1 <= j <= Ny
        if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
            @inbounds odata[i,j] = -(idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) # - representing -D2
        else
            # odata[i,j] += 1
            @inbounds odata[i,j] = 0
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

function boundary_kernel_1_1(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_W = 13/hx
    beta = -1
    if 2 <= idx <= Nx - 1
        @inbounds odata[idx,1] = (-(idata[idx-1,1] - 2*idata[idx,1] + idata[idx+1,1] + idata[idx,1] - 2*idata[idx,2] + idata[idx,3]) + 2 * beta * (1.5 * idata[idx,1]) + 2 * tau_W * idata[idx,1] * hx) / 2
        # (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        @inbounds odata[idx,2] = (2 * beta * (-1 * idata[idx,1]))
        @inbounds odata[idx,3] = (0.5 * beta * idata[idx,1])
    end
    sync_threads()
    if idx == 1 || idx == Nx
        @inbounds odata[idx,1] =  (2 * beta * (1.5 * idata[idx,1]) + 2 * tau_W * (idata[idx,1]) * hx )/ 4
        @inbounds odata[idx,2] = (2 * beta * (-1 * idata[idx,1])) / 2
        @inbounds odata[idx,3] = (0.5 * beta * idata[idx,1]) / 2
    end
    return nothing
end


function boundary_kernel_1_2(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_S = 1
    Ny = size(idata)[2]
    tau_W = 13/hx
    beta = -1
    if 4 <= idx <= Ny - 1
        @inbounds odata[1,idx] = (-(idata[1,idx] - 2*idata[2,idx] + idata[3,idx] + idata[1,idx-1] - 2*idata[1,idx] + idata[1,idx+1]) + 2 * tau_S * (1.5 * idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) / 2
    end
    if idx == 1
        @inbounds odata[1,idx] = (-(idata[1,idx] - 2*idata[1,idx+1] + idata[1,idx+2] + idata[1,idx] - 2*idata[2,idx] + idata[3,idx])
                        + 2 * tau_S * (( 1.5* idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) ) / 4 
                        # + 2 * beta * (1.5 * idata[1,idx]) + 2 * tau_W * (idata[1,idx])) * h / 4 

                       
        @inbounds odata[1,idx+1] = (-(idata[1,idx+1] - 2*idata[2,idx+1] + idata[3,idx+1] + idata[1,idx] - 2*idata[1,idx+1] + idata[1,idx+2]) + 2 * tau_S * (1.5 * idata[1,idx+1] - 2*idata[2,idx+1] + 0.5*idata[3,idx+1]))/2
                        # + 2 * beta * (-1 * idata[1,idx])) / 2  # Dirichlet

        @inbounds odata[1,idx+2] = (-(idata[1,idx+2] - 2*idata[2,idx+2] + idata[3,idx+2] + idata[1,idx+1] - 2*idata[1,idx+2] + idata[1,idx+3]) + 2 * tau_S * (1.5 * idata[1,idx+2] - 2*idata[2,idx+2] + 0.5*idata[3,idx+2]))/2
                        # + 0.5 * beta * (idata[1,idx])) / 2# Dirichlet

    end
    return nothing
end

function boundary_kernel_1_3(idata,odata,Nx,Ny,hx,hy)
    return nothing
end

function boundary_kernel_1_4(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_N = 1
    Ny = size(idata)[2]
    beta = -1
    if 4 <= idx <= Ny - 1
        @inbounds odata[end,idx] = (-(idata[end,idx] - 2*idata[end-1,idx] + idata[end-2,idx] + idata[end,idx-1] - 2*idata[end,idx] + idata[end,idx+1]) + 2 * tau_N * (1.5 * idata[end,idx] - 2*idata[end-1,idx] + 0.5*idata[end-2,idx])) / 2
    end

    if idx == 1
        @inbounds odata[end,idx] = (-(idata[end,idx] - 2*idata[end,idx+1] + idata[end,idx+2] + idata[end,idx] - 2*idata[end-1,idx] + idata[end-2,idx])
        + 2 * tau_N * (( 1.5* idata[end,idx] - 2*idata[end-1,idx] + 0.5*idata[end-2,idx]))  ) / 4 
        # + 2 * beta * (1.5 * idata[3,idx]) + 2 * alpha1 * (idata[3,idx]) * h
       
        @inbounds odata[end,idx+1] = (-(idata[end,idx+1] - 2*idata[end-1,idx+1] + idata[end-2,idx+1] + idata[end,idx] - 2*idata[end,idx+1] + idata[end,idx+2]) + 2 * tau_N * (1.5 * idata[end,idx+1] - 2*idata[end-1,idx+1] + 0.5*idata[end-2,idx+1])) / 2  # Dirichlet
        #   (2 * beta * (-1 * idata[3,idx])) / 2  
        @inbounds odata[end,idx+2] = (-(idata[end,idx+2] - 2*idata[end-1,idx+2] + idata[end-2,idx+2] + idata[end,idx+1] - 2*idata[end,idx+2] + idata[end,idx+3]) + 2 * tau_N * (1.5 * idata[end,idx+2] - 2*idata[end-1,idx+2] + 0.5*idata[end-2,idx+2])) / 2# Dirichlet
        #  (0.5 * beta * (idata[3,idx])) / 2 +
    end

    return nothing
end

function boundary_kernel_2_1(idata,odata,Nx,Ny,hx,hy)
    # This function should be empty
    return nothing
end

function boundary_kernel_2_2(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_S = 1
    Ny = size(idata)[2]
    tau_W = 13/hx
    beta = -1
    if 2 <= idx <= Ny - 1
        @inbounds odata[1,idx] = (-(idata[1,idx] - 2*idata[2,idx] + idata[3,idx] + idata[1,idx-1] - 2*idata[1,idx] + idata[1,idx+1]) + 2 * tau_S * (1.5 * idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) / 2
    end
    return nothing
end

function boundary_kernel_2_3(idata,odata,Nx,Ny,hx,hy)
    # This function should be empty
    return nothing
end

function boundary_kernel_2_4(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    # Ny = coef_D.grid[2]
    tau_N = 1
    Ny = size(idata)[2]
    tau_W = 13/hx
    beta = -1
    if 2 <= idx <= Ny - 1
        @inbounds odata[end,idx] = (-(idata[end,idx] - 2*idata[end-1,idx] + idata[end-2,idx] + idata[end,idx-1] - 2*idata[end,idx] + idata[end,idx+1]) + 2 * tau_N * (1.5 * idata[end,idx] - 2*idata[end-1,idx] + 0.5*idata[end-2,idx])) / 2
    end
    return nothing
end

function boundary_kernel_3_1(idata,odata,Nx,Ny,hx,hy)
    return nothing
end

function boundary_kernel_3_2(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_S = 1
    Ny = size(idata)[2]
    tau_W = 13/hx
    beta = -1
    if 2 <= idx <= Ny - 3
        @inbounds odata[1,idx] = (-(idata[1,idx] - 2*idata[2,idx] + idata[3,idx] + idata[1,idx-1] - 2*idata[1,idx] + idata[1,idx+1]) 
                    + 2 * tau_S * (1.5 * idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) / 2
    end
    # if idx == 1
    #     odata[1,idx] = (-(idata[1,idx] - 2*idata[1,idx+1] + idata[1,idx+2] + idata[1,idx] - 2*idata[2,idx] + idata[3,idx])
    #                     + 2 * tau_S * (( 1.5* idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) ) / 4 
    #                     # + 2 * beta * (1.5 * idata[1,idx]) + 2 * tau_W * (idata[1,idx])) * h / 4 

                       
    #     odata[1,idx+1] = (-(idata[1,idx+1] - 2*idata[2,idx+1] + idata[3,idx+1] + idata[1,idx] - 2*idata[1,idx+1] + idata[1,idx+2]) + 2 * tau_S * (1.5 * idata[1,idx+1] - 2*idata[2,idx+1] + 0.5*idata[3,idx+1]))/2
    #                     # + 2 * beta * (-1 * idata[1,idx])) / 2  # Dirichlet

    #     odata[1,idx+2] = (-(idata[1,idx+2] - 2*idata[2,idx+2] + idata[3,idx+2] + idata[1,idx+1] - 2*idata[1,idx+2] + idata[1,idx+3]) + 2 * tau_S * (1.5 * idata[1,idx+2] - 2*idata[2,idx+2] + 0.5*idata[3,idx+2]))/2
    #                     # + 0.5 * beta * (idata[1,idx])) / 2# Dirichlet

    # end
    # sync_threads()
    if idx == Ny
        @inbounds odata[1,idx] = (-(idata[1,idx] - 2*idata[1,idx-1] + idata[1,idx-2] + idata[1,idx] - 2*idata[2,idx] + idata[3,idx])
                        + 2 * tau_S * (( 1.5* idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx]))  ) / 4 
                        # + 2 * beta * (1.5 * idata[3,idx]) + 2 * alpha2 * (idata[3,idx]) * h
                       
        @inbounds odata[1,idx-1] = (-(idata[1,idx-1] - 2*idata[2,idx-1] + idata[3,idx-1] + idata[1,idx-2] - 2*idata[1,idx-1] + idata[1,idx]) 
                        + 2 * tau_S * (1.5 * idata[1,idx-1] - 2*idata[2,idx-1] + 0.5*idata[3,idx-1])) / 2 # Dirichlet
                        # (2 * beta * (-1 * idata[3,idx])) / 2 +
        @inbounds odata[1,idx-2] =  (-(idata[1,idx-2] - 2*idata[2,idx-2] + idata[3,idx-2] + idata[1,idx-3] - 2*idata[1,idx-2] + idata[1,idx-1]) 
                        + 2 * tau_S * (1.5 * idata[1,idx-2] - 2*idata[2,idx-2] + 0.5*idata[3,idx-2])) / 2# Dirichlet
                        # (0.5 * beta * (idata[3,idx])) / 2 +
    end
    return nothing
end


function boundary_kernel_3_3(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_E = 13/hx
    beta = -1
    if 2 <= idx <= Nx - 1
        @inbounds odata[idx,end] = (-(idata[idx-1,end] - 2*idata[idx,end] + idata[idx+1,end] + idata[idx,end] - 2*idata[idx,end-1] + idata[idx,end-2]) + 2 * beta * (1.5 * idata[idx,end]) + 2 * tau_E * idata[idx,end] * hx) / 2
        # (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        @inbounds odata[idx,end-1] = (2 * beta * (-1 * idata[idx,end]))
        @inbounds odata[idx,end-2] = (0.5 * beta * idata[idx,end])
    end
    sync_threads()
    if idx == 1 || idx == Nx
        @inbounds odata[idx,end] =  (2 * beta * (1.5 * idata[idx,end]) + 2 * tau_E * (idata[idx,end]) * hx )/ 4
        @inbounds odata[idx,end-1] = (2 * beta * (-1 * idata[idx,end])) / 2
        @inbounds odata[idx,end-2] = (0.5 * beta * idata[idx,end]) / 2
    end
    return nothing
end

function boundary_kernel_3_4(idata,odata,Nx,Ny,hx,hy)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    tau_N = 1
    Ny = size(idata)[2]
    beta = -1

    if 2 <= idx <= Ny - 3
        @inbounds  odata[end,idx] = (-(idata[end,idx] - 2*idata[end-1,idx] + idata[end-2,idx] + idata[end,idx-1] - 2*idata[end,idx] + idata[end,idx+1]) 
                    + 2 * tau_N * (1.5 * idata[end,idx] - 2*idata[end-1,idx] + 0.5*idata[end-2,idx])) / 2
    end

    if idx == Ny
        @inbounds odata[end,idx] = (-(idata[end,idx] - 2*idata[end,idx-1] + idata[end,idx-2] + idata[end,idx] - 2*idata[end-1,idx] + idata[end-2,idx])
        + 2 * tau_N * (( 1.5* idata[end,idx] - 2*idata[end-1,idx] + 0.5*idata[end-2,idx]))  ) / 4 
        # + 2 * beta * (1.5 * idata[3,idx]) + 2 * alpha1 * (idata[3,idx]) * h
       
        @inbounds odata[end,idx-1] = (-(idata[end,idx-1] - 2*idata[end-1,idx-1] + idata[end-2,idx-1] + idata[end,idx-2] - 2*idata[end,idx-1] + idata[end,idx]) 
                        + 2 * tau_N * (1.5 * idata[end,idx-1] - 2*idata[end-1,idx-1] + 0.5*idata[end-2,idx-1])) / 2  # Dirichlet
        #   (2 * beta * (-1 * idata[3,idx])) / 2  
        @inbounds odata[end,idx-2] = (-(idata[end,idx-2] - 2*idata[end-1,idx-2] + idata[end-2,idx-2] + idata[end,idx-3] - 2*idata[end,idx-2] + idata[end,idx-1]) 
                        + 2 * tau_N * (1.5 * idata[end,idx-2] - 2*idata[end-1,idx-2] + 0.5*idata[end-2,idx-2])) / 2# Dirichlet
        #  (0.5 * beta * (idata[3,idx])) / 2 +
    end
    return nothing
end

boundary_kernels = [boundary_kernel_1_1 boundary_kernel_1_2 boundary_kernel_1_3 boundary_kernel_1_4;
boundary_kernel_2_1 boundary_kernel_2_2 boundary_kernel_2_3 boundary_kernel_2_4;
boundary_kernel_3_1 boundary_kernel_3_2 boundary_kernel_3_3 boundary_kernel_3_4
]


function boundary(idata,odata,Nx,Ny,hx,hy;orientation=1,type=1)
    TILE_DIM_1D = 16
    blockdim_1D = TILE_DIM_1D
    griddim_1D = div(Nx,TILE_DIM_1D) + 1
    boundary_kernel = boundary_kernels[type,orientation]
    @cuda threads=blockdim_1D blocks=griddim_1D boundary_kernel(idata,odata,Nx,Ny,hx,hy)
end

# struct coef_GPU{T} where {T<:Number}
#     bhinv::T
#     d::T
#     bd::T
#     BS::T
# end

# struct coef_GPU_v2{T} where {T<:Number}
#     bhinv::T
#     d::T
#     bd::T
#     BS::T
#     grid::T
# end

# T<:CuDeviceMatrix

# struct coef_GPU_sbp_sat{T}
# # struct coef_GPU_sbp_sat{T} where {T<:CuDeviceMatrix}
#     bhinv::T
#     d::T
#     bd::T
#     BS::T
#     grid::T # Nx, Ny, hx, hy
#     sat::T # tau_W, tau_E, tau_S, tau_N, beta
# end

# # coef_GPU_sbp_sat(bhinv::T) where T = coef_GPU_sbp_sat{T}(bhinv)
# # Adapt.@adapt_structure coef_GPU
# # Adapt.@adapt_structure coef_GPU_v2
# Adapt.@adapt_structure coef_GPU_sbp_sat


struct coef_GPU_sbp_sat_v4{CuArray}
    # struct coef_GPU_sbp_sat{T} where {T<:CuDeviceMatrix}
    bhinv::CuArray
    d::CuArray
    bd::CuArray
    BS::CuArray
    grid::CuArray # Nx, Ny, hx, hy
    sat::CuArray # tau_W, tau_E, tau_S, tau_N, beta
end
Adapt.@adapt_structure coef_GPU_sbp_sat_v4

coef_eg = coef_GPU_sbp_sat_v4(CuArray([2. 0]), # The 0 here is only to make them identical arrays
    CuArray([1. -2 1]),
    CuArray([1. -2 1]),
    CuArray([3/2 -2 1/2]),
    CuArray([5 5 1/(5-1) 1/(5-1)]),
    CuArray([13/(1/(5-1)) 13/(1/(5-1)) 1 1 -1]))  
coef_eg_D = cudaconvert(coef_eg)
isbits(coef_eg_D)
# coef_p4 = coef_GPU(CuArray([48/17 48/59 48/43 48/49]),
#                 CuArray([-1/12 4/3 -5/2 4/3 -1/12]),
#                 CuArray([ 2    -5       4     -1       0      0;
#                     1    -2       1      0       0      0;
#                     -4/43 59/43 -110/43  59/43   -4/43   0;
#                     -1/49  0      59/49 -118/49  64/49  -4/49]),
#                 CuArray([11/6 -3 3/2 -1/3]))

# coef_p2 = coef_GPU_v2(CuArray([2. 0]), # The 0 here is only to make them identical arrays
#     CuArray([1. -2 1]),
#     CuArray([1. -2 1]),
#     CuArray([3/2 -2 1/2]),
#     CuArray([9 9 1/8 1/8]))   

# coef_p2_D = cudaconvert(coef_p2)

# coef_p2 = coef_GPU_sbp_sat(CuArray([2. 0]), # The 0 here is only to make them identical arrays
#     CuArray([1. -2 1]),
#     CuArray([1. -2 1]),
#     CuArray([3/2 -2 1/2]),
#     CuArray([9 9 1/8 1/8]),
#     CuArray([13/(1/8) 13/(1/8) 1 1 -1]))  
# coef_p2_D = cudaconvert(coef_p2)

# boundary_1 = CuArray(randn(10,3))
# boundary_1_out = CuArray(zeros(10,3))

# boundary(boundary_1,view(boundary_1_out,:,1);orientation=1)


# function allocate_boundaries_GPU(type,Nx,)
#     if type == 1
#         boundaries_GPUs = [CuArray()]
#     end
# end


function allocate_GPU_arrays(idata_GPU;num_blocks=length(devices()))
    # num_blocks = length(devices())
    # num_blocks = 3
    sub_block_width = div(Ny,num_blocks)

    y_indeces = 1:sub_block_width:Ny

    # odata_GPUs = Array(undef,0,num_blocks)

    # odata_GPUs = [CuArray(zeros(Nx,sub_block_width)) for i in 1:num_blocks]
    # odata_GPUs[end] = CuArray(zeros(Nx,Ny-sub_block_width*(num_blocks-1)))

    # Setting up idata_GPUs
    idata_GPUs = [CuArray(zeros(Nx,sub_block_width)) for i in 1:num_blocks]
    idata_GPUs[1] = CuArray(zeros(Nx,size(idata_GPUs[1])[2]+1))
    idata_GPUs[end] = CuArray(zeros(Nx,size(idata_GPUs[end])[2]+1))
    if length(idata_GPUs) >= 3
        for i = 2:num_blocks-1
            idata_GPUs[i] = CuArray(zeros(Nx,size(idata_GPUs[i])[2]+2))
        end
    end

    copyto!(idata_GPUs[1],idata_GPU[:,1:y_indeces[2]])
    copyto!(idata_GPUs[end],idata_GPU[:,end-size(idata_GPUs[end])[2]+1:end])

    if length(idata_GPUs) >= 3
        for i = 2:num_blocks-1
            copyto!(idata_GPUs[i],idata_GPU[:,y_indeces[i]-1:y_indeces[i+1]])
        end
    end

    # Setting up odata_GPUs
    odata_GPUs = [CuArray(zeros(Nx,sub_block_width)) for i in 1:num_blocks]
    odata_GPUs[1] = CuArray(zeros(Nx,size(odata_GPUs[1])[2]+1))
    odata_GPUs[end] = CuArray(zeros(Nx,size(odata_GPUs[end])[2]+1))
    if length(idata_GPUs) >= 3
        for i = 2:num_blocks-1
            @inbounds odata_GPUs[i] = CuArray(zeros(Nx,size(odata_GPUs[i])[2]+2))
        end
    end


    odata_boundaries_GPUs = [CuArray(zeros(Nx,3)), CuArray(zeros(1,size(odata_GPUs[1])[2])),CuArray(zeros(1,size(odata_GPUs[1])[2]))]
    for i in 2:num_blocks-1
        append!(odata_boundaries_GPUs,[CuArray(zeros(1,size(odata_GPUs[i])[2])), CuArray(zeros(1,size(odata_GPUs[i])[2]))])
    end
    append!(odata_boundaries_GPUs,[CuArray(zeros(1,size(odata_GPUs[end])[2])), CuArray(zeros(Nx,3)), CuArray(zeros(1,size(odata_GPUs[end])[2]))])

    return idata_GPUs, odata_GPUs, odata_boundaries_GPUs, num_blocks
end


function find_boundaries_GPUs(orientation,block_idx,num_blocks)
    idx = 0
    if block_idx == 1
        if orientation == 1
            idx = 1
        elseif orientation == 2
            idx = 2
        elseif orientation == 4
            idx = 3
        end
    elseif 2 <= block_idx <= num_blocks-1
        if orientation == 2
            idx = (block_idx - 2) * 2 + 3 + 1
        elseif orientation == 4
            idx = (block_idx - 2) * 2 + 3 + 2
        end
    elseif block_idx == num_blocks
        if orientation == 2
            idx = block_idx * 2
        elseif orientation == 3
            idx = block_idx * 2 + 1
        elseif orientation == 4
            idx = block_idx * 2 + 2
        end
    end
    return idx
end

# idata_GPUs, odata_GPUs, odata_boundaries_GPUs, num_blocks = allocate_GPU_arrays(idata_GPU,num_blocks=3)