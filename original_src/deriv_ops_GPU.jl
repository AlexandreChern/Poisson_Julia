using CUDA
using Printf
using StaticArrays
using GPUifyLoops: @unroll

include("deriv_ops.jl")



function D2x_GPU(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 1 # For second order derivative

    tile = @cuStaticSharedMem(eltype(d_u), (TILE_DIM1, TILE_DIM2, + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		@inbounds tile[k,l+HALO_WIDTH] = d_u[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		@inbounds tile[k,l] = d_u[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && i <= Ny && j <= Nx - HALO_WIDTH
		@inbounds tile[k,l+2*HALO_WIDTH] = d_u[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == 1
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	end

	if k <= TILE_DIM1 &&  l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && 2 <= j <= Nx-1
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
	end

	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
	end

    sync_threads()
    
    nothing
end


function D2y_GPU(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 1
	tile = @cuStaticSharedMem(eltype(d_u),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		@inbounds tile[k+HALO_WIDTH,l] = d_u[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH <= i <= Ny && j <= Nx
		@inbounds tile[k,l] = d_u[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		@inbounds tile[k+2*HALO_WIDTH,l] = d_u[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
		@inbounds d_y[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		@inbounds d_y[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
	end

	sync_threads()

	#Lower Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		@inbounds d_y[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
    end
    
    sync_threads()

    nothing

end


function tester_function(f,Nx,TILE_DIM_1,TILE_DIM_2)
    Ny = Nx
    @show f
    @eval gpu_function = $(Symbol(f,"_GPU"))
    @show gpu_function
    h = 1/Nx
	# TILE_DIM_1 = 16
	# TILE_DIM_2 = 2

	u = randn(Nx*Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	# d_y2 = similar(d_u)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	TILE_DIM = 32
	THREAD_NUM = 32
    BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1
    
    y = f(u,Nx,Ny,h)
    @cuda threads=blockdim blocks=griddim gpu_function(d_u, d_y, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
    @show y ≈ Array(d_y)
end


tester_function(D2x,100,4,4)


# Nx = 10
# Ny = Nx
# h = 1/Nx
# TILE_DIM_1 = 4
# TILE_DIM_2 = 16

# u = randn(Nx*Ny);
# d_u = CuArray(u);
# d_y = similar(d_u);
# d_y2 = similar(d_u);

# y = D2y(u,Nx,Ny,h)

# griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
# blockdim = (TILE_DIM_1,TILE_DIM_2)

# # TILE_DIM = 32
# # THREAD_NUM = 32
# # BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

# @cuda threads=blockdim blocks=griddim D2y_GPU_v7(d_u, d_y, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
# Array(d_y)

# Array(d_y) ≈ y

# D2y(u,Nx,Ny,h)