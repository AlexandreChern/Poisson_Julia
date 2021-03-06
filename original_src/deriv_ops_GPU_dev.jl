#functions for calling the matrix vector products
#i.e. function A(u,Nx,Ny,h) computes the product A*u and stores it in y

#Right now only for  the second-order accurate SBP operators for constant coefficients from
#Mattsson and Nordstrom, 2004.

#Written for the 2D domain (x,y) \in (a, b) \times (c, d),
#stacking the grid function in the vertical direction,
#so that, e.g., \partial u / \partial x \approx D \kron I and
#\partial u / \partial y \approx I \kron D
#the faces of the 2D domain are label bottom to top, left to right, i.e.
#side 1 is y = c, side 2 is y = d
#side 3 is x = a, side 4 is x = b

#D2x and D2y compute approximations to
#\partial^2 u / \partial x^2 and
#\partial^2 u / \partial y^2, respectively

#FACEtoVOL(face,u_face,Nx,Ny) maps the face value u_face to a full length vector at the
#nodes corresponding to face

#BxSx \approx the traction on faces 3 and 4
#BySy \approx the traction on faces 1 and 2

#BxSx_tran and BySy_tran are the transposes of BxSx and BySy


# include("deriv_ops.jl")

using DataFrames
using CUDA
using Printf
using StaticArrays
using GPUifyLoops: @unroll

function D2x(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)
	idx = 1:Ny
	@inbounds y[idx] = (u[idx] - 2 .* u[Ny .+ idx] + u[2*Ny .+ idx]) ./ h^2

	idx1 = Ny+1:N-Ny
	@inbounds y[idx1] = (u[idx1 .- Ny] - 2 .* u[idx1] + u[idx1 .+ Ny]) ./ h^2

	idx2 = N-Ny+1:N
	@inbounds y[idx2] = (u[idx2 .- 2*Ny] -2 .* u[idx2 .- Ny] + u[idx2]) ./ h^2

	return y
end

function D2x_GPU(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	# d_y = zeros(N)
	if tidx <= Ny
		d_y[tidx] = (d_u[tidx] - 2 * d_u[Ny + tidx] + d_u[2*Ny + tidx]) / h^2
	end
	sync_threads()

	if Ny+1 <= tidx <= N-Ny
		d_y[tidx] = (d_u[tidx - Ny] - 2 .* d_u[tidx] + d_u[tidx + Ny]) / h^2
	end

	sync_threads()

	if N-Ny+1 <= tidx <= N
		d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	end
	sync_threads()

	nothing
end

function D2x_GPU_v2(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	# d_y = zeros(N)
	if tidx <= Ny
		@inbounds d_y[tidx] = (d_u[tidx] - 2 * d_u[Ny + tidx] + d_u[2*Ny + tidx]) / h^2
		# (d_u[tidx] - 2 * d_u[Ny + tidx] + d_u[2*Ny + tidx]) / h^2
	end

	if Ny+1 <= tidx <= N-Ny
		@inbounds d_y[tidx] = (d_u[tidx - Ny] - 2 .* d_u[tidx] + d_u[tidx + Ny]) / h^2
		# (d_u[tidx - Ny] - 2 .* d_u[tidx] + d_u[tidx + Ny]) / h^2
	end


	if N-Ny+1 <= tidx <= N
		@inbounds d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
		# (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	end

	sync_threads()

	nothing
end

function D2x_GPU_v3(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	# d_y = zeros(N)
	if tidx <= Ny
		d_y[tidx] = (d_u[tidx] - 2 * d_u[Ny + tidx] + d_u[2*Ny + tidx]) / h^2
	end

	if Ny+1 <= tidx <= N-Ny
		d_y[tidx] = (d_u[tidx - Ny] - 2 * d_u[tidx] + d_u[tidx + Ny]) / h^2
	end


	if N-Ny+1 <= tidx <= N
		d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	end

	sync_threads()

	nothing
end

# function D2x_GPU_v4(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}, ::Val{BLOCK_ROWS1}, ::Val{BLOCK_ROWS2}) where {TILE_DIM1, TILE_DIM2, BLOCK_ROWS1, BLOCK_ROWS2}
function D2x_GPU_v4(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
	tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Ny

	# i = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	tile = @cuStaticSharedMem(eltype(d_u),(TILE_DIM1,TILE_DIM2+4))

	k = tidx
	l = tidy

	# @unroll for k = 0:BLOCK_ROWS1:TILE_DIM1-1
	# 	for l = 0:BLOCK_ROWS2:TILE_DIM2-1
	# 		@inbounds tile[tidx, ] # unroll function not complete
	# 	end
	# end
	# for k = 1:TILE_DIM1
	# 	for l = 1:TILE_DIM2
	# 		if global_index <= Nx*Ny
	# 			tile[k,l] = d_u[global_index]
	# 		end
	# 	end
	# end

	# if k <= TILE_DIM1 && l <= TILE_DIM2+4 && global_index <= Nx*Ny
	# 	tile[k,l] = d_u[global_index]
	# end

	if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
		tile[k,l+2] = d_u[global_index]
	end

	sync_threads()


	# Periodically fill in overlapping region within shared memory
	if k <= TILE_DIM1 && l <= 2 && 2*Ny+1 <= global_index <= (Nx+2)*Ny
		# tile[k,l] = tile[k,l+TILE_DIM2+2]
		tile[k,l] = d_u[global_index - 2*Ny]
	end

	sync_threads()

	if k <= TILE_DIM1 && l >= TILE_DIM2 - 2 && 2*Ny+1 <= global_index <= (Nx-2)*Ny
		tile[k,l+4] = d_u[global_index + 2*Ny]
	end

	sync_threads()


	# for k = 1:TILE_DIM1
	# 	for l = 1:TILE_DIM2

	# 		if global_index <= Nx*Ny
	# 			d_y[global_index] = (tile[k,l] - 2*tile[k,l+1])
	# 		end
	# 	end
	# end

	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	# 	d_y[global_index] = tile[k,l]
	# end

	if k <= TILE_DIM1 && l + 2 <= TILE_DIM2 + 4 && global_index <= Ny
		d_y[global_index] = (tile[k,l + 2] - 2*tile[k,l+3] + tile[k,l+4]) / h^2
	end

	if k <= TILE_DIM1 &&  l + 2 <= TILE_DIM2 + 4 && Ny+1 <= global_index <= (Nx-1)*Ny
		d_y[global_index] = (tile[k,l + 1] - 2*tile[k, l + 2] + tile[k,l+3]) / h^2
	end

	if k <= TILE_DIM1 && l + 2 <= TILE_DIM2 + 4 && (Nx-1)*Ny + 1 <= global_index <= Nx*Ny
		d_y[global_index] = (tile[k,l] - 2*tile[k,l + 1] + tile[k,l+2]) / h^2
	end

	sync_threads()


	# N = Nx*Ny
	# # d_y = zeros(N)
	# if tidx <= Ny
	# 	d_y[tidx] = (d_u[tidx] - 2 * d_u[Ny + tidx] + d_u[2*Ny + tidx]) / h^2
	# end
	#
	# if Ny+1 <= tidx <= N-Ny
	# 	d_y[tidx] = (d_u[tidx - Ny] - 2 * d_u[tidx] + d_u[tidx + Ny]) / h^2
	# end
	#
	#
	# if N-Ny+1 <= tidx <= N
	# 	d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	# end
	#
	# sync_threads()

	nothing
end



function D2x_GPU_v5(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
	tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Ny

	# i = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	tile = @cuStaticSharedMem(eltype(d_u),(TILE_DIM1,TILE_DIM2+4))

	k = tidx
	l = tidy

	# Writing pencil-shaped shared memory

	# for tile itself
	if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
		tile[k,l+2] = d_u[global_index]
	end

	sync_threads()

	# for left halo
	if k <= TILE_DIM1 && l <= 2 && 2*Ny+1 <= global_index <= (Nx+2)*Ny
		tile[k,l] = d_u[global_index - 2*Ny]
	end

	sync_threads()


	# for right halo
	if k <= TILE_DIM1 && l >= TILE_DIM2 - 2 && 2*Ny+1 <= global_index <= (Nx-2)*Ny
		tile[k,l+4] = d_u[global_index + 2*Ny]
	end

	sync_threads()

	# Finite difference operation starts here

	if k <= TILE_DIM1 && l + 2 <= TILE_DIM2 + 4 && global_index <= Ny
		d_y[global_index] = (tile[k,l + 2] - 2*tile[k,l+3] + tile[k,l+4]) / h^2
		# (tile[k,l + 2] - 2*tile[k,l+3] + tile[k,l+4]) / h^2
	end

	if k <= TILE_DIM1 &&  l + 2 <= TILE_DIM2 + 4 && Ny+1 <= global_index <= (Nx-1)*Ny
		d_y[global_index] = (tile[k,l + 1] - 2*tile[k, l + 2] + tile[k,l+3]) / h^2
		# (tile[k,l + 1] - 2*tile[k, l + 2] + tile[k,l+3]) / h^2
	end

	if k <= TILE_DIM1 && l + 2 <= TILE_DIM2 + 4 && (Nx-1)*Ny + 1 <= global_index <= Nx*Ny
		d_y[global_index] = (tile[k,l] - 2*tile[k,l + 1] + tile[k,l+2]) / h^2
		# (tile[k,l] - 2*tile[k,l + 1] + tile[k,l+2]) / h^2
	end

	sync_threads()

	nothing
end


# Incorrect implementations. Trying to see if I can optimize writing to shared memory
# Define arbitrary halo width
function D2x_GPU_v6(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
	tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Ny

	HALO_WIDTH = 1
	# i = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	tile = @cuStaticSharedMem(eltype(d_u),(TILE_DIM1,TILE_DIM2+2*HALO_WIDTH))

	k = tidx
	l = tidy

	# Writing pencil-shaped shared memory

	# for tile itself
	if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
		@inbounds tile[k,l+HALO_WIDTH] = d_u[global_index]
	end

	# @unroll for m = 0:TILE_DIM2-1
	# 	if k <= TILE_DIM1 && global_index <= Nx*Ny
	# 		@inbounds tile[tidx, tidy+HALO_WIDTH+m] = d_u[global_index + m*Ny]
	# 	end
	# end


	sync_threads()

	# for left halo
	if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
		@inbounds tile[k,l] = d_u[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
		@inbounds tile[k,l+2*HALO_WIDTH] = d_u[global_index + HALO_WIDTH*Ny]
	end

	# sync_threads()

	# Finite difference operation starts here

	# Using global_index is wrong here. Because i,j could be larger than Nx, and Ny, which makes accessing through global_index is not unique.
	# It can cause memory issue.
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && global_index <= Ny
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	end

	if k <= TILE_DIM1 &&  l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && Ny+1 <= global_index <= (Nx-1)*Ny
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
	end

	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && (Nx-1)*Ny + 1 <= global_index <= Nx*Ny
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
	end

	sync_threads()

	nothing
end

function D2x_GPU_v7(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
	tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Ny

	HALO_WIDTH = 1
	# i = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	tile = @cuStaticSharedMem(eltype(d_u),(TILE_DIM1,TILE_DIM2+2*HALO_WIDTH))

	k = tidx
	l = tidy

	# Writing pencil-shaped shared memory

	# for tile itself

	if k <= TILE_DIM1 && l <= TILE_DIM2 # && global_index <= Nx*Ny
		@inbounds tile[k,l+HALO_WIDTH] = d_u[global_index]
	end

	# UNROLL_NUM = 0

	# @unroll for m = 0:UNROLL_NUM
	# 	# if k <= TILE_DIM1 && l+m <= TILE_DIM2 && global_index <= Nx*Ny
	# 	if k + m <= TILE_DIM1 && l <= TILE_DIM2 # &&  global_index <= Nx*Ny
	# 		# @inbounds tile[tidx, tidy+HALO_WIDTH+m] = d_u[global_index + m*Nx]
	# 		@inbounds tile[tidx+m, tidy+HALO_WIDTH] = d_u[global_index + m]
	# 	end
	# end


	sync_threads()

	# for left halo
	if k <= TILE_DIM1 && l <= HALO_WIDTH  && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
		@inbounds tile[k,l] = d_u[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
		@inbounds tile[k,l+2*HALO_WIDTH] = d_u[global_index + HALO_WIDTH*Ny]
	end

	# sync_threads()

	# Finite difference operation starts here

	# if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && global_index <= Ny
	# 	@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	# end

	# if k <= TILE_DIM1 &&  l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && Ny+1 <= global_index <= (Nx-1)*Ny
	# 	@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
	# end

	# if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && (Nx-1)*Ny + 1 <= global_index <= Nx*Ny
	# 	@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
	# end

	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == 1
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	end

	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && 2 <= j <= Nx - 1
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
	end

	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		@inbounds d_y[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
	end

	sync_threads()

	nothing
end

function tester_D2x_v5(Nx)
	Ny = Nx
	h = 1/Nx
	TILE_DIM_1 = 4
	TILE_DIM_2 = 4

	d_u = CuArray(randn(Nx*Ny))
	d_y = similar(d_u)
	d_y2 = similar(d_u)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	TILE_DIM = 32
	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	@cuda threads=blockdim blocks=griddim D2x_GPU_v7(d_u,d_y,Nx,Ny,h,Val(TILE_DIM_1),Val(TILE_DIM_2))
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU_v2(d_u, d_y2, Nx, Ny, h, Val(TILE_DIM))
	@show Array(d_y) ≈ Array(d_y2)
	# @show Array((d_y - d_y2))
	return nothing
end



function tester_D2x(Nx)
	# Nx = Ny = 1000;
	Ny = Nx
	u = randn(Nx * Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	d_y5 = similar(d_u)
	h = 1/Nx
	TILE_DIM=32
	# TILE_DIM=128
	t1 = 0
	t2 = 0
	t3 = 0

	TILE_DIM_1 = 4
	TILE_DIM_2 = 16

	rep_times = 20

	THREAD_NUM = TILE_DIM
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	griddim = (div(Nx,TILE_DIM_1)+1,div(Ny,TILE_DIM_2)+1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	y = D2x(u,Nx,Ny,h)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	y_gpu = collect(d_y)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU_v2(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	y_gpu_2 = collect(d_y)
	@cuda threads=blockdim blocks=griddim D2x_GPU_v7(d_u,d_y5, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	y_gpu_5 = collect(d_y5)
	@show y ≈ y_gpu
	@show y ≈ y_gpu_2
	@show y ≈ y_gpu_5

	ty = time_ns()
	for i in 1:rep_times
		y = D2x(u,Nx,Ny,h)
	end
	ty_end = time_ns()
	t1 = ty_end - ty

	t_dy = time_ns()
	for i in 1:rep_times
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	end
	synchronize()
	# sync_threads()
	t_dy_end = time_ns()
	t2 = t_dy_end - t_dy

	t_dy_v2 = time_ns()
	for i in 1:rep_times
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU_v2(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	end
	synchronize()
	# sync_threads()
	t_dy_v2_end = time_ns()
	t3 = t_dy_v2_end - t_dy_v2

	t_dy_v5 = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim D2x_GPU_v7(d_u,d_y5, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	t_dy_v5_end = time_ns()
	t5 = t_dy_v5_end - t_dy_v5

	@show Float64(t1)
	@show Float64(t2)
	@show Float64(t3)
	@show Float64(t5)

	@show t1/t2
	@show t1/t3
	@show t1/t5

	memsize = length(u) * sizeof(eltype(u))
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)
	@printf("GPU Through-put %20.2f\n", 2 * memsize * rep_times / t2)
	@printf("GPU (v2) Through-put %20.2f\n", 2 * memsize * rep_times / t3)
	@printf("GPU (v5) Through-put %20.2f\n", 2 * memsize * rep_times / t5)

	return Float64(t1), Float64(t2), Float64(t3), Float64(t5)
end



function D2y(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)
	idx = 1:Ny:N-Ny+1
	y[idx] = (u[idx] - 2 .* u[idx .+ 1] + u[idx .+ 2]) ./ h^2

	idx1 = Ny:Ny:N
	y[idx1] = (u[idx1 .- 2] - 2 .* u[idx1 .- 1] + u[idx1]) ./ h^2

	for j = 1:Nx
		idx = 2+(j-1)*Ny:j*Ny-1
		y[idx] = (u[idx .- 1] - 2 .* u[idx] + u[idx .+ 1]) ./ h^2
	end

	return y

end

function D2y_GPU(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	# tidx = ((blockIdx().x - 1) * TILE_DIM + threadIdx().x - 1)*Ny + 1
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	# d_y = zeros(N)
	t1 = (tidx - 1)*Ny + 1
	if 1 <= t1 <= N - Ny + 1
		d_y[t1] = (d_u[t1] - 2 * d_u[t1 + 1] + d_u[t1 + 2]) / h^2
	end
	sync_threads()

	t2 = tidx * Ny
	if Ny <= t2 <= N
		d_y[t2] = (d_u[t2 - 2] - 2 * d_u[t2 - 1] + d_u[t2]) / h^2
	end

	sync_threads()

	for j = 1:Nx
		if 2 + (j-1)*Ny <= tidx <= j*Ny-1
			d_y[tidx] = (d_u[tidx - 1] - 2 * d_u[tidx] + d_u[tidx + 1]) / h^2
		end
	end
	sync_threads()

	# if N-Ny+1 <= tidx <= N
	# 	d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	# end

	nothing
end

function D2y_GPU_v2(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	# tidx = ((blockIdx().x - 1) * TILE_DIM + threadIdx().x - 1)*Ny + 1
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	# d_y = zeros(N)
	t1 = (tidx - 1)*Ny + 1
	if 1 <= t1 <= N - Ny + 1
		d_y[t1] = (d_u[t1] - 2 * d_u[t1 + 1] + d_u[t1 + 2]) / h^2
	end


	t2 = tidx * Ny
	if Ny <= t2 <= N
		d_y[t2] = (d_u[t2 - 2] - 2 * d_u[t2 - 1] + d_u[t2]) / h^2
	end

	for j = 1:Nx
		if 2 + (j-1)*Ny <= tidx <= j*Ny-1
			d_y[tidx] = (d_u[tidx - 1] - 2 * d_u[tidx] + d_u[tidx + 1]) / h^2
		end
	end

	# if N-Ny+1 <= tidx <= N
	# 	d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	# end

	nothing
end



function D2y_GPU_v3(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	if 1 <= tidx <= N && 1 <= tidx <= N
		d_y[tidx] = (d_u[tidx] - 2d_u[tidx+1] + d_u[tidx + 2]) / h^2
	elseif mod(tidx,Ny) == 0
		d_y[tidx] = (d_u[tidx] - 2d_u[tidx-1] + d_u[tidx - 2]) / h^2
	else
		d_y[tidx] = (d_u[tidx-1] - 2d_u[tidx] + d_u[tidx + 1]) / h^2
	end
	nothing

end


function D2y_GPU_v4(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	if 2 <= tidx <= N-1
		d_y[tidx] = (d_u[tidx-1] - 2d_u[tidx] + d_u[tidx + 1]) / h^2
	end
	sync_threads()

	# tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	# tb = tidx * Ny
	tb = tidx
	if 1 <= tb <= N && mod(tb,Ny) == 0
		d_y[tb] = (d_u[tb] - 2d_u[tb - 1] + d_u[tb - 2]) / h^2
	end
	# sync_threads()
	# if 1 <= tidx <= Ny
	# 	d_y[tidx*Ny] = (d_u[tidx*Ny] - 2d_u[tidx*Ny - 1] + d_u[tidx*Ny - 2]) / h^2
	# 	d_y[(tidx-1)*Ny+1] = (d_u[(tidx-1)*Ny+1] - 2d_u[(tidx-1)*Ny + 2] + d_u[(tidx-1)*Ny + 3]) / h^2
	# end
	sync_threads()

	# tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	# te = (tidx-1) * Ny + 1
	te = tidx
	if 1 <= te <= N && mod(te,Ny) == 1
		d_y[te] = (d_u[te] - 2d_u[te + 1] + d_u[te + 2]) / h^2
	end
	# if 1 <= tidx <= Ny
	# 	d_y[(tidx-1)*Ny+1] = (d_u[(tidx-1)*Ny+1] - 2d_u[(tidx-1)*Ny + 2] + d_u[(tidx-1)*Ny + 3]) / h^2
	# end
	sync_threads()
	nothing
end

function D2y_GPU_v5(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	if 2 <= tidx <= N-1
		d_y[tidx] = (d_u[tidx-1] - 2d_u[tidx] + d_u[tidx + 1]) / h^2
	end


	if 1 <= tidx <= N && mod(tidx,Ny) == 0
		d_y[tidx] = (d_u[tidx] - 2d_u[tidx - 1] + d_u[tidx - 2]) / h^2
		d_y[tidx-Ny+1] = (d_u[tidx-Ny+1] - 2d_u[tidx - Ny + 2] + d_u[tidx - Ny + 3]) / h^2
	end

	# if 1 <= tidx <= N && mod(tidx,Ny) == 1
	# 	d_y[tidx] = (d_u[tidx] - 2d_u[tidx + 1] + d_u[tidx + 2]) / h^2
	# end
	sync_threads()
	nothing
end

function D2y_GPU_v6(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	if 2 <= tidx <= N-1
		d_y[tidx] = (d_u[tidx-1] - 2d_u[tidx] + d_u[tidx + 1]) / h^2
	end


	if 1 <= tidx <= N && mod(tidx,Ny) == 0
		d_y[tidx] = (d_u[tidx] - 2d_u[tidx - 1] + d_u[tidx - 2]) / h^2
	end

	if 1 <= tidx <= N && mod(tidx,Ny) == 1
		d_y[tidx] = (d_u[tidx] - 2d_u[tidx + 1] + d_u[tidx + 2]) / h^2
	end
	sync_threads()
	nothing
end


function D2y_GPU_v7(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
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
	if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
		@inbounds tile[k+HALO_WIDTH,l] = d_u[global_index]
	end

	sync_threads()

	# For upper halo
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
		@inbounds tile[k,l] = d_u[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
		@inbounds tile[k+2*HALO_WIDTH,l] = d_u[global_index + HALO_WIDTH]
	end

	sync_threads()


	# Finite Difference Operations starts here

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
	
	# if global_index <= Nx*Ny
	# 	d_y[global_index] = tile[k+HALO_WIDTH,l]
	# end # Check if copying data to tile is correct, checked

	# UNROLL_NUM = TILE_DIM2 - 2

	# for m = 1:UNROLL_NUM
	# 	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l+m <= TILE_DIM2 && i == 1 && j <= Ny
	# 		d_y[global_index + m*Nx] = (tile[k+HALO_WIDTH,l+m] - 2*tile[k+HALO_WIDTH+1,l+m] + tile[k+HALO_WIDTH+2,l+m]) / h^2
	# 	end

	# 	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l+m <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
	# 		d_y[global_index + m*Nx] = (tile[k+HALO_WIDTH-1,l+m] - 2*tile[k+HALO_WIDTH,l+m] + tile[k+HALO_WIDTH+1,l+m]) / h^2
	# 	end

	# 	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l+m <= TILE_DIM2 && i == Nx && j <= Ny
	# 		d_y[global_index + m*Nx] = (tile[k+HALO_WIDTH-2,l+m] - 2*tile[k+HALO_WIDTH-1,l+m] + tile[k+HALO_WIDTH,l+m]) / h^2
	# 	end
	# end

	sync_threads()
	
	nothing

end


function tester_D2y_v7(Nx)
	Ny = Nx
	h = 1/Nx
	TILE_DIM_1 = 4
	TILE_DIM_2 = 16

	u = randn(Nx*Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	d_y2 = similar(d_u)
	y = D2y(u,Nx,Ny,h)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	TILE_DIM = 32
	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	@cuda threads=blockdim blocks=griddim D2y_GPU_v7(d_u,d_y,Nx,Ny,h,Val(TILE_DIM_1),Val(TILE_DIM_2))
	# @cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v2(d_u, d_y2, Nx, Ny, h, Val(TILE_DIM))
	# @show Array(d_y) ≈ Array(d_y2)
	@show Array(d_y) ≈ y
	@show y
	@show d_y
	# @show Array((d_y - d_y2))
	# @show Array((d_y - d_u))
	# @show Array(d_y)
	# @show y
	# @show Array(d_y - d_y2) 
	# @show Array(d_u)
	# @show Array(d_y) - y
	return nothing
end



function tester_D2y(Nx)
	# Nx = Ny = 1000;
	Ny = Nx
	u = randn(Nx * Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	d_y2 = similar(d_u)
	d_y7 = similar(d_u)
	h = 1/Nx
	TILE_DIM=32
	t1 = 0
	t2 = 0
	t3 = 0

	TILE_DIM_1 = 16
	TILE_DIM_2 = 4

	rep_times = 10

	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	griddim = (div(Nx,TILE_DIM_1)+1,div(Ny,TILE_DIM_2)+1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	y = D2y(u,Nx,Ny,h)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	y_gpu = collect(d_y)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v5(d_u,d_y2,Nx,Ny,h,Val(TILE_DIM))
	synchronize()
	y_gpu_2 = collect(d_y2)
	@cuda threads=blockdim blocks=griddim D2y_GPU_v7(d_u,d_y7, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	y_gpu_7 = collect(d_y7)
	# @show y_gpu - y
	# @show y_gpu_2 - y
	@show y ≈ y_gpu
	@show y ≈ y_gpu_2
	@show y ≈ y_gpu_7


	ty = time_ns()
	for i in 1:rep_times
		y = D2x(u,Nx,Ny,h)
	end
	ty_end = time_ns()
	t1 = ty_end - ty
	
	t_dy = time_ns()
	for i in 1:rep_times
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	end
	synchronize()
	# sync_threads()
	t_dy_end = time_ns()
	t2 = t_dy_end - t_dy

	t_dy_v2 = time_ns()
	for i in 1:rep_times
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v5(d_u,d_y2,Nx,Ny,h,Val(TILE_DIM))
	end
	synchronize()
	# sync_threads()
	t_dy_v2_end = time_ns()
	t3 = t_dy_v2_end - t_dy_v2

	t_dy_v7 = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim D2y_GPU_v7(d_u,d_y7, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	t_dy_v7_end = time_ns()
	t7 = t_dy_v7_end - t_dy_v7

	@show Float64(t1)
	@show Float64(t2)
	@show Float64(t3)
	@show Float64(t7)

	@show t1/t2
	@show t1/t3
	@show t1/t7

	memsize = length(u) * sizeof(eltype(u))
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)
	@printf("GPU Through-put %20.2f\n", 2 * memsize * rep_times / t2)
	@printf("GPU (v2) Through-put %20.2f\n", 2 * memsize * rep_times / t3)
	@printf("GPU (v7) Through-put %20.2f\n", 2 * memsize * rep_times / t7)

	return Float64(t1), Float64(t2), Float64(t3), Float64(t7)
end




function Operator_y_GPU(d_u, d_y, d2_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
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
	if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
		@inbounds tile[k+HALO_WIDTH,l] = d_u[global_index]
	end

	sync_threads()

	# For upper halo
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
		@inbounds tile[k,l] = d_u[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
		@inbounds tile[k+2*HALO_WIDTH,l] = d_u[global_index + HALO_WIDTH]
	end

	sync_threads()

	# Finite Difference Operations starts here

	# For d2_y, output second order differential operators in y direction

	#Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
		@inbounds d2_y[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		# @inbounds d_y[global_index] = (tile[k+HALO_WIDTH+1,l] - tile[k+HALO_WIDTH,l]) / h
		(tile[k+HALO_WIDTH+1,l] - tile[k+HALO_WIDTH,l]) / h
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		@inbounds d2_y[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		# @inbounds d_y[global_index] = (tile[k+HALO_WIDTH+1,l] - tile[k+HALO_WIDTH-1,l]) / (2*h)
		(tile[k+HALO_WIDTH+1,l] - tile[k+HALO_WIDTH-1,l]) / (2*h)
	end

	sync_threads()

	#Lower Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		@inbounds d2_y[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		# @inbounds d_y[global_index] = (tile[k+HALO_WIDTH,l] - tile[k+HALO_WIDTH-1,l]) / h
		(tile[k+HALO_WIDTH,l] - tile[k+HALO_WIDTH-1,l]) / h
	end

	# For d_y, output first order differential operators in y direction

	# if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
	# 	d_y[global_index] = (tile[k+HALO_WIDTH+2,l] - tile[k+HALO_WIDTH+1]) / h
	# end

	nothing
end


function tester_Operator_y_GPU(Nx)
	Ny = Nx
	h = 1/Nx
	TILE_DIM_1 = 16
	TILE_DIM_2 = 4

	u = randn(Nx*Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	d2_y = similar(d_u)
	d_y7 = similar(d_u)
	y = Dy(u,Nx,Ny,h)
	y2 = D2y(u,Nx,Ny,h)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	TILE_DIM = 32
	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	@cuda threads=blockdim blocks=griddim Operator_y_GPU(d_u,d_y, d2_y, Nx,Ny,h,Val(TILE_DIM_1),Val(TILE_DIM_2))
	@show Array(d_y) ≈ Array(y)
	@show Array(d2_y) ≈ Array(y2)


	# Starting test

	rep_times = 10

	ty = time_ns()
	for i in 1:rep_times
		y = Dy(u,Nx,Ny,h)
	end
	ty_end = time_ns()
	t1 = ty_end - ty

	memsize = length(u) * sizeof(eltype(u))
	@printf("CPU Through-put (Dy) %20.2f\n", 2 * memsize * rep_times / t1)

	t2y = time_ns()
	for i in 1:rep_times
		y2 = D2y(u,Nx,Ny,h)
	end
	t2y_end = time_ns()
	t2 = t2y_end - t2y

	@printf("CPU Through-put (D2y) %20.2f\n", 2 * memsize * rep_times / t1)
	@printf("CPU through-put (Dy + D2y, serial) %20.2f\n", 2 * memsize * rep_times / (t1+t2))

	td2y = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim D2y_GPU_v7(d_u,d_y7, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	td2y_end = time_ns()
	t3 = td2y_end - td2y
	@printf("GPU through-put (D2y_GPU_v7) %20.2f\n", 2 * memsize * rep_times / t3)

	t_GPUy = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim Operator_y_GPU(d_u,d_y, d2_y, Nx,Ny,h,Val(TILE_DIM_1),Val(TILE_DIM_2))
	end
	synchronize()
	t_GPUy_end = time_ns()
	t4 = t_GPUy_end - t_GPUy
	@printf("GPU through-put (Operator_y_GPU) %20.2f\n", 2* memsize * rep_times / t4)

	return Float64(t1), Float64(t2), Float64(t3), Float64(t4)

	return nothing
end

function tester_d2x_d2y(Nx)
	Ny = Nx
	u = randn(Nx * Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	d_y_new = similar(d_y)
	u_reordered = reshape(u,Nx,Ny)
	u_reordered_tranposed = u_reordered'
	u_reordered_transposed_aligned = u_reordered_tranposed[:]
	d_u_reordered_transposed_aligned = CuArray(u_reordered_transposed_aligned)
	h = 1/Nx
	TILE_DIM=32
	t1 = 0
	t2 = 0
	t3 = 0

	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU(d_u_reordered_transposed_aligned,d_y_new,Nx,Ny,h,Val(TILE_DIM))

	y1 = collect(d_y)
	y2 = collect(d_y_new)

	@show y1
	@show y2
end

function Dx(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny
	y[idx] = (u[idx .+ Ny] - u[idx]) ./ h

	idx1 = Ny+1:N-Ny
	y[idx1] = (u[idx1 .+ Ny]-u[idx1 .- Ny]) ./ (2*h)

	idx2 = N-Ny+1:N
	y[idx2] = (u[idx2]-u[idx2 .- Ny]) ./ h

	return y
end

function Dy(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny:N-Ny+1
	y[idx] = (u[idx .+ 1] - u[idx]) ./ h

	idx = Ny:Ny:N
	y[idx] = (u[idx] - u[idx .- 1]) ./h

	for j = 1:Nx
		idx = 2+(j-1)*Ny:j*Ny-1
		y[idx] = (u[idx .+ 1] - u[idx .- 1]) ./ (2*h)
	end

	return y
end

function Hxinv(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny
	y[idx] = (2*u[idx]) .* (1/h)

	idx = Ny+1:N-Ny
	y[idx] = (1*u[idx]) .* (1/h)

	idx = N-Ny+1:N
	y[idx] = (2*u[idx]) .* (1/h)

	return y
end

function Hyinv(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny:N-Ny+1
	y[idx] = (2*u[idx]) .* (1/h)

	idx = Ny:Ny:N
	y[idx] = (2*u[idx]) .* (1/h)

	for i = 1:Nx
		idx = 2+(i-1).*Ny:i*Ny-1
		y[idx] = u[idx] .* (1/h)
	end

	return y

end

function Hx(u, Nx, Ny, h)
	N = Nx*Ny
        y = zeros(N)

        idx = 1:Ny
	y[idx] = h .* (1/2)*u[idx]

        idx = Ny+1:N-Ny
        y[idx] = h .* 1*u[idx]

        idx = N-Ny+1:N
	y[idx] = h .* (1/2)*u[idx]

        return y


end

function Hy(u, Nx, Ny, h)
	N = Nx*Ny
        y = zeros(N)

        idx = 1:Ny:N-Ny+1
	y[idx] = h .* (1/2)*u[idx]

        idx = Ny:Ny:N
	y[idx] = h .* (1/2)*u[idx]

        for i = 1:Nx
                idx = 2+(i-1).*Ny:i*Ny-1
                y[idx] = h .* u[idx]
        end

        return y

end

function FACEtoVOL(u_face, face, Nx, Ny)
	N = Nx*Ny
	y = zeros(N)

	if face == 1
		idx = 1:Ny:N-Ny+1
	elseif face == 2
		idx = Ny:Ny:N
	elseif face == 3
		idx = 1:Ny
	elseif face == 4
		idx = N-Ny+1:N
	else
	end

	y[idx] .= u_face

	return y

end

function VOLtoFACE(u, face, Nx, Ny)
	N = Nx*Ny
        y = zeros(N)

        if face == 1
                idx = 1:Ny:N-Ny+1
        elseif face == 2
                idx = Ny:Ny:N
        elseif face == 3
                idx = 1:Ny
        elseif face == 4
                idx = N-Ny+1:N
        else
        end

	y[idx] = u[idx]
        return y
end

function Bx(Nx,Ny)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny
	y[idx] = -1 .* ones(Ny)

	idx = N-Ny+1:N
	y[idx] = 1 .* ones(Ny)
	return y
end

function Bx(u,Nx,Ny,h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny
	y[idx] = -1 .* ones(Ny)

	idx = N-Ny+1:N
	y[idx] = 1 .* ones(Ny)
	return y
end

function By(Nx,Ny)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny:N-Ny+1
	y[idx] = -1 .* ones(Ny)

	idx = Ny:Ny:N
	y[idx] = 1 .* ones(Ny)
	return y
end

function By(u,Nx,Ny,h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny:N-Ny+1
	y[idx] = -1 .* ones(Ny)

	idx = Ny:Ny:N
	y[idx] = 1 .* ones(Ny)
	return y
end

function BxSx(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny
	y[idx] = (1/h) .* (1.5 .* u[idx] - 2 .* u[idx .+ Ny] + 0.5 .* u[idx .+ 2*Ny])
	y[N-Ny .+ idx] = (1/h) .* (0.5 .* u[N-3*Ny .+ idx] - 2 .* u[N-2*Ny .+ idx] + 1.5 .* u[N-Ny .+ idx])

	return y

end

function BySy(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx = 1:Ny:N-Ny+1
	y[idx] = (1/h) .* (1.5 .* u[idx] - 2 .* u[idx .+ 1] + 0.5 .* u[idx .+ 2])

	idx = Ny:Ny:N
	y[idx] = (1/h) .* (0.5 .* u[idx .- 2] - 2 .* u[idx .- 1] + 1.5 .* u[idx])

	return y
end

function BxSx_tran(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx1 = 1:Ny
	y[idx1] += (1.5 .* u[idx1]) .* (1/h)
	idx = Ny+1:2*Ny
	y[idx] += (-2 .* u[idx1]) .* (1/h)
	idx  = 2*Ny+1:3*Ny
	y[idx] += (0.5 .* u[idx1]) .* (1/h)

	idxN = N-Ny+1:N
	y[idxN] += (1.5 .* u[idxN]) .* (1/h)
	idx = N-2*Ny+1:N-Ny
	y[idx] += (-2 .* u[idxN]) .* (1/h)
	idx = N-3*Ny+1:N-2*Ny
	y[idx] += (0.5 .* u[idxN]) .* (1/h)

	return y
end


function BySy_tran(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	idx1 = 1:Ny:N-Ny+1
	y[idx1] += (1.5 .* u[idx1]) .* (1/h)
	idx = 2:Ny:N-Ny+2
	y[idx] += (-2 .* u[idx1]) .* (1/h)
	idx = 3:Ny:N-Ny+3
	y[idx] += (0.5 .* u[idx1]) .* (1/h)

	idxN = Ny:Ny:N
	y[idxN] += (1.5 .* u[idxN]) .* (1/h)
	idx = Ny-1:Ny:N-1
	y[idx] += (-2 .* u[idxN]) .* (1/h)
	idx = Ny-2:Ny:N-2
	y[idx] += (0.5 .* u[idxN]) .* (1/h)

	return y
end
