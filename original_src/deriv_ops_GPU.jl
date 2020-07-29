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
	y[idx] = (u[idx] - 2 .* u[Ny .+ idx] + u[2*Ny .+ idx]) ./ h^2

	idx1 = Ny+1:N-Ny
	y[idx1] = (u[idx1 .- Ny] - 2 .* u[idx1] + u[idx1 .+ Ny]) ./ h^2

	idx2 = N-Ny+1:N
	y[idx2] = (u[idx2 .- 2*Ny] -2 .* u[idx2 .- Ny] + u[idx2]) ./ h^2

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

	nothing
end

function D2x_GPU_v2(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	# d_y = zeros(N)
	if tidx <= Ny
		d_y[tidx] = (d_u[tidx] - 2 * d_u[Ny + tidx] + d_u[2*Ny + tidx]) / h^2
	end

	if Ny+1 <= tidx <= N-Ny
		d_y[tidx] = (d_u[tidx - Ny] - 2 .* d_u[tidx] + d_u[tidx + Ny]) / h^2
	end


	if N-Ny+1 <= tidx <= N
		d_y[tidx] = (d_u[tidx - 2*Ny] -2 * d_u[tidx - Ny] + d_u[tidx]) / h^2
	end

	sync_threads()

	nothing
end


function tester_D2x(Nx)
	# Nx = Ny = 1000;
	Ny = Nx
	u = randn(Nx * Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	h = 1/Nx
	TILE_DIM=32
	t1 = 0
	t2 = 0
	t3 = 0

	rep_times = 10

	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	y = D2x(u,Nx,Ny,h)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	y_gpu = collect(d_y)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2x_GPU_v2(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	y_gpu_2 = collect(d_y)
	@show y ≈ y_gpu
	@show y ≈ y_gpu_2

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

	@show Float64(t1)
	@show Float64(t2)
	@show Float64(t3)

	@show t1/t2
	@show t1/t3

	memsize = length(u) * sizeof(eltype(u))
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)
	@printf("GPU Through-put %20.2f\n", 2 * memsize * rep_times / t2)
	@printf("GPU (v2) Through-put %20.2f\n", 2 * memsize * rep_times / t3)

	return Float64(t1), Float64(t2), Float64(t3)
end

function tester_D2y(Nx)
	# Nx = Ny = 1000;
	Ny = Nx
	u = randn(Nx * Ny)
	d_u = CuArray(u)
	d_y = similar(d_u)
	d_y2 = similar(d_u)
	h = 1/Nx
	TILE_DIM=32
	t1 = 0
	t2 = 0
	t3 = 0

	rep_times = 10

	THREAD_NUM = 32
	BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

	y = D2y(u,Nx,Ny,h)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	y_gpu = collect(d_y)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v4(d_u,d_y2,Nx,Ny,h,Val(TILE_DIM))
	y_gpu_2 = collect(d_y2)
	@show y ≈ y_gpu
	@show y ≈ y_gpu_2
	# @show y_gpu - y
	# @show y_gpu_2 - y

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
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v4(d_u,d_y,Nx,Ny,h,Val(TILE_DIM))
	end
	synchronize()
	# sync_threads()
	t_dy_v2_end = time_ns()
	t3 = t_dy_v2_end - t_dy_v2

	@show Float64(t1)
	@show Float64(t2)
	@show Float64(t3)

	@show t1/t2
	@show t1/t3

	memsize = length(u) * sizeof(eltype(u))
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)
	@printf("GPU Through-put %20.2f\n", 2 * memsize * rep_times / t2)
	@printf("GPU (v2) Through-put %20.2f\n", 2 * memsize * rep_times / t3)

	return Float64(t1), Float64(t2), Float64(t3)
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
	if 1 <= tidx <= N
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

	tb = tidx * Ny
	if 1 <= tb <= N
		d_y[tb] = (d_u[tb] - 2d_u[tb - 1] + d_u[tb - 2]) / h^2
	end
	sync_threads()

	te = (tidx-1) * Ny + 1
	if 1 <= te <= N
		d_y[te] = (d_u[te] - 2d_u[te + 1] + d_u[te + 2]) / h^2
	end
	sync_threads()
	nothing
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

	y[idx] = u_face

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

function By(Nx,Ny)
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