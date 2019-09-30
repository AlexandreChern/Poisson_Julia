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





function D2x(u, Nx, Ny, h) # Original implementation
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


# function D2x_alpha(u, Nx, Ny, h) # Original implementation
# 	N = Nx*Ny
# 	y = zeros(N)
# 	idx = 1:Ny
# 	y[idx] .= (u[idx] - 2 .* u[Ny .+ idx] + u[2*Ny .+ idx]) ./ h^2
#
# 	idx1 = Ny+1:N-Ny
# 	y[idx1] .= (u[idx1 .- Ny] - 2 .* u[idx1] + u[idx1 .+ Ny]) ./ h^2
#
# 	idx2 = N-Ny+1:N
# 	y[idx2] .= (u[idx2 .- 2*Ny] -2 .* u[idx2 .- Ny] + u[idx2]) ./ h^2
#
# 	return y
# end


# function D2x_new(u, Nx, Ny, h)
# 	N = Nx*Ny;
# 	y = zeros(N)
# 	idx = 1:Ny
# 	y[idx] = (u[idx] - 2 * u[Ny+1 : 2Ny] + u[2*Ny+1 : 3Ny]) / h^2
#
# 	idx1 = Ny+1:N-Ny
# 	y[idx1] = (u[1 : N - 2Ny] - 2 * u[idx1] + u[2Ny+1 : N]) / h^2
#
# 	idx2 = N-Ny+1:N
# 	y[idx2] = (u[N-3Ny+1 : N- 2*Ny] -2 * u[N-2Ny+1 : N- Ny] + u[idx2]) / h^2
#
# 	y
# end

# function D2x(u, Nx, Ny, h)
# 	N = Nx*Ny
# 	y = zeros(N)
# 	for idx = 1:Ny
# 		y[idx] = (u[idx] - 2 * u[Ny + idx] + u[2*Ny + idx]) / h^2
# 	end
#
# 	for idx1 = Ny+1:N-Ny
# 		y[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / h^2
# 	end
#
# 	for idx2 = N-Ny+1:N
# 		y[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / h^2
# 	end
# 	return y
# end

# function D2x_test(u::Array{Float64,1}, Nx::Int64, Ny::Int64, h::Float64)
# 	N = Nx*Ny
# 	#y = similar(u)
# 	y = similar(u)
# 	for idx = 1:Ny
# 		y[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / h^2
# 	end
#
# 	for idx1 = Ny+1:N-Ny
# 		y[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / h^2
# 	end
#
# 	for idx2 = N-Ny+1:N
# 		y[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / h^2
# 	end
# 	return y
# end

# function D2x_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64)
# 	N = Nx*Ny
# 	hx = Float64(1/(Nx-1))
# 	hy = Float64(1/(Ny-1))
# 	#y = similar(u)
# 	y = Array{Float64,1}(undef,N)
# 	for idx = 1:Ny
# 		y[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / hx^2
# 	end
#
# 	for idx1 = Ny+1:N-Ny
# 		y[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / hx^2
# 	end
#
# 	for idx2 = N-Ny+1:N
# 		y[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / hx^2
# 	end
# 	return y
# end

function D2x_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64,y1::Array{Float64,1})
	N = Nx*Ny
	hx = Float64(1/(Nx-1))
	hy = Float64(1/(Ny-1))
	#y = similar(u)
	#y = similar(u)
	@inbounds  for idx = 1:Ny
		y1[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / hx^2
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y1[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / hx^2
	end

	@inbounds for idx2 = N-Ny+1:N
		y1[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / hx^2
	end
	return y1
end


function D2x_beta_2(u::Array{Float64,1}, Nx::Int64, Ny::Int64,y1::Array{Float64,1})
	N = Nx*Ny
	hx = Float64(1/(Nx-1))
	hy = Float64(1/(Ny-1))
	#y = similar(u)
	#y = similar(u)
	@inbounds @simd for idx = 1:Ny
		y1[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / hx^2
	end

	@inbounds @simd for idx1 = Ny+1:N-Ny
		y1[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / hx^2
	end

	@inbounds @simd for idx2 = N-Ny+1:N
		y1[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / hx^2
	end
	return y1
end
# function D2x_beta_3(u::Array{Float64,1}, Nx::Int64, Ny::Int64,y::Array{Float64,1})
# 	N = Nx*Ny
# 	hx = Float64(1/(Nx-1))
# 	hy = Float64(1/(Ny-1))
# 	#y = similar(u)
# 	#y = similar(u)
# 	@inbounds @simd for idx = 1:Ny
# 		y[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / hx^2
# 	end
#
# 	@inbounds @simd for idx1 = Ny+1:N-Ny
# 		y[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / hx^2
# 	end
#
# 	@inbounds @simd for idx2 = N-Ny+1:N
# 		y[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / hx^2
# 	end
# 	return y
# end



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

# function D2y(u, Nx, Ny, h)
# 	N = Nx*Ny
# 	y = zeros(N)
# 	for idx = 1:Ny:N-Ny+1
# 		y[idx] = (u[idx] - 2 * u[idx + 1] + u[idx + 2]) / h^2
# 	end
#
# 	for idx1 = Ny:Ny:N
# 		y[idx1] = (u[idx1 - 2] - 2 * u[idx1 .- 1] + u[idx1]) / h^2
# 	end
#
# 	for j = 1:Nx
# 		for idx = 2+(j-1)*Ny:j*Ny-1
# 			y[idx] = (u[idx - 1] - 2 * u[idx] + u[idx + 1]) / h^2
# 		end
# 	end
#
# 	return y
#
# end

function D2y_test(u::Array{Float64,1}, Nx::Int64, Ny::Int64, h::Float64)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny:N-Ny+1
		y[idx] = (u[idx] - 2 * u[idx + 1] + u[idx + 2]) / h^2
	end

	for idx1 = Ny:Ny:N
		y[idx1] = (u[idx1 - 2] - 2 * u[idx1 .- 1] + u[idx1]) / h^2
	end

	for j = 1:Nx
		for idx = 2+(j-1)*Ny:j*Ny-1
			y[idx] = (u[idx - 1] - 2 * u[idx] + u[idx + 1]) / h^2
		end
	end

	return y

end

function D2y_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, y2::Array{Float64,1})
	N = Nx*Ny
	hx = Float64(1/(Nx-1))
	hy = Float64(1/(Ny-1))
	@inbounds for idx = 1:Ny:N-Ny+1
		y2[idx] = (u[idx] - 2 * u[idx + 1] + u[idx + 2]) / hy^2
	end

	@inbounds for idx1 = Ny:Ny:N
		y2[idx1] = (u[idx1 - 2] - 2 * u[idx1 .- 1] + u[idx1]) / hy^2
	end

	@inbounds for j = 1:Nx
		@inbounds for idx = 2+(j-1)*Ny:j*Ny-1
			y2[idx] = (u[idx - 1] - 2 * u[idx] + u[idx + 1]) / hy^2
		end
	end

	return y2

end


function D2_test(u::Array{Float64,1},Nx::Int64,Ny::Int64,h::Float64)
	N = Nx*Ny
	y = similar(u)
	z = similar(u)
	for idx = 1:Ny
		y[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / h^2
	end

	for idx1 = Ny+1:N-Ny
		y[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / h^2
	end

	for idx2 = N-Ny+1:N
		y[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / h^2
	end

	for idx = 1:Ny:N-Ny+1
		z[idx] = (u[idx] - 2 * u[idx + 1] + u[idx + 2]) / h^2
	end

	for idx1 = Ny:Ny:N
		z[idx1] = (u[idx1 - 2] - 2 * u[idx1 .- 1] + u[idx1]) / h^2
	end

	for j = 1:Nx
		for idx = 2+(j-1)*Ny:j*Ny-1
			z[idx] = (u[idx - 1] - 2 * u[idx] + u[idx + 1]) / h^2
		end
	end
	return y+z
end


function D2_beta(u::Array{Float64,1},Nx::Int64,Ny::Int64,y::Array{Float64,1})
	#y1 = Array{Float64,1}(undef,Nx*Ny)
	y1 = copy(D2y_beta_2(u::Array{Float64,1}, Nx::Int64, Ny::Int64, y::Array{Float64,1}))
	y2 = D2x_beta_2(u::Array{Float64,1}, Nx::Int64, Ny::Int64, y::Array{Float64,1})
	return y1+y2
end

function D2_beta_2(u::Array{Float64,1},Nx::Int64,Ny::Int64,y1::Array{Float64,1},y2::Array{Float64,1})
	#y1 = Array{Float64,1}(undef,Nx*Ny)

	return D2x_beta(u,Nx,Ny,y1) + D2x_beta(u,Nx,Ny,y2)
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
en


function Dx_test(u::Array{Float64,1}, Nx::Int64, Ny::Int64, h::Float64)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny
		y[idx] = (u[idx + Ny] - u[idx]) / h
	end

	for idx1 = Ny+1:N-Ny
		y[idx1] = (u[idx1 + Ny]-u[idx1 - Ny]) / (2*h)
	end

	for idx2 = N-Ny+1:N
		y[idx2] = (u[idx2]-u[idx2 .- Ny]) ./ h
	end
	return y

end



function Dx_test!(y::Array{Float64,1},u::Array{Float64,1}, N::Int64,Nx::Int64, Ny::Int64, h::Float64)
	#N = Nx*Ny
	#y = similar(u)

	for idx = 1:Ny
		y[idx] = (u[idx + Ny] - u[idx]) / h
	end

	for idx1 = Ny+1:N-Ny
		y[idx1] = (u[idx1 + Ny]-u[idx1 - Ny]) / (2*h)
	end

	for idx2 = N-Ny+1:N
		y[idx2] = (u[idx2]-u[idx2 .- Ny]) ./ h
	end
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

function Dy_test(u, Nx, Ny, h)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny:N-Ny+1
		y[idx] = (u[idx + 1] - u[idx]) / h
	end

	for idx = Ny:Ny:N
		y[idx] = (u[idx] - u[idx - 1]) /h
	end

	for j = 1:Nx
		for idx = 2+(j-1)*Ny:j*Ny-1
			y[idx] = (u[idx + 1] - u[idx - 1]) / (2*h)
		end
	end

	return y
end

#z = zeros(N)

function Dy_test_1(u, y, Nx, Ny, h)
	N = Nx*Ny
	#y = copy(z)
	for idx = 1:Ny:N-Ny+1
		y[idx] = (u[idx + 1] - u[idx]) / h
	end

	for idx = Ny:Ny:N
		y[idx] = (u[idx] - u[idx - 1]) /h
	end

	for j = 1:Nx
		for idx = 2+(j-1)*Ny:j*Ny-1
			y[idx] = (u[idx + 1] - u[idx - 1]) / (2*h)
		end
	end
	return y
end


function Dy_test_2(u, y, Nx, Ny, h)
	N = Nx*Ny
	#y = copy(z)
	for idx = 1:Ny:N-Ny+1
		@inbounds y[idx] = (u[idx + 1] - u[idx]) / h
	end

	for idx = Ny:Ny:N
		@inbounds y[idx] = (u[idx] - u[idx - 1]) /h
	end

	for j = 1:Nx
		for idx = 2+(j-1)*Ny:j*Ny-1
			@inbounds y[idx] = (u[idx + 1] - u[idx - 1]) / (2*h)
		end
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

function Hxinv_test(u,Nx,Ny,h)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny
		y[idx] = (2*u[idx]) * (1/h)
	end

	for idx1 = Ny+1:N-Ny
		y[idx1] = (1*u[idx1]) * (1/h)
	end

	for idx2 = N-Ny+1:N
		y[idx2] = (2*u[idx2]) * (1/h)
	end

	return y
end

y_Hxinv = zeros(Nx*Ny)
N

function Hxinv_beta(u,Nx,Ny,N,y_Hxinv,hx,hy)
	#N = Nx*Ny
	#y = similar(u)
	@inbounds for idx = 1:Ny
		y_Hxinv[idx] = (2*u[idx]) * (1/hx)
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_Hxinv[idx1] = (1*u[idx1]) * (1/hx)
	end

	@inbounds for idx2 = N-Ny+1:N
		y_Hxinv[idx2] = (2*u[idx2]) * (1/hx)
	end

	return y_Hxinv
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

function Hyinv_test(u,Nx,Ny,h)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny:N-Ny+1
		y[idx] = (2*u[idx]) * (1/h)
	end

	for idx1 = Ny:Ny:N
		y[idx1] = (2*u[idx1]) * (1/h)
	end

	for i = 1:Nx
		for idx2 = 2+(i-1)*Ny:i*Ny-1
			y[idx2] = (u[idx2]) * (1/h)
		end
	end

	return y
end

y_Hyinv = Array{Float64,1}(undef,Nx*Ny)

function Hyinv_beta(u,Nx,Ny,N,y_Hyinv,hx,hy)
	#N = Nx*Ny
	#y = similar(u)
	#hx = Float64(1/(Nx-1))
	#hy = Float64(1/(Ny-1))
	@inbounds for idx = 1:Ny:N-Ny+1
		y_Hyinv[idx] = (2*u[idx]) * (1/hy)
	end

	@inbounds for idx1 = Ny:Ny:N
		y_Hyinv[idx1] = (2*u[idx1]) * (1/hy)
	end

	@inbounds for i = 1:Nx
		@inbounds for idx2 = 2+(i-1)*Ny:i*Ny-1
			y_Hyinv[idx2] = (u[idx2]) * (1/hy)
		end
	end

	return y_Hyinv
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

function Hx_test(u,Nx,Ny,h)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny
		y[idx] = h*u[idx]/2
	end

	for idx1 = Ny+1:N-Ny
		y[idx1] = h*u[idx1]
	end

	for idx2 = N-Ny+1:N
		y[idx2] = h*u[idx2]/2
	end
	return y
end

y_Hx = zeros(N)

# function Hx_beta_test(u,Nx,Ny,N,hx,hy,y_Hx)
# 	for idx = 1:Ny
# 		y_Hx[idx] = hx*u[idx]/2
# 	end
#
# 	for idx1 = Ny+1:N-Ny
# 		y_Hx[idx1] = hx*u[idx1]
# 	end
#
# 	for idx2 = N-Ny+1:N
# 		y_Hx[idx2] = hx*u[idx2]/2
# 	end
# 	return y_Hx
# end

function Hx_beta(u,Nx,Ny,N,hx,hy,y_Hx)
	@inbounds for idx = 1:Ny
		y_Hx[idx] = hx*u[idx]/2
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_Hx[idx1] = hx*u[idx1]
	end

	@inbounds for idx2 = N-Ny+1:N
		y_Hx[idx2] = hx*u[idx2]/2
	end
	return y_Hx
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

function Hy_test(u,Nx,Ny,h)
	N = Nx*Ny
	y = similar(u)

	for idx = 1:Ny:N-Ny+1
		y[idx] = h*u[idx]/2
	end

	for idx1 = Ny:Ny:N
		y[idx1] = h*u[idx1]/2
	end

	for i = 1:Nx
		for idx2 = 2 + (i-1)*Ny:i*Ny-1
			y[idx2] = h*u[idx2]
		end
	end
	return y
end

y_Hy = zeros(Nx*Nx)

function Hy_beta(u,Nx,Ny,N,hx,hy,y_Hy)
	#N = Nx*Ny
	#y = similar(u)

	@inbounds for idx = 1:Ny:N-Ny+1
		y_Hy[idx] = hy*u[idx]/2
	end

	@inbounds for idx1 = Ny:Ny:N
		y_Hy[idx1] = hy*u[idx1]/2
	end

	@inbounds for i = 1:Nx
		@inbounds for idx2 = 2 + (i-1)*Ny:i*Ny-1
			y_Hy[idx2] = hy*u[idx2]
		end
	end
	return y_Hy
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

function VOLtoFACE(u, face, Nx, Ny)  ## This cause bugs for using Different Nx and Ny
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

yv2f1 = zeros(Nx*Ny)

function VOLtoFACE_beta(u,face,Nx,Ny,N,yv2f) ## Has some issue
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

	yv2f[idx] = u[idx]

	return yv2f
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

function Bx_test(Nx,Ny)
	N = Nx*Ny
	y = zeros(N)

	for idx=1:Ny
		y[idx] = -1
	end

	for idx = N-Ny+1:N
		y[idx] = 1
	end
	return y
end

y_Bx = zeros(N)
function Bx_beta(Nx,Ny,N,y_Bx)
	#N = Nx*Ny
	#y = zeros(N)

	for idx=1:Ny
		y_Bx[idx] = -1
	end

	for idx = N-Ny+1:N
		y_Bx[idx] = 1
	end
	return y_Bx
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

function By_test(Nx,Ny)
	N= Nx*Ny
	y = zeros(N)

	for idx = 1:Ny:N-Ny+1
		y[idx] = -1
	end

	for idx1 = Ny:Ny:N
		y[idx1] = 1
	end
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

function BxSx_test(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	for idx = 1:Ny
		y[idx] = (1/h) * (1.5 * u[idx] - 2 * u[idx + Ny] + 0.5 * u[idx + 2*Ny])
		y[N-Ny + idx] = (1/h) * (0.5 * u[N-3*Ny + idx] - 2 * u[N-2*Ny + idx] + 1.5 * u[N-Ny + idx])
	end
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

function BySy_test(u, Nx, Ny, h)
	N = Nx*Ny
	y = zeros(N)

	for idx = 1:Ny:N-Ny+1
		y[idx] = (1/h) * (1.5 * u[idx] - 2 * u[idx .+ 1] + 0.5 * u[idx .+ 2])
	end

	for idx = Ny:Ny:N
		y[idx] = (1/h) * (0.5 * u[idx - 2] - 2 * u[idx - 1] + 1.5 * u[idx])
	end

	return y
end

y_BySy = zeros(Nx*Ny)
function BySy_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, y_BySy::Array{Float64,1})
	N = Nx*Ny
	hx = Float64(1/(Nx-1))
	hy = Float64(1/(Ny-1))
	for idx = 1:Ny:N-Ny+1
		y_BySy[idx] = (1/hy) * (1.5 * u[idx] - 2 * u[idx .+ 1] + 0.5 * u[idx .+ 2])
	end

	for idx = Ny:Ny:N
		y_BySy[idx] = (1/hy) * (0.5 * u[idx - 2] - 2 * u[idx - 1] + 1.5 * u[idx])
	end

	return y_BySy
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

function BxSx_tran_test(u, Nx, Ny, h) # Be Careful about double foor loops
	N = Nx*Ny
	y = zeros(N)

	for idx1 = 1:Ny
		y[idx1] += (1.5 * u[idx1]) * (1/h)

		# for idx = Ny+1:2*Ny
		y[idx1+Ny] += (-2 * u[idx1]) * (1/h)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y[idx1+2Ny] += (0.5 * u[idx1]) * (1/h)
		# end
	end

	for idxN = N-Ny+1:N
		y[idxN] += (1.5 * u[idxN]) * (1/h)

		# for idx = N-2*Ny+1:N-Ny
		y[idxN-Ny] += (-2 * u[idxN]) * (1/h)
		# end

		# for idx = N-3*Ny+1:N-2*Ny
		y[idxN-2Ny] += (0.5 * u[idxN]) * (1/h)
		# end
	end

	return y
end


y_BxSx_tran = zeros(Nx*Ny)

function BxSx_tran_beta(u,Nx,Ny,N, y_BxSx_tran,hx,hy) # be careful with += expression
	#hx = Float64(1/(Nx-1))
   	#hy = Float64(1/(Ny-1))
	for idx1 = 1:Ny
		y_BxSx_tran[idx1] = (1.5 * u[idx1]) * (1/hx)

		# for idx = Ny+1:2*Ny
		y_BxSx_tran[idx1+Ny] = (-2 * u[idx1]) * (1/hx)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y_BxSx_tran[idx1+2Ny] = (0.5 * u[idx1]) * (1/hx)
		# end
	end

	for idxN = N-Ny+1:N
		y_BxSx_tran[idxN] = (1.5 * u[idxN]) * (1/hx)

		# for idx = N-2*Ny+1:N-Ny
		y_BxSx_tran[idxN-Ny] = (-2 * u[idxN]) * (1/hx)
		# end

		# for idx = N-3*Ny+1:N-2*Ny
		y_BxSx_tran[idxN-2Ny] = (0.5 * u[idxN]) * (1/hx)
		# end
	end
	return y_BxSx_tran
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


function BySy_tran_test(u, Nx, Ny, h) # Be Careful about double foor loops
	N = Nx*Ny
	y = zeros(N)

	for idx1 = 1:Ny:N-Ny+1
		y[idx1] += (1.5 * u[idx1]) * (1/h)

		# for idx = Ny+1:2*Ny
		y[idx1+1] += (-2 * u[idx1]) * (1/h)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y[idx1+2] += (0.5 * u[idx1]) * (1/h)
		# end
	end

	for idxN = Ny:Ny:N
		y[idxN] += (1.5 * u[idxN]) * (1/h)

		# for idx = N-2*Ny+1:N-Ny
		y[idxN-1] += (-2 * u[idxN]) * (1/h)
		# end

		# for idx = N-3*Ny+1:N-2*Ny
		y[idxN-2] += (0.5 * u[idxN]) * (1/h)
		# end
	end

	return y
end
