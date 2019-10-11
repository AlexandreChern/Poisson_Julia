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


# variables order: u, Nx, Ny, N, hx, hy, y_**(container)


function D2x_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_D2x::Array{Float64,1})
	#N = Nx*Ny
	#hx = Float64(1/(Nx-1))
	#hy = Float64(1/(Ny-1))
	#y = similar(u)
	#y = similar(u)
	@inbounds  for idx = 1:Ny
		y_D2x[idx] = (u[idx] - 2*u[Ny + idx] + u[2*Ny + idx]) / hx^2
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_D2x[idx1] = (u[idx1 - Ny] - 2 * u[idx1] + u[idx1 + Ny]) / hx^2
	end

	@inbounds for idx2 = N-Ny+1:N
		y_D2x[idx2] = (u[idx2 - 2*Ny] -2 * u[idx2 - Ny] + u[idx2]) / hx^2
	end
	return y_D2x
end



function D2y_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_D2y::Array{Float64,1})
	#N = Nx*Ny
	#hx = Float64(1/(Nx-1))
	#hy = Float64(1/(Ny-1))
	@inbounds for idx = 1:Ny:N-Ny+1
		y_D2y[idx] = (u[idx] - 2 * u[idx + 1] + u[idx + 2]) / hy^2
	end

	@inbounds for idx1 = Ny:Ny:N
		y_D2y[idx1] = (u[idx1 - 2] - 2 * u[idx1 .- 1] + u[idx1]) / hy^2
	end

	@inbounds for j = 1:Nx
		@inbounds for idx = 2+(j-1)*Ny:j*Ny-1
			y_D2y[idx] = (u[idx - 1] - 2 * u[idx] + u[idx + 1]) / hy^2
		end
	end

	return y_D2y

end





function Dx_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_Dx::Array{Float64,1})

	@inbounds for idx = 1:Ny
		y_Dx[idx] = (u[idx + Ny] - u[idx]) / hx
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_Dx[idx1] = (u[idx1 + Ny]-u[idx1 - Ny]) / (2*hx)
	end

	@inbounds for idx2 = N-Ny+1:N
		y_Dx[idx2] = (u[idx2]-u[idx2 .- Ny]) ./ hx
	end
	return y_Dx
end




function Dy_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_Dy::Array{Float64,1})

	@inbounds for idx = 1:Ny:N-Ny+1
		y_Dy[idx] = (u[idx + 1] - u[idx]) / h
	end

	@inbounds for idx = Ny:Ny:N
		y_Dy[idx] = (u[idx] - u[idx - 1]) /h
	end

	@inbounds for j = 1:Nx
		@inbounds for idx = 2+(j-1)*Ny:j*Ny-1
		 	y_Dy[idx] = (u[idx + 1] - u[idx - 1]) / (2*h)
		end
	end
	return y_Dy
end



function Hxinv_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_Hxinv::Array{Float64,1})
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



function Hyinv_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_Hyinv::Array{Float64,1})
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



function Hx_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_Hx::Array{Float64,1})
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



function Hy_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_Hy::Array{Float64,1})
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



function FACEtoVOL_beta(u_face::Array{Float64,1}, face::Int64, Nx::Int64, Ny::Int64, N::Int64, yf2vs::Array{Array{Float64,1},1})
	if face == 1
		idx = 1:Ny:N-Ny+1
	elseif face==2
		idx = Ny:Ny:N
	elseif face==3
		idx = 1:Ny
	elseif face == 4
		idx = N-Ny+1:N
	else
	end
	yf2v = yf2vs[face]
	yf2v[idx] = uface
	return yf2v
end



function VOLtoFACE_beta(u::Array{Float64,1},face::Int64,Nx::Int64, Ny::Int64, N::Int64, yv2fs::Array{Array{Float64,1},1}) ## Has some issue
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
    yv2f = yv2fs[face]
	yv2f[idx] = u[idx]

	return yv2f
end


function Bx_beta(Nx::Int64,Ny::Int64, N::Int64, y_Bx::Array{Float64,1})

	@inbounds for idx=1:Ny
		y_Bx[idx] = -1
	end

	@inbounds for idx = N-Ny+1:N
		y_Bx[idx] = 1
	end
	return y_Bx
end


function By_beta(Nx::Int64,Ny::Int64,N::Int64,y_By::Array{Float64,1})
	@inbounds for idx = 1:Ny:N-Ny+1
		y_By[idx] = -1
	end

	@inbounds for idx1 = Ny:Ny:N
		y_By[idx1] = 1
	end
	return y_By
end


function BxSx_beta(u::Array{Float64,1},Nx::Int64,Ny::Int64,N::Int64,hx::Float64,hy::Float64,y_BxSx::Array{Float64,1})
	@inbounds for idx = 1:Ny
		y_BxSx[idx] = (1/hx) * (1.5 * u[idx] - 2 * u[idx + Ny] + 0.5 * u[idx + 2*Ny])
		y_BxSx[N-Ny + idx] = (1/hx) * (0.5 * u[N-3*Ny + idx] - 2 * u[N-2*Ny + idx] + 1.5 * u[N-Ny + idx])
	end
	return y_BxSx
end


function BySy_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64,y_BySy::Array{Float64,1})
	@inbounds for idx = 1:Ny:N-Ny+1
		y_BySy[idx] = (1/hy) * (1.5 * u[idx] - 2 * u[idx .+ 1] + 0.5 * u[idx .+ 2])
	end

	@inbounds for idx = Ny:Ny:N
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

function BxSx_tran_beta(u::Array{Float64,1},Nx::Int64,Ny::Int64,N::Int64,hx::Float64,hy::Float64,y_BxSx_tran::Array{Float64,1}) # be careful with += expression
	#hx = Float64(1/(Nx-1))
   	#hy = Float64(1/(Ny-1))
	@inbounds for idx1 = 1:Ny
		y_BxSx_tran[idx1] = (1.5 * u[idx1]) * (1/hx)

		# for idx = Ny+1:2*Ny
		y_BxSx_tran[idx1+Ny] = (-2 * u[idx1]) * (1/hx)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y_BxSx_tran[idx1+2Ny] = (0.5 * u[idx1]) * (1/hx)
		# end
	end

	@inbounds for idxN = N-Ny+1:N
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


function BySy_tran_beta(u::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_BySy::Array{Float64,1}) # Be Careful about double foor loops

	for idx1 = 1:Ny:N-Ny+1
		y_BySy[idx1] += (1.5 * u[idx1]) * (1/h)

		# for idx = Ny+1:2*Ny
		y_BySy[idx1+1] += (-2 * u[idx1]) * (1/h)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y_BySy[idx1+2] += (0.5 * u[idx1]) * (1/h)
		# end
	end

	for idxN = Ny:Ny:N
		y_BySy[idxN] += (1.5 * u[idxN]) * (1/h)

		# for idx = N-2*Ny+1:N-Ny
		y_BySy[idxN-1] += (-2 * u[idxN]) * (1/h)
		# end

		# for idx = N-3*Ny+1:N-2*Ny
		y_BySy[idxN-2] += (0.5 * u[idxN]) * (1/h)
		# end
	end

	return y_BySy
end
