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




