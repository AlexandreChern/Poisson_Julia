#Solve 2D Poisson: u_xx + u_yy = f(x,y), on the unit square with b.c.
# u(0,y) = g3(y), u(1,y) = g4(y), -u_y(x,0) = g1(x), u_y(x,1) = g2(x)
# Take the exact solution u(x,y) = sin(pi*x + pi*y)

#Transfers to discretized system
#(D2x + D2y + P1 + P2 + P3 + P4)u = b + f where
#P1 = alpha1*Hyinv*E1*BySy
#P2 = alphas2*Hyinv*E2*BySy
#P3 = alpha3*Hxinv*E3 + beta*Hxinv*BxSx_tran*E3
#P4 = alpha4*Hxinv*E4 + beta*Hxinv*BxSx_tran*E4

#b = alpha1*Hyinv*E1*g1 + alpha2*Hyinv*E2*g2 + alpha3*Hxinv*E3*g3 + beta*Hxinv*BxSx_tran*E3*g3 + ...
#    alpha4*Hxinv*E4*g4 + beta*Hxinv*BxSx_tran*E4*g4

#to make system PD, multiply by -(H kron H):


include("deriv_ops_test.jl")
using SparseArrays
using LinearMaps
using IterativeSolvers
using Parameters



@with_kw struct variables
    h = 0.05
    dx = h
    dy = h
    x = 0:dx:1
    y = 0:dy:1
    Nx = length(x)
    Ny = length(y)
    alpha1 = -1
    alpha2 = -1
    alpha3 = -13/dy
    alpha4 = -13/dy
    beta = 1
end

var_test = variables()


# Array Containers
y1 = Array{Float64,1}(undef,Nx*Ny) # container for D2x
y2 = Array{Float64,1}(undef,Nx*Ny) # container for D2y
y_BySy = zeros(Nx*Ny)
y_Hyinv = zeros(Nx*Ny)
yv2f1 = zeros(Nx*Ny)
yv2f2 = zeros(Nx*Ny)
yv2f3 = zeros(Nx*Ny)
yv2f4 = zeros(Nx*Ny)

y_BxSx_tran = zeros(Nx*Ny)
y_Hxinv = zeros(Nx*Ny)


y_Bx = zeros(N)
y_By = zeros(N)



### Passing variables

@unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test
#function myMAT!(du::AbstractVector, u::AbstractVector,var_test::variables)
	#Chunk below should be passed as input, but for now needs to match chunk below

function myMAT_original!(du::AbstractVector, u::AbstractVector)
    # 	h = 0.05
    # 	dx = h
    # 	dy = h
    # 	x = 0:dx:1
    #         y = 0:dy:1
    # 	Nx = length(x)
    #         Ny = length(y)
    # 	alpha1 = -1
    #         alpha2 = -1
    #         alpha3 = -13/dy
    #         alpha4 = -13/dy
    #         beta = 1
    @unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test
    	########################################

        du_ops = D2x(u,Nx,Ny,dx) + D2y(u,Nx,Ny,dy) #compute action of D2x + D2y

        du1 = BySy(u,Nx,Ny,dy)
        du2 = VOLtoFACE(du1,1,Nx,Ny)
        du3 = alpha1*Hyinv(du2,Nx,Ny,dy)  #compute action of P1

        du4 = BySy(u,Nx,Ny,dy)
        du5 = VOLtoFACE(du4,2,Nx,Ny)
        du6 = alpha2*Hyinv(du5,Nx,Ny,dy) #compute action of P2

        du7 = VOLtoFACE(u,3,Nx,Ny)
        du8 = BxSx_tran(du7,Nx,Ny,dx)
        du9 = beta*Hxinv(du8,Nx,Ny,dx)
        du10 = VOLtoFACE(u,3,Nx,Ny)
        du11 = alpha3*Hxinv(du10,Nx,Ny,dx) #compute action of P3

        du12 = VOLtoFACE(u,4,Nx,Ny)
        du13 = BxSx_tran(du12,Nx,Ny,dx)
        du14 = beta*Hxinv(du13,Nx,Ny,dx)
        du15 = VOLtoFACE(u,4,Nx,Ny)
        du16 = alpha4*Hxinv(du15,Nx,Ny,dx) #compute action of P4


        du0 = du_ops + du3 + du6 + du9 + du11 + du14 + du16 #Collect together

            #compute action of -Hx kron Hy:

        du17 = Hy(du0, Nx, Ny, dy)
	du[:] = -Hx(du17,Nx,Ny,dx)
end


function myMAT!(du::AbstractVector, u::AbstractVector)
# 	h = 0.05
# 	dx = h
# 	dy = h
# 	x = 0:dx:1
#         y = 0:dy:1
# 	Nx = length(x)
#         Ny = length(y)
# 	alpha1 = -1
#         alpha2 = -1
#         alpha3 = -13/dy
#         alpha4 = -13/dy
#         beta = 1
#    #@unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test
#	########################################
#    # y1 = Array{Float64,1}(undef,Nx*Ny)
#    # y2 = Array{Float64,1}(undef,Nx*Ny)

    du_ops = D2x(u,Nx,Ny,dx) + D2y(u,Nx,Ny,dy) #compute action of D2x + D2y
    #du_ops = D2x_beta(u,Nx,Ny,y1) + D2y_beta(u,Nx,Ny,y2)
    du1 = BySy_test(u,Nx,Ny,dy)
    du2 = VOLtoFACE(du1,1,Nx,Ny)
    du3 = alpha1*Hyinv_test(du2,Nx,Ny,dy)  #compute action of P1

    du4 = BySy_test(u,Nx,Ny,dy)
    du5 = VOLtoFACE(du4,2,Nx,Ny)
    du6 = alpha2*Hyinv_test(du5,Nx,Ny,dy) #compute action of P2

    du7 = VOLtoFACE(u,3,Nx,Ny)
    du8 = BxSx_tran_test(du7,Nx,Ny,dx)
    du9 = beta*Hxinv_test(du8,Nx,Ny,dx)
    du10 = VOLtoFACE(u,3,Nx,Ny)
    du11 = alpha3*Hxinv_test(du10,Nx,Ny,dx) #compute action of P3

    du12 = VOLtoFACE(u,4,Nx,Ny)
    du13 = BxSx_tran_test(du12,Nx,Ny,dx)
    du14 = beta*Hxinv_test(du13,Nx,Ny,dx)
    du15 = VOLtoFACE(u,4,Nx,Ny)
    du16 = alpha4*Hxinv_test(du15,Nx,Ny,dx) #compute action of P4


    du0 = du_ops + du3 + du6 + du9 + du11 + du14 + du16 #Collect together

        #compute action of -Hx kron Hy:

    du17 = Hy_test(du0, Nx, Ny, dy)
	du[:] = -Hx_test(du17,Nx,Ny,dx)
end


function myMAT_new!(du::AbstractVector, u::AbstractVector)
# 	h = 0.05
# 	dx = h
# 	dy = h
# 	x = 0:dx:1
#         y = 0:dy:1
# 	Nx = length(x)
#         Ny = length(y)
# 	alpha1 = -1
#         alpha2 = -1
#         alpha3 = -13/dy
#         alpha4 = -13/dy
#         beta = 1
    #@unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test
	########################################
    # y1 = Array{Float64,1}(undef,Nx*Ny)
    # y2 = Array{Float64,1}(undef,Nx*Ny)

        #du_ops = D2x(u,Nx,Ny,dx) + D2y(u,Nx,Ny,dy) #compute action of D2x + D2y
        du_ops = D2x_beta(u,Nx,Ny,y1) + D2y_beta(u,Nx,Ny,y2)
        #du_ops = D2_beta_2(u,Nx,Ny,y1,y2)
        #du1 = BySy_test(u,Nx,Ny,dy)
        du1 = BySy_beta(u,Nx,Ny,y_BySy)
        du2 = VOLtoFACE(du1,1,Nx,Ny)
        #du2 = VOLtoFACE_beta(du1,1,N,Ny,yv2f1)
        #du3 = alpha1*Hyinv_test(du2,Nx,Ny,dy)  #compute action of P1
        du3 = alpha1*Hyinv_beta(du2,Nx,Ny,N,y_Hyinv,hx,hy)

        #du4 = BySy_test(u,Nx,Ny,dy)
        #du4 = du1
        du5 = VOLtoFACE(du1,2,Nx,Ny)
        #du5 = VOLtoFACE_beta(du1,2,N,Ny,yv2f2)
        #du6 = alpha2*Hyinv_test(du5,Nx,Ny,dy) #compute action of P2
        du6 = alpha2*Hyinv_beta(du5,Nx,Ny,N,y_Hyinv,hx,hy)

        du7 = VOLtoFACE(u,3,Nx,Ny)
        #du7 = VOLtoFACE_beta(u,3,N,Ny,yv2f3)
        #du8 = BxSx_tran_test(du7,Nx,Ny,dx)
        du8 = BxSx_tran_beta(du7,Nx,Ny,N,y_BxSx_tran,hx,hy)
        #du9 = beta*Hxinv_test(du8,Nx,Ny,dx)
        du9 = beta*Hxinv_beta(du8,Nx,Ny,N,y_Hxinv,hx,hy)
        #du10 = VOLtoFACE(u,3,Nx,Ny)
        #du10 = du7
        du11 = alpha3*Hxinv_beta(du7,Nx,Ny,N,y_Hxinv,hx,hy) #compute action of P3

        du12 = VOLtoFACE(u,4,Nx,Ny)
        #du12 = VOLtoFACE_beta(u,4,Nx,Ny,yv2f4)
        du13 = BxSx_tran_beta(du12,Nx,Ny,N,y_Hxinv,hx,hy)
        #du14 = beta*Hxinv_test(du13,Nx,Ny,dx)
        du14 = beta*Hxinv_beta(du13,Nx,Ny,N,y_Hxinv,hx,hy)
        #du15 = VOLtoFACE(u,4,Nx,Ny)
        #du16 = alpha4*Hxinv_test(du15,Nx,Ny,dx) #compute action of P4
        du16 = alpha4*Hxinv_beta(du12,Nx,Ny,N,y_Hxinv,hx,hy)


        du0 = du_ops + du3 + du6 + du9 + du11 + du14 + du16 #Collect together

        #compute action of -Hx kron Hy:

        du17 = Hy_test(du0, Nx, Ny, dy)
	du[:] = -Hx_test(du17,Nx,Ny,dx)
end



# @unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test

N = Nx*Ny
g1 = -pi .* cos.(pi .* x)
g2 = pi .* cos.(pi .* x .+ pi)
g3 = sin.(pi .* y)
g4 = sin.(pi .+ pi .* y)

f = spzeros(Nx,Ny)
exactU = spzeros(Nx,Ny)

for i = 1:Nx
	for j = 1:Ny
		f[j,i] = -pi^2 .* sin.(pi .* x[i] + pi .* y[j]) - pi^2 .* sin.(pi .* x[i] + pi .* y[j])
		exactU[j,i] = sin.(pi .* x[i] + pi .* y[j])
	end
end

f = f[:]
exact = exactU[:]

#Construct vector b
b0 = FACEtoVOL(g1,1,Nx,Ny)
b1 = alpha1*Hyinv(b0,Nx,Ny,dy)

b2 = FACEtoVOL(g2,2,Nx,Ny)
b3 = alpha2*Hyinv(b2,Nx,Ny,dy)

b4 = FACEtoVOL(g3,3,Nx,Ny)
b5 = alpha3*Hxinv(b4,Nx,Ny,dx)
b6 = BxSx_tran(b4,Nx,Ny,dx)
b7 = beta*Hxinv(b6,Nx,Ny,dx)

b8 = FACEtoVOL(g4,4,Nx,Ny)
b9 = alpha4*Hxinv(b8,Nx,Ny,dx)
b10 = BxSx_tran(b8,Nx,Ny,dx)
b11 = beta*Hxinv(b10,Nx,Ny,dx)

bb = b1  + b3  + b5 + b7 + b9 + b11 + f

#Modify b for PD system
b12 = Hx(bb,Nx,Ny,dx)
b = -Hy(b12,Nx,Ny,dy)

D0 = LinearMap(myMAT_original!,N;ismutating=true)
D = LinearMap(myMAT!, N; ismutating=true)
D1 = LinearMap(myMAT_new!,N; ismutating=true)
D2 = LinearMap(myMAT_new!,N; ismutating=true)
u0 = cg(D0,b,tol=1e-14)
u = cg(D,b,tol=1e-14)
u1 = cg(D1,b,tol=1e-14)
u2 = cg(D2,b,tol=1e-14)

diff = u - exact

Hydiff = Hy(diff,Nx,Ny,dy)
HxHydiff = Hx(Hydiff,Nx,Ny,dx)

err = sqrt(diff'*HxHydiff)

@show err
