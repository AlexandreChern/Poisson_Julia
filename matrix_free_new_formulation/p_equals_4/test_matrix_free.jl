include("Poisson.jl")
include("matrix-free-p4.jl")



level = 4
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

(D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j,h_list_x,h_list_y;SBPp=4);


idata = randn(Nx,Ny)
odata = zeros(Nx,Ny)

idata_flat = idata[:]

matrix_free_D2_p4_split(idata,odata,Nx,Ny,hx,hy)
odata_spmv = D2 * idata_flat
odata_spmv_reshaped = reshape(odata_spmv,Nx,Ny)