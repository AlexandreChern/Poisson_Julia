include("Poisson.jl")
include("matrix-free-p4.jl")
include("matrix-free-p2.jl")
include("matrix-free-p4-GPU.jl")


level = 13
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

H_D2 = reshape(-H_tilde*(D2*idata_flat),Nx,Ny)
H_A = reshape(-H_tilde*(A*idata_flat),Nx,Ny)

H_tilde_diag = diag(H_tilde)

idata_GPU = CuArray(idata)
odata_GPU = CuArray(zeros(Nx,Ny))

D2_matrix_free_p2(idata_GPU,odata_GPU)


matrix_free_N(idata,odata,Nx,Ny,hx,hy)

odata_N_D2 = zeros(Nx,Ny)
matrix_free_N_D2(idata,odata_N_D2,Nx,Ny,hx,hy)

odata_S_D2 = zeros(Nx,Ny)
matrix_free_S_D2(idata,odata_S_D2,Nx,Ny,hx,hy)

odata_W_D2 = zeros(Nx,Ny)
matrix_free_W_D2(idata,odata_W_D2,Nx,Ny,hx,hy)

odata_E_D2 = zeros(Nx,Ny)
matrix_free_E_D2(idata,odata_E_D2,Nx,Ny,hx,hy)

H_D2 - odata_N_D2 - odata_S_D2 - odata_W_D2 - odata_E_D2

idata_GPU_N = @view idata_GPU[1:6,1:end]

odata_GPU_N_D2 = CuArray(zeros(4,Ny))
matrix_free_N_D2_GPU(idata_GPU_N,odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy)
matrix_free_N_D2_GPU(idata_GPU_N,odata_GPU,coef_D,Nx,Ny,hx,hy)


odata_GPU_N_D2_v2 = CuArray(zeros(4,Ny))
matrix_free_N_D2_GPU_1D_kernel(idata_GPU_N,odata_GPU_N_D2_v2,coef_D,Nx,Ny,hx,hy)


t_GPU_N_D2 = @elapsed for _ in 1:20000
    matrix_free_N_D2_GPU(idata_GPU_N,odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy)
end

t_GPU_N_D2_v2 = @elapsed for _ in 1:20000
    matrix_free_N_D2_GPU_1D_kernel(idata_GPU_N,odata_GPU_N_D2_v2,coef_D,Nx,Ny,hx,hy)
end

# odata_N_P = zeros(Nx,Ny)
# matrix_free_N_P(idata,odata_N_P,Nx,Ny,hx,hy)

# reshape(-H_tilde*SAT_S*idata_flat,Nx,Ny)



odata_S_P = zeros(Nx,Ny)
matrix_free_S_P(idata,odata_S_P,Nx,Ny,hx,hy)
reshape(-H_tilde*SAT_N*idata_flat,Nx,Ny)


odata_W_P = zeros(Nx,Ny)
matrix_free_W_P(idata,odata_W_P,Nx,Ny,hx,hy)
reshape(-H_tilde*SAT_W*idata_flat,Nx,Ny)


odata_E_P = zeros(Nx,Ny)
matrix_free_E_P(idata,odata_E_P,Nx,Ny,hx,hy)
reshape(-H_tilde*SAT_E*idata_flat,Nx,Ny)


odata_pseudo = zeros(Nx,Ny)
matrix_free_N_pseudo(idata,odata_pseudo,Nx,Ny,hx,hy)







reshape(-H_tilde*(D2+SAT_W+SAT_E)*idata_flat,Nx,Ny)
reshape(-H_tilde*(D2+SAT_W+SAT_E+SAT_S)*idata_flat,Nx,Ny)













## Performance benchmarking

repetitions = 1000
# time_D2 = @elapsed for _ in 1:repetitions
#     odata_GPU .= 0
#     D2_matrix_free_p2(idata_GPU,odata_GPU)
# end

# through_put = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2)



time_D2_SPMV = @elapsed for _ in 1:repetitions
   D2*idata_flat
end

matrix_free_D2_p4_GPU(idata_GPU,odata_GPU)

reshape(D2*idata_flat,Nx,Ny)

time_D2_p4 = @elapsed for _ in 1:repetitions
    # odata_GPU .= 0
    matrix_free_D2_p4_GPU(idata_GPU,odata_GPU)
end

time_D2_p2 = @elapsed for _ in 1:repetitions
    # odata_GPU .= 0
    D2_matrix_free_p2(idata_GPU,odata_GPU)
end


through_put_SPMV = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_SPMV)
through_put_matrix_free_p4 = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_p4)
through_put_matrix_free_p2 = (2*Nx*Ny*8 * repetitions)/ (1024^3 * time_D2_p2)

@show through_put_SPMV
@show through_put_matrix_free_p4
@show through_put_matrix_free_p2