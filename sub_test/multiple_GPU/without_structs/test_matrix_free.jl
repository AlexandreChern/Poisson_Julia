include("laplacian.jl")
include("assembling_matrix.jl")
using Random
Random.seed!(0)

level = 4 # 2^3 +1 points in each direction
idata_cpu = randn(2^level+1,2^level+1)

(A,D2,b,H_tilde,Nx,Ny) = Assembling_matrix(level)
hx = 1/(Nx-1)
hy = 1/(Ny-1)

coef_p2 = coef_GPU_sbp_sat_v4(CuArray([2. 0]), # The 0 here is only to make them identical arrays
    CuArray([1. -2 1]),
    CuArray([1. -2 1]),
    CuArray([3/2 -2 1/2]),
    CuArray([Nx Ny 1/(Nx-1) 1/(Ny-1)]),
    CuArray([13/(1/(Nx-1)) 13/(1/(Nx-1)) 1 1 -1]))  
coef_p2_D = cudaconvert(coef_p2)

# odata_H_tilde_D2 = reshape(H_tilde*-D2*idata_cpu[:],Nx,Ny)
odata_H_tilde_A = reshape(A*idata_cpu[:],Nx,Ny)


odata_cpu = reshape(A*idata_cpu[:],Nx,Ny)

idata_GPU = CuArray(idata_cpu)
odata_GPU = CuArray(zeros(size(idata_GPU)))

idata_GPUs, odata_GPUs, odata_boundaries_GPUs, num_blocks = allocate_GPU_arrays(idata_GPU;num_blocks=3)



# odata_boundars_GPUs = [
#     CuArray(zeros(Nx,3)) CuArray(zeros(1,size(idata_GPUs[1])[2])) CuArray(zeros(1,size(idata_GPUs[1])[2])),
#     CuArray(zeros(Nx,3)) CuArray(zeros(1,size(idata_GPUs[end])[2])) CuArray(zeros(1,size(idata_GPUs[end])[2]))
# ] 

# laplacian_GPU(idata_GPU,odata_GPU)



laplacian_GPU_v2(idata_GPU,odata_GPU,Nx,Ny,hx,hy)
laplacian_GPU_v2(idata_GPUs[1],odata_GPUs[1],Nx,Ny,hx,hy)
laplacian_GPU_v2(idata_GPUs[2],odata_GPUs[2],Nx,Ny,hx,hy)
laplacian_GPU_v2(idata_GPUs[3],odata_GPUs[3],Nx,Ny,hx,hy)

odata_GPU_1_1 = CuArray(zeros(Nx,3))
odata_GPU_3_3 = CuArray(zeros(Nx,3))
odata_GPU_2_1 = CuArray(zeros(3,Ny))
odata_GPU_4_1 = CuArray(zeros(3,Ny))

boundary(idata_GPU,odata_GPU_1_1,Nx,Ny,hx,hy;orientation=1,type=1)
boundary(idata_GPU,odata_GPU_3_3,Nx,Ny,hx,hy;orientation=3,type=3)
boundary(idata_GPU,odata_GPU_2_1,Nx,Ny,hx,hy;orientation=2,type=1)
boundary(idata_GPU,odata_GPU_4_1,Nx,Ny,hx,hy;orientation=4,type=1)

boundary(idata_GPUs[1],odata_GPUs[1],Nx,Ny,hx,hy;orientation=4,type=1)

odata_GPU[1,:] .+= odata_GPU_2_1[1,:]
odata_GPU[:,1:3] .+= odata_GPU_1_1[:,1:3]
odata_GPU[:,end-2:end] .+= odata_GPU_3_3[:,1:3]


boundary(idata_GPUs[1],odata_boundaries_GPUs[find_boundaries_GPUs(1,1,num_blocks)],Nx,Ny,hx,hy;orientation=1,type=1)
boundary(idata_GPUs[1],odata_boundaries_GPUs[find_boundaries_GPUs(2,1,num_blocks)],Nx,Ny,hx,hy;orientation=2,type=1)
boundary(idata_GPUs[1],odata_boundaries_GPUs[find_boundaries_GPUs(4,1,num_blocks)],Nx,Ny,hx,hy;orientation=4,type=1)

boundary(idata_GPUs[2],odata_boundaries_GPUs[find_boundaries_GPUs(2,2,num_blocks)],Nx,Ny,hx,hy;orientation=2,type=2)
boundary(idata_GPUs[2],odata_boundaries_GPUs[find_boundaries_GPUs(4,2,num_blocks)],Nx,Ny,hx,hy;orientation=4,type=2)

boundary(idata_GPUs[3],odata_boundaries_GPUs[find_boundaries_GPUs(3,3,num_blocks)],Nx,Ny,hx,hy;orientation=3,type=3)
boundary(idata_GPUs[3],odata_boundaries_GPUs[find_boundaries_GPUs(2,3,num_blocks)],Nx,Ny,hx,hy;orientation=2,type=3)
boundary(idata_GPUs[3],odata_boundaries_GPUs[find_boundaries_GPUs(4,3,num_blocks)],Nx,Ny,hx,hy;orientation=4,type=3)

let 
    level = 4
    i = j = level
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    N_x = Integer(m_list[i]);
    N_y = Integer(n_list[j]);

    Nx = N_x + 1;
    Ny = N_y + 1;

    (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j,h_list_x,h_list_y);
    # Penalty Parameters
    tau_E = 13/hx;
    tau_W = 13/hx;
    tau_N = 1;
    tau_S = 1;

    beta = -1;

    # Forming SAT terms

    ## Formulation 1
    SAT_W = tau_W*HI_x*E_W + beta*HI_x*BS_x'*E_W;
    SAT_E = tau_E*HI_x*E_E + beta*HI_x*BS_x'*E_E;

    # SAT_S = tau_S*HI_y*E_S*D1_y
    # SAT_N = tau_N*HI_y*E_N*D1_y

    SAT_S = tau_S*HI_y*E_S*BS_y;
    SAT_N = tau_N*HI_y*E_N*BS_y;

    SAT_W_r = tau_W*HI_x*E_W*e_W + beta*HI_x*BS_x'*E_W*e_W;
    SAT_E_r = tau_E*HI_x*E_E*e_E + beta*HI_x*BS_x'*E_E*e_E;
    SAT_S_r = tau_S*HI_y*E_S*e_S;
    SAT_N_r = tau_N*HI_y*E_N*e_N;


    (alpha1,alpha2,alpha3,alpha4,beta) = (tau_N,tau_S,tau_W,tau_E,beta);


    g_W = sin.(π*y);
    g_E = -sin.(π*y);
    g_S = -π*cos.(π*x);
    # g_N = -π*cos.(π*x)
    g_N = π*cos.(π*x .+ π);
end

odata_H_tilde_SAT_W =  reshape(H_tilde * SAT_W * idata_cpu[:],Nx,Ny)
odata_H_tilde_SAT_E =  reshape(H_tilde * SAT_E * idata_cpu[:],Nx,Ny)
odata_H_tilde_SAT_S =  reshape(H_tilde * SAT_S * idata_cpu[:],Nx,Ny)
odata_H_tilde_SAT_N =  reshape(H_tilde * SAT_N * idata_cpu[:],Nx,Ny)

odata_cpu

# odata_H_tilde_SAT_W + odata_H_tilde_D2

odata_H_tilde_D2 + odata_H_tilde_SAT_W + odata_H_tilde_SAT_E + odata_H_tilde_SAT_S + odata_H_tilde_SAT_N - odata_H_tilde_A