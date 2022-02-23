include("../diagonal_sbp.jl")


function matrix_free_D2_p4(idata,odata,Nx,Ny,hx,hy)
    ## D2 in 1D
    # SBP coefficients
    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    odata[1] = (bd[1,1] * idata[1] + bd[1,2] * idata[2] + bd[1,3]*idata[3] + bd[1,4]*idata[4] + bd[1,5]*idata[5])/(hx^2)
    
    odata[2] = (bd[2,1] * idata[1] + bd[2,2] * idata[2] + bd[2,3] * idata[3] + bd[2,4] * idata[4] + bd[2,5] * idata[5]) / (hx^2)
    odata[3] = (bd[3,1] * idata[1] + bd[3,2] * idata[2] + bd[3,3] * idata[3] + bd[3,4] * idata[4] + bd[3,5] * idata[5]) / (hx^2)
    odata[4] = (bd[4,1] * idata[1] + bd[4,2] * idata[2] + bd[4,3] * idata[3] + bd[4,4] * idata[4] + bd[4,5] * idata[5] + bd[4,6] * idata[6]) / (hx^2)

    for i in 5:Ny-4
        odata[i] = (d[1] * idata[i-2] + d[2] * idata[i-1] + d[3] * idata[i] + d[4] * idata[i+1] + d[5] * idata[i+2]) / (hx^2)
    end

    odata[end-3] = (bd[4,1]*idata[end] + bd[4,2]*idata[end-1] + bd[4,3]*idata[end-2] + bd[4,4]*idata[end-3] + bd[4,5] * idata[end-4] + bd[4,6] * idata[end-5]) / (hx^2)
    odata[end-2] = (bd[3,1]*idata[end] + bd[3,2]*idata[end-1] + bd[3,3]*idata[end-2] + bd[3,4]*idata[end-3] + bd[3,5] * idata[end-4]) / (hx^2)
    odata[end-1] = (bd[2,1]*idata[end] + bd[2,2]*idata[end-1] + bd[2,3]*idata[end-2] + bd[2,4]*idata[end-3] + bd[2,5] * idata[end-4]) / (hx^2)
    odata[end] = (bd[1,1]*idata[end] + bd[1,2]*idata[end-1] + bd[1,3]*idata[end-2] + bd[1,4]*idata[end-3] + bd[1,5]*idata[end-4]) / (hx^2)
    nothing
end


function matrix_free_D2x_p4(idata,odata,Nx,Ny,hx,hy)
    ## D2x in 2D
    # SBP coefficients
    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    

end


function matrix_free_D2_p4_split(idata,odata,Nx,Ny,hx,hy)
     ## D2 in 1D
    # SBP coefficients
    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];


    odata .= 0

    for i in 5:Nx-4
        for j in 5:Ny-4
            odata[i,j] = (d[1] * idata[i,j-2] + d[2] * idata[i,j-1] + d[3] * idata[i,j] + d[4] * idata[i,j+1] + d[5] * idata[i,j+2]
                + d[1] * idata[i-2,j] + d[2] * idata[i-1,j] + d[3] * idata[i,j] + d[4] * idata[i+1,j] + d[5] * idata[i+2,j]
            ) / hx^2
        end
    end

    nothing
end




function matrix_free_N(idata,odata,Nx,Ny,hx,hy;beta=1)
    ## D2 in 1D
    # SBP coefficients

    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    # D2 calculation first

    for i in 1:4
        for j in 1:4
                odata[i,j] = - (bd[j,1] * idata[i,1] + bd[j,2] * idata[i,2] + bd[j,3]*idata[i,3] + bd[j,4]*idata[i,4] + bd[j,5]*idata[i,5] + bd[j,6] * idata[i,6]
        + bd[i,1] * idata[1,j] + bd[i,2] * idata[2,j] + bd[i,3] * idata[3,j] + bd[i,4]*idata[4,j] + bd[i,5]*idata[5,j] + bd[i,6] * idata[6,j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
        end
    end

    for i in 1:4
        for j in 1:4
                odata[i,end+1-j] = - (bd[j,1] * idata[i,end] + bd[j,2] * idata[i,end-1] + bd[j,3]*idata[i,end-2] + bd[j,4]*idata[i,end-3] + bd[j,5]*idata[i,end-4] + bd[j,6] * idata[i,end-5]
        + bd[i,1] * idata[1,end+1-j] + bd[i,2] * idata[2,end+1-j] + bd[i,3] * idata[3,end+1-j] + bd[i,4]*idata[4,end+1-j] + bd[i,5]*idata[5,end+1-j] + bd[i,6] * idata[6,end+1-j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
        end
    end
   
    for i in 1:4
        for j in 5:Ny-4
            odata[i,j] = - (d[1] * idata[i,j-2] + d[2] * idata[i,j-1] + d[3]*idata[i,j] + d[4]*idata[i,j+1] + d[5]*idata[i,j+2]
            + bd[i,1] * idata[1,j] + bd[i,2] * idata[2,j] + bd[i,3] * idata[3,j] + bd[i,4]*idata[4,j] + bd[i,5]*idata[5,j] + bd[i,6] * idata[6,j]) /  (bhinv[i]) # calculation for the left upper corner
        end
    end

    
    # Boundary terms
    # SAT_W & SAT_E
    for i in 1:4
        for j in 1:1
            odata[i,j] += -(-13*idata[i,j]/bhinv[i])  # only -H_tilde * tau_W*HI_x*E_W
            odata[i,end] += -(-13*idata[i,end])/bhinv[i] # only -H_tilde * tau_E*HI_x*E_E
        end
        for j in 1:4
            odata[i,j] += -(beta*BS[j]/bhinv[i] * idata[i,1]) # only -H_tilde * beta*HI_x*BS_x'*E_W
            odata[i,end+1-j] += -(beta*BS[j]/bhinv[i] * idata[i,end])  # only -H_tilde * beta*HI_x*BS_x'*E_E
        end
    end

    # SAT_N

    # for j in 1:4
    #     for i in 1:4
    #         odata[1,j] += -(tau_N*BS[i]/bhinv[j] * idata[i,j]) # tau_N*HI_y*E_N*BS_y
    #     end
    # end

    # for j in 5:Ny-4
    #     for i in 1:4
    #         odata[1,j] += -(tau_N*BS[i] * idata[i,j])
    #     end
    # end

    # for j in 1:4
    #     for i in 1:4
    #         odata[1,end+1-j] += -(tau_N*BS[i]/bhinv[j] * idata[i,end+1-j])
    #     end
    # end

    # alternative form

    for i in 1:4
        for j in 1:4
            odata[1,j] += -(tau_N*BS[i]/bhinv[j] * idata[i,j]) # tau_N*HI_y*E_N*BS_y
            odata[1,end+1-j] += -(tau_N*BS[i]/bhinv[j] * idata[i,end+1-j])
        end
        for j in 5:Ny-4
            odata[1,j] += -(tau_N*BS[i]*idata[i,j])
        end
    end


    nothing
end



function matrix_free_N_pseudo(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];


    for j in 1:4
        if j == 1
            for i in 1:4
                odata[i,j] = ( - (bd[j,1] * idata[i,1] + bd[j,2] * idata[i,2] + bd[j,3]*idata[i,3] + bd[j,4]*idata[i,4] + bd[j,5]*idata[i,5] + bd[j,6] * idata[i,6]
                + bd[i,1] * idata[1,j] + bd[i,2] * idata[2,j] + bd[i,3] * idata[3,j] + bd[i,4]*idata[4,j] + bd[i,5]*idata[5,j] + bd[i,6] * idata[6,j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
                + -(-13*idata[i,j]/bhinv[i])  # only -H_tilde * tau_W*HI_x*E_W
                + -(beta*BS[j]/bhinv[i] * idata[i,1]) # only -H_tilde * beta*HI_x*BS_x'*E_W
                # + -(tau_N*BS[i]/bhinv[j] * idata[i,j]) # tau_N*HI_y*E_N*BS_y
                )

                odata[i,end+1-j] = ( - (bd[j,1] * idata[i,end] + bd[j,2] * idata[i,end-1] + bd[j,3]*idata[i,end-2] + bd[j,4]*idata[i,end-3] + bd[j,5]*idata[i,end-4] + bd[j,6] * idata[i,end-5]
                + bd[i,1] * idata[1,end+1-j] + bd[i,2] * idata[2,end+1-j] + bd[i,3] * idata[3,end+1-j] + bd[i,4]*idata[4,end+1-j] + bd[i,5]*idata[5,end+1-j] + bd[i,6] * idata[6,end+1-j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
                + -(-13*idata[i,end])/bhinv[i] 
                + -(beta*BS[j]/bhinv[i] * idata[i,end])
                # + -(tau_N*BS[i]/bhinv[j] * idata[i,end+1-j])
                )
            end
        end
    end
end



function matrix_free_N_D2(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 1:4
        for j in 1:4
                odata[i,j] = - (bd[j,1] * idata[i,1] + bd[j,2] * idata[i,2] + bd[j,3]*idata[i,3] + bd[j,4]*idata[i,4] + bd[j,5]*idata[i,5] + bd[j,6] * idata[i,6]
        + bd[i,1] * idata[1,j] + bd[i,2] * idata[2,j] + bd[i,3] * idata[3,j] + bd[i,4]*idata[4,j] + bd[i,5]*idata[5,j] + bd[i,6] * idata[6,j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
        end
    end

    for i in 1:4
        for j in 1:4
                odata[i,end+1-j] = - (bd[j,1] * idata[i,end] + bd[j,2] * idata[i,end-1] + bd[j,3]*idata[i,end-2] + bd[j,4]*idata[i,end-3] + bd[j,5]*idata[i,end-4] + bd[j,6] * idata[i,end-5]
        + bd[i,1] * idata[1,end+1-j] + bd[i,2] * idata[2,end+1-j] + bd[i,3] * idata[3,end+1-j] + bd[i,4]*idata[4,end+1-j] + bd[i,5]*idata[5,end+1-j] + bd[i,6] * idata[6,end+1-j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
        end
    end
   
    for i in 1:4
        for j in 5:Nx-4
            odata[i,j] = - (d[1] * idata[i,j-2] + d[2] * idata[i,j-1] + d[3]*idata[i,j] + d[4]*idata[i,j+1] + d[5]*idata[i,j+2]
            + bd[i,1] * idata[1,j] + bd[i,2] * idata[2,j] + bd[i,3] * idata[3,j] + bd[i,4]*idata[4,j] + bd[i,5]*idata[5,j] + bd[i,6] * idata[6,j]) /  (bhinv[i]) # calculation for the left upper corner
        end
    end

    nothing
end

function matrix_free_S_D2(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 1:4
        for j in 1:4
                odata[end+1-i,j] = - (bd[j,1] * idata[end+1-i,1] + bd[j,2] * idata[end+1-i,2] + bd[j,3]*idata[end+1-i,3] + bd[j,4]*idata[end+1-i,4] + bd[j,5]*idata[end+1-i,5] + bd[j,6] * idata[end+1-i,6]
        + bd[i,1] * idata[end,j] + bd[i,2] * idata[end-1,j] + bd[i,3] * idata[end-2,j] + bd[i,4]*idata[end-3,j] + bd[i,5]*idata[end-4,j] + bd[i,6] * idata[end-5,j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
        end
    end

    for i in 1:4
        for j in 1:4
                odata[end+1-i,end+1-j] = - (bd[j,1] * idata[end+1-i,end] + bd[j,2] * idata[end+1-i,end-1] + bd[j,3]*idata[end+1-i,end-2] + bd[j,4]*idata[end+1-i,end-3] + bd[j,5]*idata[end+1-i,end-4] + bd[j,6] * idata[end+1-i,end-5]
        + bd[i,1] * idata[end,end+1-j] + bd[i,2] * idata[end-1,end+1-j] + bd[i,3] * idata[end-2,end+1-j] + bd[i,4]*idata[end-3,end+1-j] + bd[i,5]*idata[end-4,end+1-j] + bd[i,6] * idata[end-5,end+1-j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
        end
    end
   
    for i in 1:4
        for j in 5:Nx-4
            odata[end+1-i,j] = - (d[1] * idata[end+1-i,j-2] + d[2] * idata[end+1-i,j-1] + d[3]*idata[end+1-i,j] + d[4]*idata[end+1-i,j+1] + d[5]*idata[end+1-i,j+2]
            + bd[i,1] * idata[end,j] + bd[i,2] * idata[end-1,j] + bd[i,3] * idata[end-2,j] + bd[i,4]*idata[end-3,j] + bd[i,5]*idata[end-4,j] + bd[i,6] * idata[end-5,j]) /  (bhinv[i]) # calculation for the left upper corner
        end
    end

    nothing
end

function matrix_free_W_D2(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 5:Ny-4
        for j in 1:4
            odata[i,j] = - (d[1]*idata[i-2,j] + d[2] * idata[i-1,j] + d[3] * idata[i,j] + d[4] * idata[i+1,j] + d[5]*idata[i+2,j]
                + bd[j,1]*idata[i,1] + bd[j,2]*idata[i,2] + bd[j,3] * idata[i,3] + bd[j,4]*idata[i,4] + bd[j,5]*idata[i,5] + bd[j,6]*idata[i,6]) / bhinv[j]
        end
    end
end

function matrix_free_E_D2(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 5:Ny-4
        for j in 1:4
            odata[i,end+1-j] = - (d[1]*idata[i-2,end+1-j] + d[2] * idata[i-1,end+1-j] + d[3] * idata[i,end+1-j] + d[4] * idata[i+1,end+1-j] + d[5]*idata[i+2,end+1-j]
                + bd[j,1]*idata[i,end] + bd[j,2]*idata[i,end-1] + bd[j,3] * idata[i,end-2] + bd[j,4]*idata[i,end-3] + bd[j,5]*idata[i,end-4] + bd[j,6]*idata[i,end-5]) / bhinv[j]
        end
    end
end

function matrix_free_N_P(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 1:4
        for j in 1:4
            odata[1,j] += -(tau_N*BS[i]/bhinv[j] * idata[i,j]) # tau_N*HI_y*E_N*BS_y
            odata[1,end+1-j] += -(tau_N*BS[i]/bhinv[j] * idata[i,end+1-j])
        end
        for j in 5:Nx-4
            odata[1,j] += -(tau_N*BS[i]*idata[i,j])
        end
    end

    nothing
end

function matrix_free_S_P(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;
    beta = 1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 1:4
        for j in 1:4
            odata[end,j] += -(tau_N*BS[i]/bhinv[j] * idata[end+1-i,j]) # tau_N*HI_y*E_N*BS_y
            odata[end,end+1-j] += -(tau_N*BS[i]/bhinv[j] * idata[end+1-i,end+1-j])
        end
        for j in 5:Ny-4
            odata[end,j] += -(tau_N*BS[i]*idata[end+1-i,j])
        end
    end
end


function matrix_free_W_P(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;
    beta = 1

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 1:4
        odata[i,1] = -(-13*idata[i,1]/bhinv[i])
        odata[end+1-i,1] = -(-13*idata[end+1-i,1]/bhinv[i])
        for j = 1:4
            odata[i,j] += -(beta*BS[j]/bhinv[i] * idata[i,1])
            odata[end+1-i,j] += -(beta*BS[j]/bhinv[i] * idata[end+1-i,1])
        end
    end

    for i in 5:Ny-4
        odata[i,1] = -(-13*idata[i,1])
        for j in 1:4
            odata[i,j] += -(beta*BS[j] * idata[i,1])
        end
    end
    nothing
end

function matrix_free_E_P(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;
    beta = 1

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

    for i in 1:4
        odata[i,end] = -(-13*idata[i,end]/bhinv[i])
        odata[end+1-i,end] = -(-13*idata[end+1-i,end]/bhinv[i])
        for j = 1:4
            odata[i,end+1-j] += -(beta*BS[j]/bhinv[i] * idata[i,end])
            odata[end+1-i,end+1-j] += -(beta*BS[j]/bhinv[i] * idata[end+1-i,end])
        end
    end

    for i in 5:Ny-4
        odata[i,end] = -(-13*idata[i,end])
        for j in 1:4
            odata[i,end+1-j] += -(beta*BS[j] * idata[i,end])
        end
    end
    nothing
end