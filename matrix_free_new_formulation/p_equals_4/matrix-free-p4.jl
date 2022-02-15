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
    
    odata[end] = (bd[end,end]*idata[end] + bd[end,end-1]*idata[end-1] + bd[end,end-2]*idata[end-2] + bd[end,end-3]*idata[end-3] + bd[end,end-4]*idata[end-4] + bd[end,end-5]*idata[end-5]) / (hx^2)

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