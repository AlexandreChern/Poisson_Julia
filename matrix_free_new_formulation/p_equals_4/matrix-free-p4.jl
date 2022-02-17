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


function matrix_free_N(idata,odata,Nx,Ny,hx,hy)
    
    nothing
end