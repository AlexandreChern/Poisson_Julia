function D2_test(idata,odata,Nx,Ny,h)
    for i = 1:Nx
        for j = 1:Ny
            global_index_x = (i-1)*Ny+j

            # idata_index_x = (i-1)*Ny + j + test_mod(i,Nx) * Nx
            # idata_index_y = (i-1)*Ny + j + test_mod(j,Ny)

            idata_index_x = (i-1)*Ny + j + (div(2*Nx-i-3,Nx-2) - 1) * Nx
            idata_index_y = (i-1)*Ny + j + div(2*Ny-j-3,Ny-2) - 1



            odata[global_index_x] = ( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny]) 
                                    + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) ) / h^2
        end
    end
end

function Boundary_Conditions(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    odata .= 0
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny + j
            if j == Ny # N
                odata[global_index] += alpha1 * 2 * (1.5*idata[global_index] - 2*idata[global_index-1] + 0.5*idata[global_index-2]) / h^2
                # odata[global_index] = alpha1 * 2 * (1.5*idata[global_index] - 2*idata[global_index-1] + 0.5*idata[global_index-2]) / h^2
            end

            if j == 1 # S
                odata[global_index] += alpha2 * 2 *(1.5*idata[global_index] - 2*idata[global_index+1] + 0.5*idata[global_index+2]) / h^2
            end

            if i == Nx # E
                odata[global_index] += beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h
                odata[global_index-Ny] += beta * 2 * (-1*idata[global_index]) / h^2
                odata[global_index-2*Ny] += beta * (0.5*idata[global_index]) / h^2
            end

            if i == 1 # W
                odata[global_index] += beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha3 * 2* (idata[global_index]) / h
                odata[global_index+Ny] += beta * 2 * (-1*idata[global_index]) / h^2
                odata[global_index+2*Ny] += beta * (0.5*idata[global_index]) / h^2
            end
        end
    end
end

function H_tilde_matrix_free(idata,odata,Nx,Ny,h)
    for i in 1:Nx
        for j in 1:Ny
            global_index = (i-1)*Ny+j
            odata[global_index] = idata[global_index] * h^2
            if i == 1
                odata[global_index] /= 2
            end
            if j == 1
                odata[global_index] /= 2
            end
            if i == Nx
                odata[global_index] /= 2
            end
            if j == Ny
                odata[global_index] /= 2
            end
        end
    end
end


function matrix_free_A(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    odata1 = spzeros(Nx*Ny)
    odata2 = spzeros(Nx*Ny)
    D2_test(idata,odata1,Nx,Ny,h)
    Boundary_Conditions(idata,odata2,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    # odata3 = odata1 .+ odata2
    # return odata
    odata .= odata1 .+ odata2
    H_tilde_matrix_free(odata,odata,Nx,Ny,h)
    # return odata3
end
