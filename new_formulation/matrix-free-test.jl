using BenchmarkTools
# include("diagonal_sbp.jl")
include("deriv_ops.jl")

using BenchmarkTools

function D2(idata,odata,Nx,Ny,h)
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny + j
            if (i==1) && (j==1)
                odata[global_index] = (2*idata[global_index] - 2*idata[global_index+Ny] - 2*idata[global_index+1] + idata[global_index+2*Ny] + idata[global_index + 2]) / h^2
            end

            if ((i == 1) && (j > 1) && (j < Ny))
                odata[global_index] = ( - idata[global_index] - 2*idata[global_index + Ny] + idata[global_index + 2*Ny] + idata[global_index - 1] + idata[global_index + 1]) / (h*h);
                # odata[global_index] = 0.0;
            end

            if ((i == 1) && (j == Ny))
                odata[global_index] = (2*idata[global_index] - 2*idata[global_index - 1] + idata[global_index - 2] - 2*idata[global_index + Ny] + idata[global_index + 2*Ny]) / (h*h);
                # odata[global_index] = 0.0;
            end

            if ((i > 1) && (i < Nx ) && (j == 1))
                odata[global_index] = ( - idata[global_index] - 2*idata[global_index + 1] + idata[global_index + 2] + idata[global_index - Ny] + idata[global_index + Ny]) / (h*h);
                # odata[global_index] = 2342352345;
            end

            if ((i > 1) && (i < Nx ) && (j == Ny ))
                odata[global_index] = ( - idata[global_index] - 2*idata[global_index - 1] + idata[global_index - 2] + idata[global_index - Ny] + idata[global_index + Ny]) / (h*h);
                # odata[global_index] = 2.0;
            end

            if ((i > 1) && (i < Nx ) && (j > 1) && (j < Ny))
                odata[global_index] = (idata[global_index-1] + idata[global_index+1] + idata[global_index-Ny] + idata[global_index+Ny] - 4*idata[global_index]) / (h*h);
                # odata[global_index] = (idata[global_index - 1] - 2*idata[global_index] + idata[global_index+1] ) / (h*h);
                # odata[global_index] = (idata[global_index-Ny] - 2*idata[global_index] + idata[global_index+Ny]) / (h*h);
                # odata[global_index] = 1.0;
            end

            if ((i == Nx) && (j == 1))
                odata[global_index] = (2*idata[global_index] - 2*idata[global_index - Ny] - 2*idata[global_index + 1] + idata[global_index - 2*Ny] + idata[global_index + 2]) / (h*h);
                # odata[global_index] = 0.0;
            end
            
            if ((i == Nx) && (j > 1) && (j < Ny))
                odata[global_index] = ( - idata[global_index] - 2*idata[global_index - Ny] + idata[global_index - 2*Ny] + idata[global_index - 1] + idata[global_index + 1]) / (h*h);
                # odata[global_index] = 0.0;
            end
            
            if ((i == Nx) && (j == Ny))
                odata[global_index] = (2*idata[global_index] - 2*idata[global_index - 1] + idata[global_index - 2] - 2*idata[global_index - Ny] + idata[global_index - 2*Ny]) / (h*h);
                # odata[global_index] = 0.0;
            end
        end
    end
end


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

function D2_pseudo(idata,odata,Nx,Ny,h)
    # Need to think about how to write GPU pseudo code
    i = 1:Nx
    j = 1:Ny
    global_index = (i .- 1) .* Ny .+ j

    # idata_index_x = (i-1)*Ny + j + test_mod(i,Nx) * Nx
    # idata_index_y = (i-1)*Ny + j + test_mod(j,Ny)

    idata_index_x = (i .- 1) *Ny .+ j .+ (div.(2*Nx .-i .-3,Nx-2) .- 1) .* Nx
    idata_index_y = (i .- 1) *Ny .+ j .+ div.(2*Ny .- j .-3, Ny-2) .- 1



    odata[global_index] .= ( (idata[idata_index_x .- Ny] .- 2*idata[idata_index_x] .+ idata[idata_index_x .+ Ny]) 
                            .+ (idata[idata_index_y .- 1] .- 2*idata[idata_index_y] .+ idata[idata_index_y .+ 1]) ) ./ h^2
end



function Boundary_Conditions(idata,odata,Nx,Ny,h)

end


function test_mod(x,N)
    return div(2*N-x-3,N-2) - 1
end


function test(Nx,Ny)
    h = 1/(Nx-1)
    idata = randn(Nx*Ny)
    odata = zeros(Nx*Ny)
    odata_2 = zeros(Nx*Ny)

    output = D2x(idata,Nx,Ny,h) + D2y(idata,Nx,Ny,h)
    D2(idata,odata,Nx,Ny,h)
    D2_test(idata,odata_2,Nx,Ny,h)

    
    @assert odata ≈ output 
    @assert odata_2 ≈ output

    result_1 =  @benchmark D2x($(idata),$(Nx),$(Ny),$(h)) + D2y($(idata),$(Nx),$(Ny),$(h))
    result_2 =  @benchmark D2($(idata),$(odata),$(Nx),$(Ny),$(h))
    result_3 =  @benchmark D2_test($(idata),$(odata_2),$(Nx),$(Ny),$(h))

    println("Result 1")
    io = IOBuffer()
    show(io,"text/plain",result_1)
    s = String(take!(io))
    println(s)

    println("Result 1")
    io = IOBuffer()
    show(io,"text/plain",result_2)
    s = String(take!(io))
    println(s)
    # show(io,"text/plain",result_2)
    # show(io,"text/plain",result_2)

    println("Result 3")
    io = IOBuffer()
    show(io,"text/plain",result_3)
    s = String(take!(io))
    println(s)

    # print(result_2)
end