include("diagonal_sbp.jl")
include("deriv_ops_new.jl")
include("split_matrix_free_dev.jl")
include("split_matrix_free.jl")


using LinearAlgebra
using SparseArrays
using Plots
using CUDA
using IterativeSolvers
using BenchmarkTools
using MAT



function matrix_prolongation(idata)
    (Nx,Ny) = size(idata)
    odata_Nx = 2*Nx-1
    odata_Ny = 2*Ny-1
    odata = zeros(odata_Nx,odata_Ny)
    for i in 1:odata_Nx
        for j in 1:odata_Ny
            if i % 2 == 1
                if j % 2 == 1
                    odata[i,j] = idata[div(i,2)+1,div(j,2)+1]
                else 
                    odata[i,j] = (idata[div(i,2)+1,div(j,2)] +  idata[div(i,2)+1,div(j,2)+1]) / 2
                end
            else
                if j % 2 == 1
                    odata[i,j] = (idata[div(i,2),div(j,2)+1] +  idata[div(i,2)+1,div(j,2)+1]) / 2
                else 
                    odata[i,j] = (idata[div(i,2),div(j,2)] +  idata[div(i,2)+1,div(j,2)] + idata[div(i,2),div(j,2) + 1] +  idata[div(i,2)+1,div(j,2) + 1]) / 4
                end
            end
        end
    end
    return odata
end


function matrix_restriction(idata)
    (Nx,Ny) = size(idata)
    odata_Nx = div(Nx+1,2)
    odata_Ny = div(Ny+1,2)
    odata = zeros(odata_Nx,odata_Ny)
    for i in 1:odata_Nx
        for j in 1:odata_Ny
        end
    end
end