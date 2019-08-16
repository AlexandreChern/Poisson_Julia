using LinearAlgebra
using SparseArrays
using BenchmarkTools


n_list = [10,20,30,50,100,200]
#for k = 1:5
k = 6
n = n_list[k]

A = rand(n,n);
A = A*A'
b = rand(n);

result_1 = @benchmark A\b
x1 = A\b
result_2 = @benchmark lu(A)
F = lu(A)
result_3 = @benchmark F.U\(F.L\b[F.p,:])
x2 = F.U\(F.L\b[F.p,:])

println("Matrix dim n: ", n)
println(norm(x1-x2))
println()
display(result_1)
println()
display(result_2)
println()
display(result_3)
println()
#end
