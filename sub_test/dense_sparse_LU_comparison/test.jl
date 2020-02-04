using SparseArrays
using LinearAlgebra


function rerun(n);
    test_times = 10
    t_0 = zeros(test_times);
    t_1 = zeros(test_times);
    #M = sparse(randn(n,n));
    #b1 = sparse(randn(n,n));
    #b2 = sparse(randn(n,n));
    for i in range(1,step=1,stop=test_times)
        #b1 = sparse(randn(n));
        #b2 = sparse(randn(n));
        M = sparse(randn(n,n));
        b1 = randn(n);
        b2 = randn(n);
        t_0[i] = @elapsed M\b1;
        t_1[i] = @elapsed M\b2;
    end
    return t_0, t_1
end

function rerun_with_ldiv(n);
    test_times = 10
    t_2 = zeros(test_times);
    t_3 = zeros(test_times);
    #M = sparse(randn(n,n));
    #b1 = sparse(randn(n,n));
    #b2 = sparse(randn(n,n));
    for i in range(1,step=1,stop=test_times)
        #b1 = sparse(randn(n));
        #b2 = sparse(randn(n));
        M = sparse(randn(n,n));
        b1 = randn(n);
        b2 = randn(n);
        x1 = similar(b1);
        x2 = similar(b2);
        lu_M = lu(M);
        t_2[i] = @elapsed ldiv!(x1,lu_M,b1);
        t_3[i] = @elapsed M\b1;
        @assert x1 == M\b1
    end
    return t_2, t_3
end

n = 1000;
t_0,t_1 = rerun(n)
t_2,t_3 = rerun_with_ldiv(n)


@show t_0
@show t_1
@show t_2
@show t_3

