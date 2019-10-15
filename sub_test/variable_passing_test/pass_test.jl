using Parameters
using BenchmarkTools
using Profile

@with_kw struct st
    n = 10;
    a = randn(n)
    b = randn(n)
    c = randn(n)
end

st

@with_kw struct st_new
    n = 100
    a = randn(n)
    b = randn(n)
    c = randn(n)
    d = randn(n)
end

function f1(a,b,c)
    c = a+b
    return c
end

function f1_test(a,b,c)
    c .= a .+ b
    return c
end

function f2(a,b,c)
    c = a-b
    return c
end

function f2_test(a,b,c)
    c .= a .- b
    return c
end

function f2_beta(a,b,d)
    d .= a .- b
    return d
end

t1 = st_new()
@unpack a,b,c,d = t1


n = 100
u = randn(n)
v = randn(n)
w = randn(n)
function g(t1,u,v,w)
    @unpack a,b,c,d = t1
    u = f1(a,b,c)
    v = f2(a,b,c)
    #println(u)
    #println(v)
    return w .= u .* v
end

function g_v2(u,v,w)
    u = f1(a,b,c)
    v = f2(a,b,c)
    return w .= u .* v
end

function g_test(t1,u,v,w)
    @unpack a,b,c,d = t1
    u = f1_test(a,b,c)
    v = f2_test(a,b,c)
    #println(u)
    #println(v)
    return w .= u .* v
end

function g_beta(t1,u,v,w)
    @unpack a,b,c,d = t1
    u = f1_test(a,b,c)
    v = f2_test(a,b,d)
    #println(u)
    #println(v)
    return w .= u .* v
end




function g1(u,v,w)
    u = f1(a,b,c)
    v = f2(a,b,c)
    return w = u .* v
end

function g2(u,v,w,a,b,c)
    u = f1(a,b,c)
    v = f2(a,b,c)
    return w = u.* v
end

function g2_test(u,v,w,a,b,c)
    u = f1_test(a,b,c)
    v = f2_test(a,b,c)
    return w = u.*v
end

function g3_test(u,v,w,a,b,c)
    u = f1_test(a,b,c)
    v = f2_test(a,b,c)
    return w .= u .* v
end


function g3(u,v,w,a,b,c,f1,f2)
    u = f1(a,b,c)
    v = f2(a,b,c)
    return w = u.*v
end

function g4(u,v,w,a,b,c,f1,f2)
    u = f1(a,b,c)
    v = f2(a,b,c)
end

g(t1,u,v,w)

@benchmark g(t1,u,v,w)
