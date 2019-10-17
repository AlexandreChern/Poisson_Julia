n = 100
a = randn(n)
b = randn(n)
c = randn(n)
d = randn(n)

function f1(a,b,c)
    c = a + b
    return c
end

function f2(a,b,c)
    c = a - b
    return c
end

function f1_test(a,b,c)
    c .= a .+ b
    return c
end

function f2_test(a,b,c)
    c .= a .- b
    return c
end


u = similar(a);
v = similar(a);
w = similar(a);

function g1(u,v,w,a,b,c)
    u = f1(a,b,c)
    v = f2(a,b,c)
    w .= u .* v
    return w
end


function g2(u,v,w,a,b,c)
    u = f1_test(a,b,c)
    v = f2_test(a,b,c)
    w .= u .* v
    return w
end

g1(u,v,w,a,b,c)

g2(u,v,w,a,b,c)

g1(u,v,w,a,b,c) == g2(u,v,w,a,b,c)



f1(a,b,c) == f1_test(a,b,c)
f2(a,b,c) == f2_test(a,b,c)

f1(a,b,c) + f2(a,b,c) == f1_test(a,b,c) + f2_test(a,b,c)
