using LinearAlgebra

function enorm(A::UniformScaling{T}, g, tmp) where T
  g' * g
end
function enorm(M::Vector{T}, g, tmp) where T
    @. tmp = M * g
    g' * tmp
end
function cg(u0, b, A; tol=1e-8, MaxIter=100, M = I)
  cg(u0, b, (y,x)->mul!(y, A, x); tol=tol, MaxIter=MaxIter, M=M)
end
function cg(u0, b, A::Function; tol=1e-8, MaxIter=100, M = I)
  u = copy(u0)
  w = similar(u0)
  d = similar(u0)
  g = similar(u0)
  tmp = similar(u0)
  k = cg!(u, w, d, g, tmp, b, A; tol=tol, MaxIter=MaxIter, M=M)
  (u, k)
end
function cg!(u, w, d, g, tmp, b, A::Function; tol=1e-8, MaxIter=100, M = I)

  A(w, u)
  @. d = b - w
  @. g = -d

  gkTgk = g' * g

  err = enorm(M, g, tmp)
  nmx = enorm(M, u, tmp)
  tol2 = tol^2
  for k = 1:MaxIter
    if err < tol2 * (1 + nmx)
      return k
    end
    A(w, d)

    alpha = gkTgk / (d' * w)

    @. u = u + alpha * d

    @. g = g + alpha * w

    gk1Tgk1 = g' * g

    beta = gk1Tgk1 / gkTgk

    @. d = -g + beta * d

    gkTgk = gk1Tgk1

    err = enorm(M, g, tmp)
    nmx = enorm(M, u, tmp)
    # err = g' * g
    # nmx = u' * u
  end
  if err < tol2 * (1 + nmx)
    return MaxIter
  end
  -MaxIter
end

function testcg(N, tol = 1e-6)
  (Q,R) = qr(rand(N,N))

  A = Q * Diagonal(rand(N)) * Q'

  A = (A+A')/2

  x = rand(N)

  b = A*x

  function f!(y, x)
    y .= A*x
  end

  (x0,k) = cg(b, b, f!, tol=tol, MaxIter=N, M=I)

  @show ((norm(b - A*x) < tol, k))

  nothing
end
