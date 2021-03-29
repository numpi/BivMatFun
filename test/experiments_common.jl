using GenericSchur;
using BivMatFun;
import BivMatFun.mp;

function kahan(::Type{T}, m::Integer, n::Integer, theta, pert) where T
  theta = convert(T, theta)
  pert = convert(T, pert)
  s = sin(theta)
  c = cos(theta)
  dim = min(m,n)
  A = zeros(T, m, n)
  for i = 1:m, j = 1:n
      i > dim ? A[i,j] = zero(T) :
      i > j   ? A[i,j] = zero(T) :
      i==j    ? A[i,j] = s^(i-1)+pert*eps(T)*(m-i+1) : A[i,j] = -c*s^(i-1)
  end
  return A
end
kahan(::Type{T}, n::Integer, theta, pert) where T = kahan(T, n, n, theta, pert)
kahan(::Type{T}, n::Integer) where T = kahan(T, n, n, 1.2, 25.)
kahan(args...) = kahan(Float64, args...)
kahan(::Type, args...) = throw(MethodError(kahan, Tuple(args)))

function sampling(::Type{T}, x::Vector) where T
  n = length(x)
  A = zeros(T, n, n)
  for j = 1:n, i = 1:n
      if i != j
          A[i,j] = x[i] / (x[i] - x[j])
      end
  end
  d = sum(A, dims=2)
  A = A + diagm(0 => d[:])
  return A
end
function grcar(::Type{T}, n::Integer, k::Integer = 3) where T
  # Compute grcar matrix
  G = tril(triu(ones(T, n,n)), min(k, n-1)) - diagm(-1 => ones(T, n-1))
  return G
end
grcar(args...) = grcar(Float64, args...)
grcar(::Type, args...) = throw(MethodError(grcar, Tuple(args)))

function grcar(::Type{T}, n::Integer, k::Integer = 3) where T
  # Compute grcar matrix
  G = tril(triu(ones(T, n,n)), min(k, n-1)) - diagm(-1 => ones(T, n-1))
  return G
end
grcar(args...) = grcar(Float64, args...)
grcar(::Type, args...) = throw(MethodError(grcar, Tuple(args)))

function construct_random_benchmark(f, n)
  d = 64;

  VA = mp(randn(n, n).+ 1im * randn(n, n), d);
  DA = mp(1.0 .+ rand(n) .+ 1im * randn(n), d)
  VB = mp(randn(n, n).+ 1im * randn(n, n), d);
  DB = mp(1.0 .+ rand(n).+ 1im * randn(n), d)
  C  = mp(randn(n, n).+ 1im * randn(n, n), d)

  A = convert(Matrix{ComplexF64}, VA * diagm(DA) / VA);
  B = convert(Matrix{ComplexF64}, VB * diagm(DB) / VB);

  FF = convert(Matrix{ComplexF64}, VA * BivMatFun.diag_fun(f, DA, DB, C) / VB)
  C = convert(Matrix{ComplexF64}, VA * C / VB)

  return A, B, C, FF
end

function evaluate_reference_solution(f, A, B, C)
  dgts = 128

  return BivMatFun.with_digits(dgts) do
    Y2 = similar(C)

    A = mp(A, dgts);
    B = mp(B, dgts);
    C = mp(C, dgts);
    
    FA = schur(A);
    FB = schur(B);
    
    C = FA.Z' * C * FB.Z;
    
    # size of the perturbation, make sure it's sufficiently far given the 
    # current precision level. 
    ep = 10.0^(-dgts / 2.0);
    
    TA = FA.T + diagm(randn(size(FA.T, 1))) * ep * norm(FA.T, Inf);
    TB = FB.T + diagm(randn(size(FB.T, 1))) * ep * norm(FB.T, Inf);
    Y = BivMatFun.diag_fun(f, TA, TB, C);

    return FA.Z * Y * FB.Z';
  end
end