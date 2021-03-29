using GenericSchur;
using BivMatFun;
import BivMatFun.mp;

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

    return convert(Matrix{ComplexF64}, FA.Z * Y * FB.Z');
  end
end