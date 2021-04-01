using LinearAlgebra;
using Printf;
using Random;
using BenchmarkTools;

("." in LOAD_PATH) ? [] : push!(LOAD_PATH, ".");

using BivMatFun;

Random.seed!(123);

function run_test()
    n = 512;
    
    A = randn(n, n);
    # A = A / opnorm(A);
    A = A + one(A);
    
    B = randn(n, n);
    # B = B / opnorm(B);
    B = B + one(B);
    
    C = randn(n, n);
    
    A = convert(Matrix{ComplexF64}, A);
    B = convert(Matrix{ComplexF64}, B);
    C = convert(Matrix{ComplexF64}, C);
    
    f = (x,y,i,j) -> 1 ./ (x-y) ./ abs(x+y);
    # f = (x,y,i,j) -> 1 ./ (x-y) ./ abs(x+y);
    # f = (x,y,i,j) -> x - y;
    
    method = BivMatFun.DiagNoHp
    
    # Profile.clear();
    @btime fun2m($f, $A, $B, $C, delta = $0.05, method = $method, parallel = $true);
    # @time F2, info2 = fun2m(f, A, B, C, delta = 0.05, method = method, parallel = true);
    # @btime fun2m($f, $A, $B, $C, 0.05, $method);
    # Juno.profiler();
    
    @btime BivMatFun.diag_fun($f, $A, $B, $C);
    
    # @btime BivMatFun.diag_fun($f, $A, $B, $C);
    res = norm(X - F);
    # res = 0.0;
    
    @printf("Residue: %e, nblocks = %d, %d\n", res, info.nblocksA, info.nblocksB);
end

run_test();
