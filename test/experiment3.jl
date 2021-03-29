using LinearAlgebra;
using Printf;
using Random;
using BenchmarkTools;
using DelimitedFiles;

using BivMatFun;
import BivMatFun.mp;

include("experiments_common.jl")

f = (x,y,i,j) -> 1 ./ (x-y) ./ sqrt(x+y);

Random.seed!(123);


function run_test()
    ntests = 7;

    # (ntest, method, f) => f ∈ (time, blocksA, blocksB, res)
    data = zeros(ntests, 3, 4);

    # Set this to false to disable timings
    check_times = true;

    for i = 1 : ntests
      n = 2^(i-1) * 64;

      A = randn(n, n) + 1im * randn(n, n)
      B = randn(n, n) + 1im * randn(n, n)
      C = randn(n, n) + 1im * randn(n, n)

      # A, B, C, FF = construct_random_benchmark(f, n)
      # @time FF = evaluate_reference_solution(f, A, B, C)

      F, info = fun2m(f, A, B, C, method = BivMatFun.Diag);

      if check_times
        thp = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Diag);
        thp = median(thp.times) / 1e9;
      else
        thp = 0.0;
      end

      F2, info2 = fun2m(f, A, B, C, method = BivMatFun.DiagNoHp);

      if check_times
        tnohp = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.DiagNoHp);
        tnohp = median(tnohp.times) / 1e9;
      else
        tnohp = 0.0;
      end

      XD = BivMatFun.diag_fun(f, A, B, C);
      if check_times
        tdiag = @benchmark BivMatFun.diag_fun($f, $A, $B, $C);
        tdiag = median(tdiag.times) / 1e9
        else
        tdiag = 0.0
      end

      # @btime BivMatFun.diag_fun($f, $A, $B, $C);
      reshp = 0.0 # norm(F - FF) / norm(FF)
      res = Float64(norm(F - F2) / norm(F)) # norm(FF - F2) / norm(FF);
      res2 = Float64(norm(F - XD) / norm(F)) # norm(XD - FF) / norm(FF)

      @printf("n = %d\n", n)
      @printf("DIAG_HP: time = %f, res = %e, nblocks = %d, %d\n", thp, reshp, info.nblocksA, info.nblocksB)
      @printf("DIAG_NOHP time = %f, Residual: %e, nblocks = %d, %d\n", tnohp, res, info2.nblocksA, info2.nblocksB);
      @printf("DIAG time = %f, Residual: %e\n", tdiag, res2);

      data[i, 1, 1:4] = [ thp, info.nblocksA, info.nblocksB, reshp ]
      data[i, 2, 1:4] = [ tnohp, info2.nblocksA, info2.nblocksB, res ]
      data[i, 3, 1:4] = [ tdiag, 0, 0, res2 ]

      writedlm("experiment3.dat", data, '\t')
    end

end

run_test();
