using BivMatFun;
using BenchmarkTools;
using Printf;
using LinearAlgebra;
using DelimitedFiles;
using MAT;

include("experiments_common.jl")

function run_test() 

  success = true;
  
  matrices = [ "jordbloc1", "grcar", "smoke", "kahan2", "lesp", "sampling", "grcar-randn" ];
  
  m = 64; 
  n = m;
  a = .5;
  
  fncts = [
    Dict("name" => "sqrt", "f" => (x,y,i,j) -> sqrt(x+y)), 
    Dict("name" => "invsqrt", "f" => (x,y,i,j)  -> 1 ./ sqrt(x+y)),
    Dict("name" => "phi1", "f" => (x,y,i,j) -> expm1(x+y) ./ (x+y)), 
    Dict("name" => "expa", "f" => (x,y,i,j) -> exp((x+y).^a))
  ]

  shift = 1.00;
  
  lesp = n -> convert(Matrix{ComplexF64}, diagm(-1 => 1 ./ (2:n), 0 => -(5 : 2 : 2*n+3), 1 => 2 : n));
  smoke = n -> diagm(0 => exp.(2im * pi * (1 : n) ./ n), 1 => complex(ones(n-1)), -n+1 => [ 1.0 + 0im ]);
  
  # err_fun2m, err_diag, err_diagm, time_fun2m, time_diag, time_diagm, nblocksA, nblocksB, d_fun2m, d_fundiag, errest
  data = zeros(length(fncts), length(matrices), 11);
  
  for i = 1 : length(fncts)
    for j = 1 : length(matrices);
      matrix_name = matrices[j];
      
      fname = fncts[i]["name"];
      f = fncts[i]["f"];

      vars = matread("$(data_directory)/experiment2_$(matrix_name)_$(fname).mat")
      A = vars["A"]; B = vars["B"]; C = vars["C"]; Y = vars["Y"]; errest = vars["errest"]

      Y2, infom = fun2m(f, A, B, C, method = BivMatFun.Diag, bs = max(m,n));

      if ! is_ci_test()
        tdiagm = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Diag, bs = $max($m, $n))
        tdiagm = median(tdiagm.times) / 1e9;
      else
        tdiagm = 0.0
      end
            
      XD = BivMatFun.diag_fun(f, A, B, C);
      if ! is_ci_test()
        tdiag = @benchmark BivMatFun.diag_fun($f, $A, $B, $C)
        tdiag = median(tdiag.times) / 1e9;
      else
        tdiag = 0.0
      end
      
      X, info = fun2m(f, A, B, C, method = BivMatFun.Diag);
      if ! is_ci_test()
        tsp = @benchmark fun2m($f, $A, $B, $C, method = BivMatFun.Diag);
        tsp = median(tsp.times) / 1e9;
      else
        tsp = 0.0
      end
      
      #% err_fun2m, err_diag, time_fun2m, time_diag, time_diagm, nblocksA, nblocksB, d_fun2m, d_fundiagm, errest
      data[i, j, 1] = norm(X - Y) / norm(Y);
      data[i, j, 2] = norm(XD - Y) / norm(Y);
      data[i, j, 3] = norm(Y2 - Y) / norm(Y);
      data[i, j, 4] = tsp;
      data[i, j, 5] = tdiag;
      data[i, j, 6] = tdiagm;
      data[i, j, 7] = info.nblocksA;
      data[i, j, 8] = info.nblocksB;
      data[i, j, 9] = info.digits;
      data[i, j, 10] = infom.digits;
      data[i, j, 11] = errest;

      # In these tests, it is expected that fun2m with Diag should work in all cases, and that 
      # the method using Taylor may fail. Hence, we only check if the residue is sufficiently
      # for the case with Diag; we allow some tolerance, because the error estimate is rather 
      # rough.
      success &= data[i, j, 1] < errest * 1e2;

      # writedlm("experiment2.dat", data, '\t');

      @printf("%s: %s: |X - Y|/|Y|: %e (nblocks = %d, %d, time = %fs, digits = %d); errest = %e\n", 
        fname, matrix_name, norm(X - Y) / norm(Y), info.nblocksA, info.nblocksB, tsp, info.digits, errest);
      @printf("%s: %s: |Y2 - Y| / |Y|: %e, time = %fs\n", fname, matrix_name, norm(Y2 - Y) / norm(Y), tdiagm)
      @printf("%s: %s: |XD - Y|/|Y|: %e, time = %fs\n", fname, matrix_name, norm(XD - Y) / norm(Y), tdiag);
    end

    # We format the data in a way that is easier to handle and ready to include 
    # in the paper TeX files. 
    writedlm("exp2-$i.dat", data[i,:,:], '\t')
  end

  return success;
  
end

run_test();
