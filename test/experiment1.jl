using LinearAlgebra;
using BivMatFun;
using SpecialFunctions;
using Printf;
using BenchmarkTools;
using DelimitedFiles;
using MAT;

include("experiments_common.jl");

function run_test(use_grcar = false)

  success = true;

  a = 1/2;
  f = (x,y,i,j) -> (x+y).^(-a-i-j) * gamma(a+i+j) ./ gamma(a) * (-1)^(i+j);

  ntests = 5;

  # n, errD, errT, nblocksDA, nblocksDB, nblocksTA, nblocksTB, digitsD, digitsT, errest, timeD, timeT
  data = zeros(2*ntests, 12);

  for i = 1 : ntests
    n = 32 * i;

    vars = matread("$(data_directory)/experiment1_$(n)_random.mat")
    A = vars["A"]; B = vars["B"]; C = vars["C"]; Y = vars["Y"]; errest1 = vars["errest"]

    @printf("n = %d, A,B,C random\n", n);

    XD, infod = fun2m(f, A, B, C, method = BivMatFun.Diag);
    if ! is_ci_test()
      tdiag = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Diag);
      tdiag = median(tdiag.times) / 1e9
    else
      tdiag = 0.0
    end

    @printf("  Diag: err = %e, nblocks = %d %d, max digits = %d, time = %f, errest = %e\n", norm(XD - Y) / norm(Y), infod.nblocksA, infod.nblocksB, infod.digits, tdiag, errest1);  

    XT, infot = fun2m(f, A, B, C, method = BivMatFun.Taylor);

    if ! is_ci_test()
      tt = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Taylor);
      tt = median(tt.times) / 1e9;
    else
      tt = 0.0
    end

    @printf("  Taylor: err = %e, nblocks = %d %d, max deg = %d, time = %f, errest = %e\n", norm(XT - Y) / norm(Y), infot.nblocksA, infot.nblocksB, infot.digits, tt, errest1);

    success &= norm(XD - Y) / norm(Y) < errest1 * 100;
    success &= norm(XT - Y) / norm(Y) < errest1 * 100;

    # GRCAR
    vars = matread("$(data_directory)/experiment1_$(n)_grcar.mat")
    A = vars["A"]; B = vars["B"]; C = vars["C"]; Y2 = vars["Y"]; errest2 = vars["errest"]

    @printf("n = %d, A grcar, B,C random\n", n);

    XD2, infod2 = fun2m(f, A, B, C, method = BivMatFun.Diag);

    if ! is_ci_test()
      tdiag2 = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Diag);
      tdiag2 = median(tdiag2.times) / 1e9;
    else
      tdiag2 = 0.0
    end

    @printf("  Diag: err = %e, nblocks = %d %d, max digits = %d, time = %f, errest = %e\n", Float64(norm(XD2 - Y2) / norm(Y2)), infod2.nblocksA, infod2.nblocksB, infod2.digits, tdiag2, errest2);  

    BivMatFun.warnings(false);
    XT2, infot2 = fun2m(f, A, B, C, method = BivMatFun.Taylor);
    if ! is_ci_test()
      tt2 = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Taylor);
      tt2 = median(tt2.times) / 1e9;
    else
      tt2 = 0.0
    end

    BivMatFun.warnings(true);

    @printf("  Taylor: err = %e, nblocks = %d %d, max deg = %d, time = %f, errest = %e\n", Float64(norm(XT2 - Y2) / norm(Y2)), infot2.nblocksA, infot2.nblocksB, infot2.digits, tt2, errest2);  

    # For the GRCAR matrix, which involves a-posteriori re-blocking due to a 
    # highly non-normal eigenvector matrix, we expect Taylor to fail, hence
    # we do not check it. 
    success &= norm(XD2 - Y2) / norm(Y) < errest2 * 100;

    #
    # % n, errD, errT, nblocksDA, nblocksDB, nblocksTA, nblocksTB, digitsD, digitsT, errest, timeD, timeT
    data[2*i-1, 1] = n;
    data[2*i-1, 2] = norm(XD - Y) / norm(Y);
    data[2*i-1, 3] = norm(XT - Y) / norm(Y);
    data[2*i-1, 4] = infod.nblocksA;
    data[2*i-1, 5] = infod.nblocksB;
    data[2*i-1, 6] = infot.nblocksA;
    data[2*i-1, 7] = infot.nblocksB;
    data[2*i-1, 8] = infod.digits;
    data[2*i-1, 9] = infot.digits;
    data[2*i-1, 10] = errest1;
    data[2*i-1, 11] = tdiag;
    data[2*i-1, 12] = tt;
    #
    data[2*i, 1] = n;
    data[2*i, 2] = norm(XD2 - Y2) / norm(Y2);
    data[2*i, 3] = norm(XT2 - Y2) / norm(Y2);
    data[2*i, 4] = infod2.nblocksA;
    data[2*i, 5] = infod2.nblocksB;
    data[2*i, 6] = infot2.nblocksA;
    data[2*i, 7] = infot2.nblocksB;
    data[2*i, 8] = infod2.digits;
    data[2*i, 9] = infot2.digits;
    data[2*i, 10] = errest2;
    data[2*i, 11] = tdiag2;
    data[2*i, 12] = tt2;

    writedlm("exp1.dat", data, '\t');
  end

  return success;

end

run_test(true);
