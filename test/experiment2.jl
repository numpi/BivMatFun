using MatrixDepot;
using BivMatFun;
using BenchmarkTools;
using Printf;
using LinearAlgebra;
using DelimitedFiles;

include("experiments_common.jl")

function run_test() 
  
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
      
      if matrix_name == "jordbloc1"
        k = min(m, n, 8);
        
        J = diagm(1 => ones(k-1)) + 0.1 * I;
        
        A = randn(m-k, m-k); A = A / opnorm(A); A = A + shift * I;
        A = cat(J, A, dims=[1,2]);
        Q = qr(randn(m, m)).Q;
        A = Q * A * Q';
        
        B = randn(n-k, n-k); B = B / opnorm(B); B = B + shift * I;
        B = cat(J, B, dims=[1,2]);
        Q = qr(randn(m, m)).Q;
        B = Q * B * Q';
        
      elseif matrix_name == "kahan2"
        k = min(min(m,n), 64);
        XA = randn(m-k, m-k)
        XA = XA / opnorm(XA) + shift * I;
        XB = randn(n-k, n-k)
        XB = XB / opnorm(XB) + shift * I;
        A = cat(matrixdepot("kahan", k), XA, dims = [1,2]);
        B = cat(matrixdepot("kahan", k), XB, dims = [1,2]);
        
      elseif matrix_name == "smoke"
        A = schur(smoke(m)).T;
        B = A;
        
      elseif matrix_name == "grcar-randn"
        A = matrixdepot("grcar", m);
        B = randn(n, n);
        
      elseif matrix_name == "lesp"
        k = min(32, min(m,n));
        XA = randn(m-k, m-k)
        XA = XA / opnorm(XA) - shift * I;
        XB = randn(n-k, n-k)
        XB = XB / opnorm(XB) - shift * I;
        A = -cat(schur(lesp(k)).T, XA, dims = [1,2]);
        B = -cat(schur(lesp(k)).T, XB, dims = [1,2]);

      elseif matrix_name == "sampling"
        k = min(32, min(m,n));

        A = matrixdepot(matrix_name, k) + I;
        B = matrixdepot(matrix_name, k) + I;

        XA = randn(m-k, m-k)
        XA = XA / opnorm(XA) + shift * I;
        XB = randn(n-k, n-k)
        XB = XB / opnorm(XB) + shift * I;

        A = cat(A, XA, dims=[1,2])
        B = cat(B, XB, dims=[1,2])
        
      else
        A = matrixdepot(matrix_name, m);
        B = matrixdepot(matrix_name, n);
      end
      
      A = complex(A);
      B = complex(B)
      
      C = randn(m, n); C = C / norm(C);
      C = complex(C)

      Y = evaluate_reference_solution(f, A, B, C)
      errest = BivMatFun.fun2m_condest(f, A, B, C) * eps();

      Y2, infom = fun2m(f, A, B, C, method = BivMatFun.Diag, bs = max(m,n));
      tdiagm = @benchmark fun2m($f, $A, $B, $C, method = $BivMatFun.Diag, bs = $max($m, $n))
      tdiagm = median(tdiagm.times) / 1e9;
            
      XD = BivMatFun.diag_fun(f, A, B, C);
      tdiag = @benchmark BivMatFun.diag_fun($f, $A, $B, $C)
      tdiag = median(tdiag.times) / 1e9;
      
      X, info = fun2m(f, A, B, C, method = BivMatFun.Diag);
      tsp = @benchmark fun2m($f, $A, $B, $C, method = BivMatFun.Diag);
      tsp = median(tsp.times) / 1e9;
      
      @printf("%s: %s: |X - Y|/|Y|: %e (nblocks = %d, %d, time = %fs, digits = %d); errest = %e\n", 
        fname, matrix_name, norm(X - Y) / norm(Y), info.nblocksA, info.nblocksB, tsp, info.digits, errest);
      @printf("%s: %s: |Y2 - Y| / |Y|: %e, time = %fs\n", fname, matrix_name, norm(Y2 - Y) / norm(Y), tdiagm)
      @printf("%s: %s: |XD - Y|/|Y|: %e, time = %fs\n", fname, matrix_name, norm(XD - Y) / norm(Y), tdiag);
      
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

      writedlm("experiment2.dat", data, '\t');
    end
    
  end
  
end

run_test();
