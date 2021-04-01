#
# This is a script that re-generates the test cases used in experiment1.jl 
# and experiment2.jl, that are save in MAT files. Having these tests already
# pre-computed saves an enourmous time while testing. 
#

using MAT
using BivMatFun
using LinearAlgebra
using SpecialFunctions

include("experiments_common.jl")

println("Data directory: ", data_directory)

function generate_test_1()
    # This data needs to be kept in sync with experiment1.jl
    a = 1/2;
    f = (x,y,i,j) -> (x+y).^(-a-i-j) * gamma(a+i+j) ./ gamma(a) * (-1)^(i+j);
    ntests = 5;
    
    for i = 1 : ntests
        n = 32 * i;
        
        println("Generating experiment1, n = ", n, " random")
        
        A, B, C, Y = construct_random_benchmark(f, n)
        errest1 = BivMatFun.fun2m_condest(f, A, B, C) * eps();
        
        matwrite("$(data_directory)/experiment1_$(n)_random.mat", Dict(
        "A" => A, 
        "B" => B,
        "C" => C,
        "Y" => Y,
        "errest" => errest1
        ));
        
        println("Generating experiment1, n = ", n, " grcar-random")
        
        A = grcar(n) + I
        
        A = convert(Matrix{ComplexF64}, A)
        B = convert(Matrix{ComplexF64}, B)
        C = convert(Matrix{ComplexF64}, C)
        
        Y2 = evaluate_reference_solution(f, A, B, C)
        errest2 = BivMatFun.fun2m_condest(f, A, B, C) * eps();
        
        matwrite("$(data_directory)/experiment1_$(n)_grcar.mat", Dict(
        "A" => A, 
        "B" => B,
        "C" => C,
        "Y" => Y2,
        "errest" => errest2
        ));
    end
end

function generate_test_2()
    
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
    
    for i = 1 : length(fncts)
        fname = fncts[i]["name"];
        f = fncts[i]["f"];
        
        for j = 1 : length(matrices);
            matrix_name = matrices[j];
            
            println("Generating experiment2, $(matrix_name), $(fname)")
            
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
                A = cat(kahan(k), XA, dims = [1,2]);
                B = cat(kahan(k), XB, dims = [1,2]);
                
            elseif matrix_name == "smoke"
                A = schur(smoke(m)).T;
                B = A;
                
            elseif matrix_name == "grcar-randn"
                A = grcar(m);
                B = randn(n, n);
                
            elseif matrix_name == "grcar"
                A = grcar(m);
                B = grcar(n);
                
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
                
                A = sampling(k) + I;
                B = sampling(k) + I;
                
                XA = randn(m-k, m-k)
                XA = XA / opnorm(XA) + shift * I;
                XB = randn(n-k, n-k)
                XB = XB / opnorm(XB) + shift * I;
                
                A = cat(A, XA, dims=[1,2])
                B = cat(B, XB, dims=[1,2])
            end
            
            A = complex(A);
            B = complex(B)
            
            C = randn(m, n); C = C / norm(C);
            C = complex(C)
            
            Y = evaluate_reference_solution(f, A, B, C)
            errest = BivMatFun.fun2m_condest(f, A, B, C) * eps();
            
            matwrite("$(data_directory)/experiment2_$(matrix_name)_$(fname).mat", Dict(
            "A" => A,
            "B" => B,
            "C" => C,
            "Y" => Y,
            "errest" => errest
            ))
        end
    end
end

# generate_test_1()
generate_test_2()