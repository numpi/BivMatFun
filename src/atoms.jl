using LinearAlgebra;
using GenericSchur;

function diag_fun!(fun, A::Array{T, 1}, B::Array{T, 1}, C::Array{T, 2}) where T
    
    (m, n) = size(C)
    
    for j = 1 : n
        for i = 1 : m
            C[i,j] = fun(A[i], B[j], 0, 0) .* C[i,j];
        end
    end
    
    return C;
end

function diag_fun(fun, A::Array{T, 1}, B::Array{T, 1}, C::Array{T, 2}) where T
    F = copy(C);
    diag_fun!(fun, A, B, F)
end

function isdiagonal(A::Matrix)
    (m, n) = size(A)
    for i = 1 : m
        for j = 1 : n
            if i != j && A[i,j] != 0
                return false
            end
        end
    end
    
    return true
end

function diag_fun(fun, A::Matrix, B::Matrix, C::Matrix)
    F = similar(C);
    
    isdiagA = isdiagonal(A);
    isdiagB = isdiagonal(B);
    
    # Computes fun{A, B}(C) when at least one among A and B is diagonal
    if isdiagA && isdiagB
        F .= C;
        diag_fun!(fun, diag(A), diag(B), C)
        return F;
    end
    
    if isdiagA
        dA = diag(A);
        for j = 1:size(A, 1)
            # g = @(y, k) fun(dA(j), y, 0, k);
            g = (y,k) -> fun(dA[j], y, 0, k);
            # funB = @(y, k) arrayfun(g, y, k * ones(size(y)));
            funB = (y, k) -> map(g, y, k * ones(size(y)));
            F[j, :] = C[j, :] * funm(B, funB);
        end
        return F;
    end
    
    if isdiagB
        dB = diag(B);
        for j = 1:size(B, 1)
            # g = @(x, k) fun(x, dB(j), k, 0);
            g = (x, k) -> fun(x, dB(j), k, 0);
            # funA = @(x, k) arrayfun(g, x, k * ones(size(x)));
            funA = (x, k) -> map(g, x, k * ones(size(x)));
            F[:, j] = funm(A, funA) * C(:, j);
        end
        return F;
    end
    
    ta = Threads.@spawn eigen(A);	
    tb = Threads.@spawn eigen(B);
    
    A, VA = fetch(ta)
    B, VB = fetch(tb)
    
    F = VA * diag_fun(fun, A, B, VA \ C * VB) / VB;
end


#----------------------------------- Auxiliary functions-----------------------------------
function fun2m_atom(f, A::Matrix, B::Matrix, C::Matrix, treeA::BivMatTree, treeB::BivMatTree, meth::Algorithm, min_digits::Int64, user_function = nothing)
    # [F, digits] =
    #FUN2M_ATOM  Bivariate function of triangular coefficients with nearly constant diagonals.
    #   F = FUNM_ATOM(A, B, C, FUN,  MAXTERMS)
    #   applies the function FUN at the upper triangular matrices A,B on C
    #   where A and B have nearly constant diagonals.
    #   A bivariate Taylor series is used, taking MAXTERMS^2 terms.
    #   The function represented by FUN must have a Taylor series with an
    #   infinite radius of convergence.
    #   FUN(X,Y,K,H) must return the (K,H)-th derivative of
    #   the function represented by FUN evaluated at (X,Y).
    #   N_TERMS^2 is the number of terms taken in the Taylor series.
    
    digits = 0;
    
    nA = size(A, 1); nB = size(B, 1);
    
    if meth == Taylor
        F, digits = taylor_eval(f, A, B, C, eps());
    elseif meth == Diag
        uh = eps() / treeA.kV / treeB.kV;
        d_uh = convert(Int64, max(ceil(log10(1 / uh)), min_digits));
        
        F, digits = trim_diagpertub(treeA.V, treeA.D, treeB.V, treeB.D, C, f, d_uh);
    elseif meth == DiagNoHp # diagonalization without high precision
        VA = treeA.V; DA = treeA.D;
        VB = treeB.V; DB = treeB.D;
        F = VA \ C * VB;
        diag_fun!(f, DA, DB, F);
        F = VA * F / VB;
    elseif meth == User # handle function provided by the user
        F = user_function(A, B, C);
    end
    
    return F, digits;
end

function trim_diagpertub(VA, DA, VB, DB, C, fun, d_uh)
    F = copy(C)
    F, d_uh = trim_diagpertub!(VA, DA, VB, DB, F, fun, d_uh)
end

function trim_diagpertub!(VA, DA, VB, DB, C, fun, d_uh)
    #TRIM_DIAGPERTUB Compute general matrix function of a triangular matrix.
    #   TRIM_DIAGPERTUB(TA, TB, C, fun) evaluates the function_handle fun at the
    #   triangular matrices TA and TB.
    #   The algorithm computes the function represented by fun
    #   using randomized approximate
    #   diagonalization with a diagonal perturbation.
    #   d_uh returns equivalent decimal digits of the
    #   possibly higher than u^2 precision.
    
    if d_uh <= 18
        VA = UpperTriangular(convert(Matrix{ComplexF64}, VA))
        VB = UpperTriangular(convert(Matrix{ComplexF64}, VB))    
        C  = convert(Matrix{ComplexF64}, C)
        DA = convert(Array{ComplexF64}, DA)
        DB = convert(Array{ComplexF64}, DB)
        
        C = VA \ C * VB;  
        diag_fun!(fun, DA, DB, C)
        
        C = VA * C / VB  
    else
        C = with_digits(d_uh) do
            VA = UpperTriangular(mp(VA, d_uh))
            VB = UpperTriangular(mp(VB, d_uh))
            DA = mp(DA, d_uh)
            DB = mp(DB, d_uh)
            C = mp(C, d_uh)
            
            C = VA \ C * VB;  
            diag_fun!(fun, DA, DB, C)
            
            C = VA * C / VB  
            
            C = convert(Matrix{ComplexF64}, C);
            
            return C;
        end
    end
    
    return C, d_uh
end

function fun2m_taylor(fun, x, y)
    # FUN2M_TAYLOR
    
    rx = maximum(abs(x));
    ry = maximum(abs(y));
    
    # Start by computing a degree 2 Taylor expansion
    C = zeros(3);
    
    for i = 1 : 3
        for j = 1 : 3
            C[i,j] = fun(0, 0, i-1, j-1) / factorial(i-1) / factorial(j-1);
        end
    end
    
    can_stop = false;
    
    dx = 2;
    dy = 2;
    
    tx = rx^dx;
    ty = ry^dy;
    
    vx = [ 1, rx, rx.^2 ];
    vy = [ 1, ry, ry.^2 ];
    
    fx = 1 / 2;
    fy = 1 / 2;
    
    tol = eps();
    
    while !can_stop
        can_stop = true;
        
        res_x = norm( (C[end, :] .* vy) * tx, Inf);
        res_y = norm( (C[:, end] .* vx) * ty, Inf);
        
        if res_x > tol * maximum(abs(C(:)))
            dx = dx + 1;
            C = [ C ; zeros(1, size(C, 2)) ];
            
            fx = fx / dx;
            
            for i = 1 : size(C, 2)
                C[end, i] = fun(0, 0, size(C,1)-1, i-1) * fx / factorial(i-1);
            end
            
            tx = tx * rx;
            vx = [ vx  vx[end] * rx ];
            can_stop = false;
        end
        
        if res_y > tol * maximum(abs(C))
            dy = dy + 1;
            C = [ C zeros(size(C, 1), 1) ];
            
            fy = fy / dy;
            
            for i = 1 : size(C, 1)
                C[i, end] = fun(0, 0, i-1, size(C,2)-1) * fy / factorial(i-1);
            end
            
            ty = ty * ry;
            vy = [ vy vy[end]*ry ];
            can_stop = false;
        end
        
        if can_stop
            # FIXME: We may want to check the maximum of the derivative over
            # the rectangle [-rx, rx] x [-ry, ry], to see if we can really stop
            C = C[1:end-1, 1:end-1];
        end
        
        return C
    end
    
end
