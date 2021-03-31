"""
    estimate_condeig(V)

Estimate the condition number of an upper triangular matrix V; the matrix 
V is expected to be either a `Matrix{ComplexF64}` or `Matrix{Complex{BigFloat}}`. 

The condition number is first estimated converting the matrix in standard floating 
point; if the estimate is not accurate enough, it is replaced with the condition number 
of |V|; if the latter is too loose, it is computed in higher precision. 
"""
function estimate_condeig(V::UpperTriangular)

  # Compute a floating point version of V
  VF = convert(Matrix{ComplexF64}, V)
  p = 1

  nF = opnorm(VF, p)
  kV = nF * opnorm(inv(VF), p)

  if kV <= 1e-2 / eps()
    return kV
  end

  VV = -abs.(VF)
  VV = VV - diagm(2 * diag(VV))
  kV2 = opnorm(inv(VV), p) * nF

  if kV2 <= kV * 1.0e4
    return kV2
  end

  kV3 = opnorm(VF, p) * 
    opnorm(convert(Matrix{ComplexF64}, inv(V)), p)

  return kV3
end

"""
    build_eigendecomp_tree!(T::BivMatTree, maxkV::Float64, A::Union{Matrix{ComplexF64}, Matrix{Float64}}, meth::Algorithm, use_mp = true)

This recursive routine is used to construct the eigenvector matrices related to the atomic blocks of the 
matrix A (i.e., the diagonal blocks corresponding to the selected partitioning).
"""
function build_eigendecomp_tree!(T::BivMatTree, maxkV::Float64, A::Union{Matrix{ComplexF64}, Matrix{Float64}}, meth::Algorithm, use_mp = true)	
	if isa(T.A11, BivMatTreeTail)
		uh = eps() / T.kV / maxkV;
		d_uh = convert(Int64, max(32, ceil(log10(1 / uh))));
		
      if use_mp
		ok = false
		while !ok
				with_digits(d_uh) do
						ok = true
						T1 = mp(T.Ttilde, d_uh)

						# FIXME: Need to handle the 2x2 blocks in the real Schur form
						(T.D, T.V) = eigentri(T1);
						T.kV = estimate_condeig(T.V);

						uh = eps() / T.kV / maxkV;
						d_uh_new = convert(Int64, max(32, ceil(log10(1 / uh))));

						if d_uh_new > d_uh
								d_uh = d_uh_new
								ok = false
						end
				end
		end
      else
        (T.D, T.V) = eigentri(T.Ttilde);
        T.kV = cond(T.V)
		  end
	else
		build_eigendecomp_tree!(T.A11, maxkV, A, meth, use_mp);
		build_eigendecomp_tree!(T.A22, maxkV, A, meth, use_mp);
	end
end

"""
    build_partition_tree!(T, A, ind, strategy, method, delta = 0.1, bs = 1, mergeblocks = true)

Recursive function constructing the partition tree for a matrix A, according to the given parameters for 
separating the eigenvalues. 
"""
function build_partition_tree!(T, A, ind, strategy, method, delta = 0.1, bs = 1, mergeblocks = true)
	
	l = length(ind);
	
	maxkV = 1.0;
	
	if l == 1
		T.ind = ind[1];
		nblocks = 1;
		
		if method == Diag
			(T.Ttilde, T.kV) = condeigvec(A[T.ind, T.ind]);
			maxkV = T.kV;
		else
			T.Ttilde = A[T.ind, T.ind];
		end
		
		T.A11 = BivMatTreeTail();
		T.A22 = BivMatTreeTail();
		
		return nblocks, maxkV;
	elseif sum(map(length, ind)) <= bs
		T.ind = reduce(vcat, ind);
		
		nblocks = 1;
		
		if method == Diag
			(T.Ttilde, T.kV) = condeigvec(A[T.ind, T.ind]);
			maxkV = T.kV;
		else
			T.Ttilde = A[T.ind, T.ind];
		end
		
		T.A11 = BivMatTreeTail();
		T.A22 = BivMatTreeTail();
		
		return nblocks, maxkV;
	else
		T.ind = reduce(vcat, ind);
		
		if strategy == "balanced"
			v = cumsum(map(length, ind));
			_, ii = findmin(map(abs, v .- v[end]/2));
			ind1 = ind[1:ii];
			ind2 = ind[ii + 1: end];
		elseif strategy == "single"
			v = cumsum(map(length, ind[end:-1:1]));
			i = max(1, sum(v .< bs));
			ind1 = ind[1:end - i];
			ind2 = ind[end-i+1:end];
		else
			error("Unsupported strategy argument")
		end
		
		i1 = ind1[1][1]; i2 = ind1[end][end];
		j1 = ind2[1][1]; j2 = ind2[end][end];
		
		T.A11 = newBivMatTree();
		T.A22 = newBivMatTree();
		
		nblocks1, kV1 = build_partition_tree!(T.A11, A, ind1, strategy, method, delta, bs, mergeblocks);
		nblocks2, kV2 = build_partition_tree!(T.A22, A, ind2, strategy, method, delta, bs, mergeblocks);
		
		maxkV = max(kV1, kV2);
		
		nblocks = nblocks1 + nblocks2;
    
    
		T.sylv = sylvester(A[i1:i2, i1:i2], -A[j1:j2, j1:j2], -A[i1:i2, j1:j2])
		
		ninv = norm(T.sylv) / norm(A[i1:i2, j1:j2]);
		
		# If the Sylvester equation is
		# too ill-conditioned we merge the corresponding blocks
		if ninv > 10 / delta
			if mergeblocks
				T.ind = reduce(vcat, ind);
				T.sylv = nothing;
				T.A11 = BivMatTreeTail();
				T.A22 = BivMatTreeTail();
				nblocks = 1;
				if method == Diag
					T.Ttilde, T.kV = condeigvec(A[T.ind, T.ind]);
					maxkV = T.kV;
				else
					T.Ttilde = A[T.ind, T.ind]
				end
			else
				warn("Ill-conditioned Sylvester equation: results may be inaccurate");
			end
		end
		
		return nblocks, maxkV
	end
end

function fun2m_preprocessing(A, deltaA, method, bs, strategy)

  if istriu(A)
    TA = A; UA = one(A);
    diagTA = diag(TA);
                      FA = Schur(TA, UA, diag(TA))
  else
    FA = schur(A);
    TA = FA.T; UA = FA.Z;
    diagTA = diag(TA);
  end
		
  # Determine reordering of the Schur forms into block forms.
  ordA = blocking(TA, deltaA);
  ordA, indA = swapping(ordA);  # Gives the blocking.
  ordA::Array{LinearAlgebra.BlasInt,1} = maximum(ordA) .- ordA .+ 1;    # Since ORDSCHUR puts highest index top left.
  ordschur!(FA, ordA)
  TA = FA.T; UA = FA.Z;
	
	mergeblocks = method != Taylor;
		
	treeA = newBivMatTree();

	nblocksA, maxkVA = build_partition_tree!(treeA, TA, indA, strategy, method, deltaA, bs, mergeblocks);
	
  if method == Diag || method == DiagNoHp
    use_mp = method == Diag
    build_eigendecomp_tree!(treeA, maxkVA, A, method, use_mp);
  end

  return UA, TA, treeA, nblocksA
end

"""
    fun2m(f, A, B, C; <keyword arguments>)

Evaluate the bivariate matrix functions X = f{A, B^T}(C); the function f should have 
the prototype:

    f = (x,y,i,j) -> f^(i,j)(x,y),

which denotes the i-th derivative of f with respect to x, the j-th derivative with respect to y, evaluate at a point (x, y).

Note that, unless `BivMatFun.Taylor` is used (see below), this function is always called with `i = j = 0`. 

# Arguments
Several optional arguments can be specified to control the behavior of `fun2m`:
- `delta::Float64 = 0.05`: Minimum separation between the eigenvalues, used to determined the partitioning. 
- `method::Algorithm = BivMatFun.Diag`: Algorithm used to evaluate the function. The default is `BivMatFun.Diag`, which implements a mixed precision perturb and diagonalize approach. Other options are `BivMatFun.Taylor` (using Taylor expansions at the atomic blocks), or `BivMatFun.DiagNoHp`, which uses perturb-and-diagonalize avoiding multiprecision. The last option is provided for benchmarks only, and is not expected to be accurate. The only option that needs to evaluate the derivatives is `BivMatFun.Taylor`. For all other cases the function `f` is always called with `i = j = 0`. 
- `bs::Union{Nothing, Int64} = nothing`: Minimum block-size to use in the partitioning. Specify `nothing` to automatically choose an optimal value based on the chosen `method`.
- `min_digits::Int64 = 0`: Minimum number of digits to use in multiprecision computations. Use `0` (default) to not pose any constraint. 
- `strategy::String = "balanced"`: strategy to use for splitting the blocks. Valid options are "balanced" or "single". 
- `user_function = nothing`: if different from nothing, this function is used to evaluate the atomic blocks. It needs to have the prototype (A, B, C) -> X. 
- `parallel::Bool = true`: Use the multithreaded implementation.
"""
function fun2m(f, A, B, C; delta::Float64 = 0.05, method::Algorithm = Diag, bs::Union{Nothing, Int64} = nothing, 
               min_digits::Int64 = 0, strategy::String = "balanced", user_function = nothing, parallel::Bool = true)
	
	info = Fun2MInfo(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing);
	
	# Default parameters.
	if bs === nothing
		if method == Taylor
			bs = 1;
		elseif method == Diag
			bs = 4;
		elseif method == DiagNoHp
			bs = 4;
		else
			bs = 1;
		end
	end
	
	# Checking input dimensions
	m = size(A, 1);
	n = size(B, 1);
	
	if m != size(A, 2)
		error("A must be square")
	end
	
	if n != size(B, 2)
		error("B must be square")
	end
	
	if m != size(C, 1) || n != size(C, 2)
		error("Incompatible dimensions");
	end
	
	# When at least one among A and B is a scalar
	if m == 1 || n == 1
		F = scalar_fun(f, A, B, C);
		return;
	end

  t1 = Threads.@spawn fun2m_preprocessing(A, delta, method, bs, strategy)
  t2 = Threads.@spawn fun2m_preprocessing(B, delta, method, bs, strategy)
	
  UA, TA, treeA, info.nblocksA = fetch(t1)
  UB, TB, treeB, info.nblocksB = fetch(t2)
	
	info.time_fun2mric = @timed begin
		
		C = UA' * C * UB;
		F = similar(C)
		info.digits = fun2m_ric!(f, F, TA, TB, C, treeA, treeB, method, min_digits, 0, user_function, parallel);
		F = UA * F * UB';
		
	end # End of time fun2mric
	
	if isreal(A) && isreal(B) && isreal(C) && norm(imag(F),1) <= 10*n*eps()*norm(F,1)
		F = real(F);
	end
	
	return F, info;
end

# ----------------------------------Recursive part---------------------------------------------------
function fun2m_ric!(f, F, A, B, C, treeA::BivMatTree, treeB::BivMatTree, method::Algorithm, min_digits::Int64, level::Int64, user_function, parallel::Bool)
	# Computes the bivariate matrix functions on triangularized and blocked
	# coefficients via recursion and Sylvester solution
	# Base cases
	
	digits = 0;
	
	(m, n) = size(C);
	
	if isa(treeA, BivMatTreeTail) || isa(treeB, BivMatTreeTail)
		F .= 0;
		return digits;
	end
	
	iA = treeA.ind; iB = treeB.ind;
	
	if isa(treeA.A11, BivMatTreeTail) && isa(treeB.A11, BivMatTreeTail) # atomic block
    F[:], digits = fun2m_atom(f, A[iA, iA], B[iB, iB], C, treeA, treeB, method, min_digits, user_function)
		return digits;
	else # if at least one of the two has children
		
		treeA_A11 = treeA.A11; treeA_A22 = treeA.A22;
		treeB_A11 = treeB.A11; treeB_A22 = treeB.A22;
		
		if ! isa(treeA.A11, BivMatTreeTail)
			m1 = length(treeA.A11.ind);
		else
			m1 = length(iA);
			treeA_A11 = treeA; # copy the node into the first child
		end
		if ! isa(treeB.A11, BivMatTreeTail)
			n1 = length(treeB.A11.ind);
		else
			n1 = length(iB);
			treeB_A11 = treeB; # copy the node into the first child
    end
		
    # Split the problem by halving the blocks of A and B
    V1 = (treeA.sylv !== nothing) ? treeA.sylv : zeros(m1, 0)
    W1 = (treeB.sylv !== nothing) ? treeB.sylv : zeros(n1, 0)
				
		C11 = C[1:m1, 1:n1] + V1 * C[m1+1:end,1:n1]
		C21 = C[m1+1:end,1:n1]
		C12 = -C[1:m1,1:n1] * W1 + C[1:m1,n1+1:end] - V1 * C[m1+1:end,1:n1] * W1 + V1 * C[m1+1:end,n1+1:end]
		C22 = -C[m1+1:end,1:n1] * W1 + C[m1+1:end,n1+1:end]

		F11 = similar(C11)
		F21 = similar(C21)
		F12 = similar(C12)
    F22 = similar(C22)

    if parallel && level <= 2
      t11 = Threads.@spawn fun2m_ric!(f, F11, A, B, C11, treeA_A11, treeB_A11, method, min_digits, level+1, user_function, parallel);
      t21 = Threads.@spawn fun2m_ric!(f, F21, A, B, C21, treeA_A22, treeB_A11, method, min_digits, level+1, user_function, parallel);
      t12 = Threads.@spawn fun2m_ric!(f, F12, A, B, C12, treeA_A11, treeB_A22, method, min_digits, level+1, user_function, parallel);
      t22 = Threads.@spawn fun2m_ric!(f, F22, A, B, C22, treeA_A22, treeB_A22, method, min_digits, level+1, user_function, parallel);
      digits = max(fetch(t11), fetch(t12), fetch(t21), fetch(t22))
    else
      digits11 = fun2m_ric!(f, F11, A, B, C11, treeA_A11, treeB_A11, method, min_digits, level+1, user_function, parallel);
      digits21 = fun2m_ric!(f, F21, A, B, C21, treeA_A22, treeB_A11, method, min_digits, level+1, user_function, parallel);
      digits12 = fun2m_ric!(f, F12, A, B, C12, treeA_A11, treeB_A22, method, min_digits, level+1, user_function, parallel);
      digits22 = fun2m_ric!(f, F22, A, B, C22, treeA_A22, treeB_A22, method, min_digits, level+1, user_function, parallel);
      digits = max(digits11, digits12, digits21, digits22);
    end
		
		
		F[1:m1,1:n1] = F11 - V1 * F21;
		F[1:m1,n1+1:end] = F11 * W1 - V1 * F21 * W1 + F12 - V1 * F22;
		F[m1+1:end,1:n1] = F21;
		F[m1+1:end,n1+1:end] = F21 * W1 + F22;
		
		return digits;
	end
	
end
