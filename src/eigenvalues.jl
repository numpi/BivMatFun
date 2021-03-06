using LinearAlgebra;

import LinearAlgebra.ordschur;
import LinearAlgebra.ordschur!;

function eigentri(T)
  n = size(T, 1);
  
  n == size(T, 2) || error("Non-square matrix passed to eigentri");
  
  D = diag(T);
  V = -UpperTriangular(one(T));
  
  @inbounds for j = 2 : n
    # We solve the  shifted linear system, and compute the infinity norm as
    # we do it for the final scaling.
    nrm = abs(V[j,j]);
    
    V[1:j-1,j] .= T[1:j-1,j];
    
    @inbounds for k = j-1 : -1 : 1
      V[k,j] = V[k,j] / (T[k,k] - D[j]);
      nrm = max(nrm, abs(V[k,j]));
      
      # Apply the correction to the other entries of the vector
      @inbounds for i = 1 : k-1
        V[i,j] = V[i,j] - T[i,k] * V[k,j];
      end
    end
    
    # Make the vector of unit infinity norm
    @inbounds for i = 1 : j
      V[i,j] = V[i,j] / nrm;
    end
  end
  
  return (D, V)
end


function condeigvec(TA)
  #Estimate the condition number of the eigenvector matrix of a triangular matrix
  # Returns [TAtilde, kVA]
  
  d_uh = 16; # the default
  u = eps(1/2);
  m = size(TA, 1);
  
  delta1 = 5e-3;
  
  EA = diagm(randn(m));
  max_tAij = maximum(map(abs, triu(TA, 1)));
  
  EA = EA / norm(EA) * u * max_tAij;
  
  # Use precision u^2 to form T + E since E might be
  # too small to be added by T in precision u
  d_default = 48; # precision with unit roundoff u^2
  
  # TAtilde = mp(TA) + mp(EA);
  # TAtilde = convert(Matrix{ArbComplex{b_defaults}}, TA) +
  # convert(Matrix{ArbComplex{b_defaults}}, EA)
  TAtilde = mp(TA, d_default) + mp(EA, d_default)
  
  # Calculate the largest group size k
  ordA = blocking(convert(Matrix{ComplexF64}, TAtilde),delta1);
  kA = largest_block(ordA);
  
  if kA > 1  # evaluate the required precision uh
    diagTAtilde = diag(TAtilde);
    sA = length(diagTAtilde);
    sepA = minimum([ convert(Float64, abs(x - y)) for x in diagTAtilde, y in diagTAtilde ] + diagm(Inf * ones(sA)));
    kVA = m * max_tAij * (1 + max_tAij / sepA)^(kA-2) / sepA;
  else
    kVA = 1.0;
  end
  
  return TAtilde, kVA;
end

function ordschur!(F::Schur{<: Real}, clusters::Array{LinearAlgebra.BlasInt, 1})
  perm = sortperm(clusters, rev = true)
  n = size(F.T, 1)
  i = 0
  
  cids = unique(clusters);
  sort!(cids, rev = true)
  
  for j = 1 : length(cids)
    cid = cids[j]
    select = convert(Array{LinearAlgebra.BlasInt, 1}, clusters[i+1:n] .== cid)::Array{LinearAlgebra.BlasInt, 1}
    ncluster = sum(select)
    
    LAPACK.trsen!('V', 'N', select, 
    view(F.T, i+1:n, i+1:n), view(F.Z, 1:n, i+1:n))
    
    clusters[i+ncluster+1:n] .= filter((j) -> j != cid, clusters[i+1:n])
    clusters[i+1:i+ncluster] .= cid;
    
    i = i + ncluster
  end
  
  F
end

function ordschur!(F::Schur{<: Complex}, clusters::Array{LinearAlgebra.BlasInt, 1})
  
  perm = sortperm(clusters, rev = true);
  n = length(perm);
  
  for i = 1 : n
    # We swap move perm[i] to i
    if i != perm[i]
      ifst::LinearAlgebra.BlasInt = perm[i];
      ilst::LinearAlgebra.BlasInt = i;
      
      # FIXME: This does not work in real arithmetic, because 
      # blocks are moved in 2x2 patterns. 
      LAPACK.trexc!('V', ifst, ilst, F.T, F.Z);      
      
      l = perm[i];
      perm[(perm .>= i) .* (perm .< l)] .+= 1
      perm[i] = i;
    end
  end
  
  F.values[:] = F.values[perm];
  
  F
end

function ordschur(F::Schur, clusters::Array{LinearAlgebra.BlasInt, 1})
  ordschur!(deepcopy(F), deepcopy(clusters))
end
