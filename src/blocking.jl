using LinearAlgebra;

function blocking(A, delta = 0.1)
    #BLOCKING  Produce blocking pattern for block Parlett recurrence in FUNM.
    #   M = BLOCKING(A, DELTA, SHOWPLOT) accepts an upper triangular matrix
    #   A and produces a blocking pattern, specified by the vector M,
    #   for the block Parlett recurrence.
    #   M(i) is the index of the block into which A(i,i) should be placed,
    #   for i=1:LENGTH(A).
    #   DELTA is a gap parameter (default 0.1) used to determine the blocking.
    #
    #   For A coming from a real matrix it should be posible to take
    #   advantage of the symmetry about the real axis.  This code does not.
    
    a = diag(A); n = length(a);
    m = zeros(LinearAlgebra.BlasInt, n); maxM = 0;
    
    for i = 1 : n
        
        if m[i] == 0
            m[i] = maxM + 1; # If a(i) hasn`t been assigned to a set
            maxM = maxM + 1; # then make a new set and assign a(i) to it.
        end
        
        for j = i+1 : n
            if m[i] != m[j]    # If a(i) and a(j) are not in same set.
                if abs(a[i]-a[j]) <= delta
                    
                    if m[j] == 0
                        m[j] = m[i]; # If a(j) hasn`t been assigned to a
                        # set, assign it to the same set as a(i).
                    end
                    
                    for j = i+1 : n
                        if m[i] != m[j]    # If a(i) and a(j) are not in same set.
                            if abs(a[i]-a[j]) <= delta
                                
                                if m[j] == 0
                                    m[j] = m[i]; # If a(j) hasn`t been assigned to a
                                    # set, assign it to the same set as a(i).
                                else
                                    p = max(m[i],m[j]);
                                    q = min(m[i],m[j]);
                                    
                                    m[m .== p] .= q; # If a(j) has been assigned to a set
                                    # place all the elements in the set
                                    # containing a(j) into the set
                                    # containing a(i) (or vice versa).
                                    m[m .> p] .= m[m .> p] .- 1;
                                    maxM = maxM - 1;
                                    # Tidying up. As we have deleted set
                                    # p we reduce the index of the sets
                                    # > p by 1.
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return m
end

function largest_block(ind)
    n = length(ind);
    v = zeros(LinearAlgebra.BlasInt, n);
    
    for i = 1 : n
        v[ind[i]] += 1;
    end
    
    return maximum(v)
end

function merge_blocks(ord, ind, block_size)
    #MERGE_BLOCKS returns [ord, ind]
    
    ind2 = [];
    cluster = [];
    
    for j = 1 : length(ind)
        if length(cluster) < block_size
            cluster = [ cluster ind{j} ];
        else
            ind2 = [ ind2 cluster ];
            ord[cluster] = j;
            cluster = [ ind[j] ];
        end
    end
    
    if cluster != []
        ind2 = [ ind2 cluster ];
        ord[cluster] = j;
    end
    
    ind = ind2;
    
end

function swapping(m)
    #SWAPPING  Choose confluent permutation ordered by average index.
    #   [MM,IND] = SWAPPING(M) takes a vector M containing the integers
    #   1:K (some repeated if K < LENGTH(M)), where M(J) is the index of
    #   the block into which the element T(J,J) of a Schur form T
    #   should be placed.
    #   It constructs a vector MM (a permutation of M) such that T(J,J)
    #   will be located in the MM(J)'th block counting from the (1,1) position.
    #   The algorithm used is to order the blocks by ascending
    #   average index in M, which is a heuristic for minimizing the number
    #   of swaps required to achieve this confluent permutation.
    #   The cell array vector IND defines the resulting block form:
    #   IND{i} contains the indices of the i'th block in the permuted form.
    
    mmax = maximum(m); mm = zeros(LinearAlgebra.BlasInt, size(m));
    g = zeros(mmax); h = zeros(mmax);
    
    for i = 1:mmax
        p = findall(m -> m==i, m);
        h[i] = length(p);
        g[i] = sum(p)/h[i];
    end
    
    y = sortperm(g);
    h = [0 ; cumsum(h[y])];
    
    ind = [];
    for i = 1 : mmax
        mm[m .== y[i]] .= i;
        rg = convert(Array{LinearAlgebra.BlasInt}, h[i]+1 : h[i+1])
        ind = [ ind ; [ rg ] ];
    end
    
    return mm, ind;
end
