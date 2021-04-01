using LinearAlgebra;

"""
Algorithm used to evaluate a bivariate function using the 
    bivariate Schur-Parlett scheme. 
    """
    @enum Algorithm begin
        Taylor
        Diag
        DiagNoHp
        User
    end
    
    abstract type BivMatTree end
    
    mutable struct BivMatTreeImpl <: BivMatTree
        A11::BivMatTree
        A22::BivMatTree
        kV::Float64
        V::Union{UpperTriangular, Nothing} # Eigenvector basis
        D::Union{Array, Nothing} # Diagonal matrix
        Ttilde::Union{Matrix, Nothing} # Perturbed Schur form
        ind::Union{Array{LinearAlgebra.BlasInt}, Nothing}
        sylv::Union{Matrix{ComplexF64}, Nothing}
    end
    
    struct BivMatTreeTail <: BivMatTree
    end
    
    function newBivMatTree()
        return BivMatTreeImpl(
        BivMatTreeTail(), # A11
        BivMatTreeTail(), # A22
        1.0, # kV
        nothing, # V
        nothing, # D
        nothing, # Ttilde
        nothing, # ind
        nothing) # Sylv
    end
    
    mutable struct Fun2MInfo
        nblocksA
        nblocksB
        time_schur_form
        time_blocking
        time_fun2mric
        time_sylvester
        time_diag
        maxdeg
        digits
    end
    