function bits_for_digits(digits)
    Int64(ceil(digits * log2(10)))
end

function mp(A::UpperTriangular, digits)
    UpperTriangular(mp(A.data, digits))
end

function with_digits(fun, digits)
    MPFR_TLS.setprecision(bits_for_digits(digits)) do 
        return fun();
    end
end

function mp(A::Array{T, N}, digits::Int64) where {T <: Complex, N}
    MPFR_TLS.setprecision(bits_for_digits(digits)) do
        return convert(Array{Complex{MPFR_TLS.BigFloat}, N}, A)
    end
end

function mp(x::T, digits::Int64) where T <: Complex
    MPFR_TLS.setprecision(bits_for_digits(digits)) do
        return convert(Complex{MPFR_TLS.BigFloat}, x)
    end
end

function mp(A::Array{T, N}, digits::Int64) where {T <: AbstractFloat, N}
    MPFR_TLS.setprecision(bits_for_digits(digits)) do
        return convert(Array{MPFR_TLS.BigFloat, N}, A)
    end
end

function mp(x::T, digits::Int64) where T <: AbstractFloat
    MPFR_TLS.setprecision(bits_for_digits(digits)) do
        return convert(MPFR_TLS.BigFloat, x)
    end
end
