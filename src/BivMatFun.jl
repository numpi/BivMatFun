module BivMatFun

export fun2m

include("types.jl")
include("mpfr_tls.jl")
include("mp.jl")
include("utils.jl")

include("atoms.jl")
include("blocking.jl")
include("eigenvalues.jl")
include("fun2m.jl")
include("taylor.jl")

include("condest.jl")

end
