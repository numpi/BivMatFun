using BivMatFun
using Test

@testset "BivMatFun.jl" begin
  if isempty(ARGS) || "all" in ARGS
    all_tests = true
  else
    all_tests = false
  end

  if all_tests || "experiment1" in ARGS
    @test include("experiment1.jl");
  end

  if all_tests || "experiment2" in ARGS
    @test include("experiment2.jl");
  end

  if all_tests || "experiment3" in ARGS
    @test include("experiment3.jl");
  end
end

