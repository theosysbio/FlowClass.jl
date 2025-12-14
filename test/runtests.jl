"""
Main test runner for FlowClass.jl
"""

using Test
using FlowClass

@testset "FlowClass.jl" begin
    include("test_types.jl")
    include("test_jacobian.jl")
    include("test_curl.jl")
    include("test_fixed_points.jl")
    include("test_periodic_orbits.jl")
    include("test_manifolds.jl")
    include("test_classification.jl")
end
