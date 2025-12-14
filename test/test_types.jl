"""
Tests for types.jl - DynamicalSystem struct and related functionality
"""

using Test
using FlowClass

@testset "DynamicalSystem Construction" begin
    
    @testset "Basic construction with dimension" begin
        # Simple linear system
        f = x -> -x
        ds = DynamicalSystem(f, 2)
        
        @test dimension(ds) == 2
        @test ds.dim == 2
    end
    
    @testset "Construction with sample point" begin
        # Infer dimension from sample point
        f = x -> [-x[1], -x[2], -x[3]]
        x₀ = [1.0, 2.0, 3.0]
        ds = DynamicalSystem(f, x₀)
        
        @test dimension(ds) == 3
    end
    
    @testset "Invalid dimension throws error" begin
        f = x -> -x
        @test_throws ArgumentError DynamicalSystem(f, 0)
        @test_throws ArgumentError DynamicalSystem(f, -1)
    end
    
    @testset "Dimension mismatch in constructor throws error" begin
        # Function returns 2D but sample point is 3D
        f = x -> [-x[1], -x[2]]
        x₀ = [1.0, 2.0, 3.0]
        @test_throws DimensionMismatch DynamicalSystem(f, x₀)
    end
end

@testset "DynamicalSystem Evaluation" begin
    
    @testset "Callable interface" begin
        A = [-1.0 0.5; -0.5 -1.0]
        ds = DynamicalSystem(x -> A * x, 2)
        
        x = [1.0, 2.0]
        result = ds(x)
        expected = A * x
        
        @test result ≈ expected
    end
    
    @testset "Dimension check on evaluation" begin
        ds = DynamicalSystem(x -> -x, 2)
        
        # Wrong dimension should throw
        @test_throws DimensionMismatch ds([1.0, 2.0, 3.0])
        @test_throws DimensionMismatch ds([1.0])
    end
    
    @testset "Various system types" begin
        # Gradient system: dx/dt = -∇V where V(x) = (x₁² + x₂²)/2
        ds_grad = DynamicalSystem(x -> [-x[1], -x[2]], 2)
        @test ds_grad([1.0, 1.0]) ≈ [-1.0, -1.0]
        
        # Lorenz system
        function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
            return [σ * (x[2] - x[1]),
                    x[1] * (ρ - x[3]) - x[2],
                    x[1] * x[2] - β * x[3]]
        end
        ds_lorenz = DynamicalSystem(lorenz, 3)
        @test length(ds_lorenz([1.0, 1.0, 1.0])) == 3
    end
end

@testset "DynamicalSystem Display" begin
    ds = DynamicalSystem(x -> -x, 2)
    
    # Test that show doesn't throw
    io = IOBuffer()
    show(io, ds)
    str = String(take!(io))
    @test occursin("DynamicalSystem", str)
    @test occursin("2", str)
    
    # Test text/plain MIME
    io = IOBuffer()
    show(io, MIME"text/plain"(), ds)
    str = String(take!(io))
    @test occursin("Dimension: 2", str)
end
