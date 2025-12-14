"""
Tests for jacobian.jl - Jacobian computation and symmetry analysis
"""

using Test
using FlowClass
using LinearAlgebra

@testset "compute_jacobian" begin
    
    @testset "Linear systems have constant Jacobian" begin
        # For dx/dt = Ax, the Jacobian is A everywhere
        A = [-1.0 0.5; -0.5 -1.0]
        ds = DynamicalSystem(x -> A * x, 2)
        
        # Test at origin
        J_origin = compute_jacobian(ds, [0.0, 0.0])
        @test J_origin ≈ A
        
        # Test at arbitrary point (should be the same)
        J_point = compute_jacobian(ds, [3.14, 2.71])
        @test J_point ≈ A
    end
    
    @testset "Nonlinear system Jacobian" begin
        # f(x) = [x₁², x₁*x₂]
        # Jacobian = [2x₁  0; x₂  x₁]
        f = x -> [x[1]^2, x[1] * x[2]]
        ds = DynamicalSystem(f, 2)
        
        x = [2.0, 3.0]
        J = compute_jacobian(ds, x)
        
        expected = [2*x[1]  0.0;
                    x[2]    x[1]]
        @test J ≈ expected
    end
    
    @testset "Gradient system has symmetric Jacobian" begin
        # Gradient system: dx/dt = -∇V where V(x) = x₁² + x₁*x₂ + x₂²
        # -∇V = [-2x₁ - x₂, -x₁ - 2x₂]
        # Jacobian = [-2  -1; -1  -2] which is symmetric
        f = x -> [-2x[1] - x[2], -x[1] - 2x[2]]
        ds = DynamicalSystem(f, 2)
        
        J = compute_jacobian(ds, [1.0, 1.0])
        @test J ≈ transpose(J)  # Symmetric
    end
    
    @testset "Higher dimensional system" begin
        # 4D linear system
        A = [-1.0 0.1 0.0 0.0;
              0.1 -1.0 0.1 0.0;
              0.0 0.1 -1.0 0.1;
              0.0 0.0 0.1 -1.0]
        ds = DynamicalSystem(x -> A * x, 4)
        
        J = compute_jacobian(ds, zeros(4))
        @test size(J) == (4, 4)
        @test J ≈ A
    end
    
    @testset "Direct function interface" begin
        # Test compute_jacobian with function directly (no DynamicalSystem wrapper)
        f = x -> [x[1]^2 + x[2], x[1] - x[2]^2]
        x = [1.0, 2.0]
        
        J = compute_jacobian(f, x)
        
        # J = [2x₁  1; 1  -2x₂]
        expected = [2.0 1.0; 1.0 -4.0]
        @test J ≈ expected
    end
    
    @testset "Dimension mismatch throws error" begin
        ds = DynamicalSystem(x -> -x, 2)
        @test_throws DimensionMismatch compute_jacobian(ds, [1.0, 2.0, 3.0])
    end
end

@testset "is_jacobian_symmetric" begin
    
    @testset "Symmetric matrix" begin
        J = [-2.0 0.5; 0.5 -1.0]
        @test is_jacobian_symmetric(J)
    end
    
    @testset "Non-symmetric matrix" begin
        J = [-1.0 0.5; -0.5 -1.0]
        @test !is_jacobian_symmetric(J)
    end
    
    @testset "Nearly symmetric matrix with tolerance" begin
        # Symmetric plus small perturbation
        J_base = [-2.0 0.5; 0.5 -1.0]
        ε = 1e-9
        J_perturbed = J_base + [0.0 ε; -ε 0.0]
        
        # With loose tolerance, should be considered symmetric
        @test is_jacobian_symmetric(J_perturbed; atol=1e-8)
        
        # With strict tolerance AND rtol=0, should detect the asymmetry
        # (Need rtol=0 to isolate the absolute tolerance test)
        @test !is_jacobian_symmetric(J_perturbed; rtol=0.0, atol=1e-10)
    end
    
    @testset "Zero matrix is symmetric" begin
        J = zeros(3, 3)
        @test is_jacobian_symmetric(J)
    end
    
    @testset "Identity matrix is symmetric" begin
        J = Matrix{Float64}(I, 4, 4)
        @test is_jacobian_symmetric(J)
    end
    
    @testset "DynamicalSystem interface" begin
        # Gradient system
        ds_grad = DynamicalSystem(x -> [-2x[1] - x[2], -x[1] - 2x[2]], 2)
        @test is_jacobian_symmetric(ds_grad, [1.0, 1.0])
        
        # Non-gradient system with rotation
        ds_rot = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        @test !is_jacobian_symmetric(ds_rot, [1.0, 1.0])
    end
    
    @testset "Non-square matrix throws error" begin
        J = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test_throws ArgumentError is_jacobian_symmetric(J)
    end
end

@testset "jacobian_symmetry_error" begin
    
    @testset "Symmetric matrix has zero error" begin
        J = [-2.0 0.5; 0.5 -1.0]
        @test jacobian_symmetry_error(J) ≈ 0.0 atol=1e-14
    end
    
    @testset "Antisymmetric matrix has maximum error" begin
        # Purely antisymmetric matrix
        J = [0.0 1.0; -1.0 0.0]
        err = jacobian_symmetry_error(J)
        @test err > 0
        @test err ≈ norm(J)  # For antisymmetric, error equals full norm
    end
    
    @testset "Known antisymmetric part" begin
        # J = symmetric + antisymmetric
        # Symmetric part: [-1 0; 0 -1]
        # Antisymmetric part: [0 0.5; -0.5 0]
        J = [-1.0 0.5; -0.5 -1.0]
        
        expected_antisym = [0.0 0.5; -0.5 0.0]
        expected_error = norm(expected_antisym)
        
        @test jacobian_symmetry_error(J) ≈ expected_error
    end
    
    @testset "DynamicalSystem interface" begin
        ds = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        err = jacobian_symmetry_error(ds, [0.0, 0.0])
        @test err > 0
    end
end

@testset "relative_jacobian_symmetry_error" begin
    
    @testset "Symmetric matrix has zero relative error" begin
        J = [-2.0 0.5; 0.5 -1.0]
        @test relative_jacobian_symmetry_error(J) ≈ 0.0 atol=1e-14
    end
    
    @testset "Antisymmetric matrix has relative error of 1" begin
        J = [0.0 1.0; -1.0 0.0]
        @test relative_jacobian_symmetry_error(J) ≈ 1.0
    end
    
    @testset "Scale independence" begin
        # Same matrix scaled by different factors should have same relative error
        J1 = [-1.0 0.5; -0.5 -1.0]
        J2 = 100.0 * J1
        J3 = 0.01 * J1
        
        @test relative_jacobian_symmetry_error(J1) ≈ relative_jacobian_symmetry_error(J2)
        @test relative_jacobian_symmetry_error(J1) ≈ relative_jacobian_symmetry_error(J3)
    end
    
    @testset "Zero matrix returns zero" begin
        J = zeros(3, 3)
        @test relative_jacobian_symmetry_error(J) ≈ 0.0
    end
    
    @testset "DynamicalSystem interface" begin
        ds = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        err = relative_jacobian_symmetry_error(ds, [0.0, 0.0])
        @test 0 ≤ err ≤ 1
    end
end

@testset "Gradient vs Non-Gradient Systems" begin
    
    @testset "Pure gradient system" begin
        # V(x) = x₁² + x₂² + x₃²  (sphere)
        # -∇V = [-2x₁, -2x₂, -2x₃]
        ds = DynamicalSystem(x -> -2 .* x, 3)
        
        # Test at multiple points
        for _ in 1:10
            x = randn(3)
            @test is_jacobian_symmetric(ds, x)
            @test jacobian_symmetry_error(ds, x) < 1e-10
        end
    end
    
    @testset "Gradient system with cross terms" begin
        # V(x) = x₁² + x₁x₂ + x₂²
        # -∇V = [-2x₁ - x₂, -x₁ - 2x₂]
        ds = DynamicalSystem(x -> [-2x[1] - x[2], -x[1] - 2x[2]], 2)
        
        x = randn(2)
        @test is_jacobian_symmetric(ds, x)
    end
    
    @testset "System with rotation (non-gradient)" begin
        # Damped harmonic oscillator with rotation
        # dx₁/dt = -x₁ + ω*x₂
        # dx₂/dt = -ω*x₁ - x₂
        ω = 1.0
        ds = DynamicalSystem(x -> [-x[1] + ω*x[2], -ω*x[1] - x[2]], 2)
        
        x = randn(2)
        @test !is_jacobian_symmetric(ds, x)
        @test jacobian_symmetry_error(ds, x) > 0
    end
    
    @testset "Lorenz system (non-gradient)" begin
        function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
            return [σ * (x[2] - x[1]),
                    x[1] * (ρ - x[3]) - x[2],
                    x[1] * x[2] - β * x[3]]
        end
        ds = DynamicalSystem(lorenz, 3)
        
        # Lorenz is definitely not a gradient system
        x = [1.0, 1.0, 1.0]
        @test !is_jacobian_symmetric(ds, x)
    end
end
