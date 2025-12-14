"""
Tests for curl.jl - Curl computation and analysis
"""

using Test
using FlowClass
using LinearAlgebra

@testset "Matrix Decomposition" begin
    
    @testset "compute_antisymmetric_part" begin
        # General matrix
        J = [-1.0 0.5; -0.5 -1.0]
        A = compute_antisymmetric_part(J)
        
        # Check antisymmetry: A = -Aᵀ
        @test A ≈ -transpose(A)
        
        # Check known result
        expected = [0.0 0.5; -0.5 0.0]
        @test A ≈ expected
        
        # Symmetric matrix has zero antisymmetric part
        J_sym = [-2.0 0.5; 0.5 -1.0]
        A_sym = compute_antisymmetric_part(J_sym)
        @test norm(A_sym) ≈ 0.0 atol=1e-14
        
        # Antisymmetric matrix equals its antisymmetric part
        J_antisym = [0.0 1.0; -1.0 0.0]
        A_antisym = compute_antisymmetric_part(J_antisym)
        @test A_antisym ≈ J_antisym
    end
    
    @testset "compute_symmetric_part" begin
        J = [-1.0 0.5; -0.5 -1.0]
        S = compute_symmetric_part(J)
        
        # Check symmetry: S = Sᵀ
        @test S ≈ transpose(S)
        
        # Check known result
        expected = [-1.0 0.0; 0.0 -1.0]
        @test S ≈ expected
        
        # Symmetric matrix equals its symmetric part
        J_sym = [-2.0 0.5; 0.5 -1.0]
        S_sym = compute_symmetric_part(J_sym)
        @test S_sym ≈ J_sym
    end
    
    @testset "decompose_jacobian" begin
        J = [-1.0 0.5; -0.5 -1.0]
        S, A = decompose_jacobian(J)
        
        # J = S + A
        @test J ≈ S + A
        
        # S is symmetric
        @test S ≈ transpose(S)
        
        # A is antisymmetric
        @test A ≈ -transpose(A)
    end
    
    @testset "decompose_jacobian with DynamicalSystem" begin
        ds = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        x = [1.0, 2.0]
        
        S, A = decompose_jacobian(ds, x)
        J = compute_jacobian(ds, x)
        
        @test J ≈ S + A
    end
    
    @testset "Non-square matrix throws error" begin
        J_rect = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test_throws ArgumentError compute_antisymmetric_part(J_rect)
        @test_throws ArgumentError compute_symmetric_part(J_rect)
    end
end

@testset "Curl Magnitude (n-dimensional)" begin
    
    @testset "curl_magnitude basics" begin
        # Symmetric matrix: zero curl
        J_sym = [-2.0 0.5; 0.5 -1.0]
        @test curl_magnitude(J_sym) ≈ 0.0 atol=1e-14
        
        # Antisymmetric matrix: full curl
        J_antisym = [0.0 1.0; -1.0 0.0]
        @test curl_magnitude(J_antisym) ≈ norm(J_antisym)
        
        # Mixed matrix
        J = [-1.0 0.5; -0.5 -1.0]
        A = compute_antisymmetric_part(J)
        @test curl_magnitude(J) ≈ norm(A)
    end
    
    @testset "curl_magnitude with DynamicalSystem" begin
        # Gradient system
        ds_grad = DynamicalSystem(x -> [-2x[1] - x[2], -x[1] - 2x[2]], 2)
        @test curl_magnitude(ds_grad, [1.0, 1.0]) ≈ 0.0 atol=1e-10
        
        # System with rotation
        ds_rot = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        @test curl_magnitude(ds_rot, [0.0, 0.0]) > 0
    end
    
    @testset "curl_magnitude with function" begin
        f = x -> [-x[1] + x[2], -x[1] - x[2]]
        @test curl_magnitude(f, [0.0, 0.0]) > 0
    end
    
    @testset "relative_curl_magnitude" begin
        # Symmetric: relative curl = 0
        J_sym = [-2.0 0.5; 0.5 -1.0]
        @test relative_curl_magnitude(J_sym) ≈ 0.0 atol=1e-14
        
        # Antisymmetric: relative curl = 1
        J_antisym = [0.0 1.0; -1.0 0.0]
        @test relative_curl_magnitude(J_antisym) ≈ 1.0
        
        # Zero matrix: relative curl = 0
        J_zero = zeros(3, 3)
        @test relative_curl_magnitude(J_zero) ≈ 0.0
        
        # Scale independence
        J = [-1.0 0.5; -0.5 -1.0]
        @test relative_curl_magnitude(J) ≈ relative_curl_magnitude(10.0 * J)
        @test relative_curl_magnitude(J) ≈ relative_curl_magnitude(0.01 * J)
    end
    
    @testset "Higher dimensions" begin
        # 4D system
        J = [-1.0 0.1 0.0 0.0;
             -0.1 -1.0 0.2 0.0;
              0.0 -0.2 -1.0 0.1;
              0.0 0.0 -0.1 -1.0]
        
        # Should work without error
        c = curl_magnitude(J)
        @test c > 0
        
        rc = relative_curl_magnitude(J)
        @test 0 ≤ rc ≤ 1
    end
end

@testset "3D Curl Vector" begin
    
    @testset "compute_curl_3d basics" begin
        # Pure rotation around z-axis: F = [-y, x, 0] = [-x₂, x₁, 0]
        # Jacobian: J[i,j] = ∂Fᵢ/∂xⱼ
        # J = [∂F₁/∂x₁  ∂F₁/∂x₂  ∂F₁/∂x₃]   [0  -1  0]
        #     [∂F₂/∂x₁  ∂F₂/∂x₂  ∂F₂/∂x₃] = [1   0  0]
        #     [∂F₃/∂x₁  ∂F₃/∂x₂  ∂F₃/∂x₃]   [0   0  0]
        # Curl = [0, 0, 1 - (-1)] = [0, 0, 2]
        J = [0.0 -1.0 0.0; 
             1.0  0.0 0.0; 
             0.0  0.0 0.0]
        curl = compute_curl_3d(J)
        @test curl ≈ [0.0, 0.0, 2.0]
    end
    
    @testset "compute_curl_3d for gradient system" begin
        # Gradient system: F = -∇V = [-2x₁, -2x₂, -2x₃]
        # Jacobian is diagonal: J = diag(-2, -2, -2)
        # Curl should be zero
        J = [-2.0 0.0 0.0;
              0.0 -2.0 0.0;
              0.0 0.0 -2.0]
        curl = compute_curl_3d(J)
        @test norm(curl) ≈ 0.0 atol=1e-14
    end
    
    @testset "compute_curl_3d with DynamicalSystem" begin
        # F = [-x₂, x₁, 0] (rotation around z)
        ds = DynamicalSystem(x -> [-x[2], x[1], 0.0], 3)
        curl = compute_curl_3d(ds, [1.0, 2.0, 3.0])
        @test curl ≈ [0.0, 0.0, 2.0]
    end
    
    @testset "compute_curl_3d with function" begin
        f = x -> [-x[2], x[1], 0.0]
        curl = compute_curl_3d(f, [0.0, 0.0, 0.0])
        @test curl ≈ [0.0, 0.0, 2.0]
    end
    
    @testset "Dimension errors" begin
        # 2D matrix
        J_2d = [1.0 2.0; 3.0 4.0]
        @test_throws ArgumentError compute_curl_3d(J_2d)
        
        # 2D dynamical system
        ds_2d = DynamicalSystem(x -> -x, 2)
        @test_throws ArgumentError compute_curl_3d(ds_2d, [1.0, 2.0])
        
        # 2D vector input
        f = x -> -x
        @test_throws ArgumentError compute_curl_3d(f, [1.0, 2.0])
    end
end

@testset "Classification Helpers" begin
    
    @testset "is_curl_free" begin
        # Symmetric Jacobian (gradient system)
        J_sym = [-2.0 0.5; 0.5 -1.0]
        @test is_curl_free(J_sym)
        
        # Non-symmetric Jacobian
        J_rot = [-1.0 1.0; -1.0 -1.0]
        @test !is_curl_free(J_rot)
        
        # Nearly symmetric with tolerance
        J_base = [-2.0 0.5; 0.5 -1.0]
        ε = 1e-9
        J_perturbed = J_base + [0.0 ε; -ε 0.0]
        @test is_curl_free(J_perturbed; atol=1e-8)
        @test !is_curl_free(J_perturbed; rtol=0.0, atol=1e-10)
    end
    
    @testset "is_curl_free with DynamicalSystem" begin
        # Gradient system
        ds_grad = DynamicalSystem(x -> [-2x[1] - x[2], -x[1] - 2x[2]], 2)
        @test is_curl_free(ds_grad, [1.0, 2.0])
        
        # Non-gradient system
        ds_rot = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        @test !is_curl_free(ds_rot, [0.0, 0.0])
    end
    
    @testset "is_approximately_curl_free" begin
        # Exactly curl-free
        J_grad = [-2.0 0.5; 0.5 -1.0]
        @test is_approximately_curl_free(J_grad; threshold=0.1)
        
        # Small curl (gradient-like)
        J_small_curl = [-1.0 0.05; -0.05 -1.0]
        @test is_approximately_curl_free(J_small_curl; threshold=0.1)
        
        # Large curl
        J_large_curl = [-1.0 0.5; -0.5 -1.0]
        @test !is_approximately_curl_free(J_large_curl; threshold=0.1)
        
        # Threshold validation
        @test_throws ArgumentError is_approximately_curl_free(J_grad; threshold=-0.1)
        @test_throws ArgumentError is_approximately_curl_free(J_grad; threshold=0.0)
    end
    
    @testset "is_approximately_curl_free with DynamicalSystem" begin
        # Small rotation
        ω = 0.01
        ds_small = DynamicalSystem(x -> [-x[1] + ω*x[2], -ω*x[1] - x[2]], 2)
        @test is_approximately_curl_free(ds_small, [0.0, 0.0]; threshold=0.1)
        
        # Large rotation
        ω = 1.0
        ds_large = DynamicalSystem(x -> [-x[1] + ω*x[2], -ω*x[1] - x[2]], 2)
        @test !is_approximately_curl_free(ds_large, [0.0, 0.0]; threshold=0.1)
    end
end

@testset "Gradient Component Magnitude" begin
    
    @testset "gradient_component_magnitude" begin
        # Pure gradient (all symmetric)
        J_sym = [-2.0 0.5; 0.5 -1.0]
        @test gradient_component_magnitude(J_sym) ≈ norm(J_sym)
        
        # Pure rotation (no gradient)
        J_antisym = [0.0 1.0; -1.0 0.0]
        @test gradient_component_magnitude(J_antisym) ≈ 0.0 atol=1e-14
        
        # Mixed
        J = [-1.0 0.5; -0.5 -1.0]
        S = compute_symmetric_part(J)
        @test gradient_component_magnitude(J) ≈ norm(S)
    end
    
    @testset "gradient_component_magnitude with DynamicalSystem" begin
        ds = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        g = gradient_component_magnitude(ds, [0.0, 0.0])
        @test g > 0
    end
end

@testset "Curl to Gradient Ratio" begin
    
    @testset "curl_to_gradient_ratio" begin
        # Pure gradient: ratio = 0
        J_sym = [-2.0 0.5; 0.5 -1.0]
        @test curl_to_gradient_ratio(J_sym) ≈ 0.0 atol=1e-14
        
        # Pure rotation: ratio = Inf
        J_antisym = [0.0 1.0; -1.0 0.0]
        @test curl_to_gradient_ratio(J_antisym) == Inf
        
        # Zero matrix: ratio = NaN
        J_zero = zeros(2, 2)
        @test isnan(curl_to_gradient_ratio(J_zero))
        
        # Equal parts: ratio = 1
        # J = S + A where ‖S‖ = ‖A‖
        # Example: J = [0 1; 0 0] has S = [0 0.5; 0.5 0], A = [0 0.5; -0.5 0]
        J = [0.0 1.0; 0.0 0.0]
        S = compute_symmetric_part(J)
        A = compute_antisymmetric_part(J)
        @test norm(S) ≈ norm(A)  # Verify equal magnitudes
        @test curl_to_gradient_ratio(J) ≈ 1.0
    end
    
    @testset "curl_to_gradient_ratio with DynamicalSystem" begin
        ds = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        ratio = curl_to_gradient_ratio(ds, [0.0, 0.0])
        @test ratio > 0
        @test isfinite(ratio)
    end
end

@testset "Consistency with Jacobian Functions" begin
    
    @testset "curl_magnitude equals jacobian_symmetry_error" begin
        for _ in 1:10
            J = randn(4, 4)
            @test curl_magnitude(J) ≈ jacobian_symmetry_error(J)
        end
    end
    
    @testset "relative_curl_magnitude equals relative_jacobian_symmetry_error" begin
        for _ in 1:10
            J = randn(3, 3)
            @test relative_curl_magnitude(J) ≈ relative_jacobian_symmetry_error(J)
        end
    end
    
    @testset "is_curl_free is equivalent to is_jacobian_symmetric" begin
        J_sym = [-2.0 0.5; 0.5 -1.0]
        J_rot = [-1.0 1.0; -1.0 -1.0]
        
        @test is_curl_free(J_sym) == is_jacobian_symmetric(J_sym)
        @test is_curl_free(J_rot) == is_jacobian_symmetric(J_rot)
    end
end

@testset "Gradient vs Non-Gradient Classification" begin
    
    @testset "Pure gradient systems are curl-free everywhere" begin
        # V(x) = x₁² + x₁x₂ + x₂² + x₃²
        # -∇V = [-2x₁ - x₂, -x₁ - 2x₂, -2x₃]
        ds = DynamicalSystem(x -> [-2x[1] - x[2], -x[1] - 2x[2], -2x[3]], 3)
        
        for _ in 1:10
            x = randn(3)
            @test is_curl_free(ds, x)
            @test curl_magnitude(ds, x) < 1e-10
        end
    end
    
    @testset "Lorenz system has non-zero curl" begin
        function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
            return [σ * (x[2] - x[1]),
                    x[1] * (ρ - x[3]) - x[2],
                    x[1] * x[2] - β * x[3]]
        end
        ds = DynamicalSystem(lorenz, 3)
        
        x = [1.0, 1.0, 1.0]
        @test !is_curl_free(ds, x)
        @test curl_magnitude(ds, x) > 0
        
        # Can also check the 3D curl vector
        curl = compute_curl_3d(ds, x)
        @test norm(curl) > 0
    end
end
