"""
Tests for fixed_points.jl - Fixed point finding and classification
"""

using Test
using FlowClass
using LinearAlgebra

@testset "Eigenvalue Analysis" begin
    
    @testset "compute_eigenvalues" begin
        # Linear system with known eigenvalues
        # dx/dt = Ax where A has eigenvalues -1 and -2
        A = [-1.5 0.5; 0.5 -1.5]  # eigenvalues are -1 and -2
        ds = DynamicalSystem(x -> A * x, 2)
        
        eigs = compute_eigenvalues(ds, [0.0, 0.0])
        eigs_sorted = sort(real.(eigs))
        @test eigs_sorted ≈ [-2.0, -1.0]
    end
    
    @testset "is_hyperbolic" begin
        # Hyperbolic: all eigenvalues have non-zero real parts
        eigs_hyperbolic = [ComplexF64(-1.0, 0.0), ComplexF64(-2.0, 0.0)]
        @test is_hyperbolic(eigs_hyperbolic)
        
        # Non-hyperbolic: has zero real part
        eigs_center = [ComplexF64(0.0, 1.0), ComplexF64(0.0, -1.0)]
        @test !is_hyperbolic(eigs_center)
        
        # Complex but hyperbolic
        eigs_focus = [ComplexF64(-1.0, 2.0), ComplexF64(-1.0, -2.0)]
        @test is_hyperbolic(eigs_focus)
    end
    
    @testset "is_stable" begin
        # Stable: all Re(λ) < 0
        eigs_stable = [ComplexF64(-1.0, 0.0), ComplexF64(-2.0, 1.0)]
        @test is_stable(eigs_stable)
        
        # Unstable: some Re(λ) > 0
        eigs_unstable = [ComplexF64(1.0, 0.0), ComplexF64(-2.0, 0.0)]
        @test !is_stable(eigs_unstable)
        
        # Marginally stable (center): Re(λ) = 0
        eigs_marginal = [ComplexF64(0.0, 1.0), ComplexF64(0.0, -1.0)]
        @test !is_stable(eigs_marginal)
    end
end

@testset "Fixed Point Type Classification" begin
    
    @testset "classify_fixed_point_type" begin
        # Stable node: all real, negative
        @test classify_fixed_point_type([ComplexF64(-1.0), ComplexF64(-2.0)]) == STABLE_NODE
        
        # Unstable node: all real, positive
        @test classify_fixed_point_type([ComplexF64(1.0), ComplexF64(2.0)]) == UNSTABLE_NODE
        
        # Saddle: real, mixed sign
        @test classify_fixed_point_type([ComplexF64(-1.0), ComplexF64(2.0)]) == SADDLE
        
        # Stable focus: complex with negative real parts
        @test classify_fixed_point_type([ComplexF64(-1.0, 2.0), ComplexF64(-1.0, -2.0)]) == STABLE_FOCUS
        
        # Unstable focus: complex with positive real parts
        @test classify_fixed_point_type([ComplexF64(1.0, 2.0), ComplexF64(1.0, -2.0)]) == UNSTABLE_FOCUS
        
        # Center: pure imaginary
        @test classify_fixed_point_type([ComplexF64(0.0, 1.0), ComplexF64(0.0, -1.0)]) == CENTER
        
        # Saddle focus: mixed stability with complex eigenvalues
        @test classify_fixed_point_type([ComplexF64(-1.0, 2.0), ComplexF64(-1.0, -2.0), ComplexF64(1.0)]) == SADDLE_FOCUS
    end
    
    @testset "classify_fixed_point with DynamicalSystem" begin
        # Simple stable node at origin
        ds = DynamicalSystem(x -> [-x[1], -2x[2]], 2)
        fp = classify_fixed_point(ds, [0.0, 0.0])
        
        @test fp.type == STABLE_NODE
        @test fp.is_stable
        @test fp.is_hyperbolic
        @test fp.location ≈ [0.0, 0.0]
    end
    
    @testset "classify_fixed_point verification" begin
        ds = DynamicalSystem(x -> [-x[1], -x[2]], 2)
        
        # Should succeed for actual fixed point
        @test_nowarn classify_fixed_point(ds, [0.0, 0.0])
        
        # Should throw for non-fixed point when verify=true
        @test_throws ArgumentError classify_fixed_point(ds, [1.0, 1.0]; verify=true)
        
        # Should not throw when verify=false
        @test_nowarn classify_fixed_point(ds, [1.0, 1.0]; verify=false)
    end
end

@testset "Latin Hypercube Sampling" begin
    
    @testset "Basic sampling" begin
        bounds = ((-2.0, 2.0), (-1.0, 1.0))
        n = 50
        points = latin_hypercube_sample(bounds, n)
        
        @test length(points) == n
        @test all(length(p) == 2 for p in points)
        
        # Check all points are within bounds
        for p in points
            @test -2.0 ≤ p[1] ≤ 2.0
            @test -1.0 ≤ p[2] ≤ 1.0
        end
    end
    
    @testset "Higher dimensions" begin
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        n = 100
        points = latin_hypercube_sample(bounds, n)
        
        @test length(points) == n
        @test all(length(p) == 4 for p in points)
    end
    
    @testset "Coverage property" begin
        # LHS should give good coverage - each interval sampled once
        bounds = ((0.0, 10.0),)
        n = 10
        points = latin_hypercube_sample(bounds, n)
        
        # Each interval [0,1), [1,2), ..., [9,10) should have exactly one point
        values = [p[1] for p in points]
        intervals = [floor(Int, v) for v in values]
        @test sort(intervals) == 0:9
    end
    
    @testset "Input validation" begin
        @test_throws ArgumentError latin_hypercube_sample(((-1.0, 1.0),), 0)
        @test_throws ArgumentError latin_hypercube_sample(((-1.0, 1.0),), -1)
        @test_throws ArgumentError latin_hypercube_sample(((1.0, -1.0),), 10)  # inverted bounds
    end
    
    @testset "DynamicalSystem interface" begin
        ds = DynamicalSystem(x -> -x, 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        points = latin_hypercube_sample(ds, bounds, 20)
        
        @test length(points) == 20
        
        # Dimension mismatch should throw
        bounds_3d = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        @test_throws DimensionMismatch latin_hypercube_sample(ds, bounds_3d, 10)
    end
end

@testset "Clustering" begin
    
    @testset "cluster_points basic" begin
        # Two distinct clusters
        points = [
            [0.0, 0.0], [0.01, 0.0], [0.0, 0.01],  # cluster 1 near origin
            [5.0, 5.0], [5.01, 5.0], [5.0, 5.01]   # cluster 2 near (5,5)
        ]
        
        centroids = cluster_points(points; distance_tol=0.1)
        @test length(centroids) == 2
        
        # Check centroids are near expected locations
        centroid_norms = sort([norm(c) for c in centroids])
        @test centroid_norms[1] < 0.1  # near origin
        @test abs(centroid_norms[2] - norm([5.0, 5.0])) < 0.1  # near (5,5)
    end
    
    @testset "unique_fixed_points" begin
        # Duplicate points should be merged
        candidates = [
            [1.0, 2.0], [1.0 + 1e-8, 2.0], [1.0, 2.0 + 1e-8],  # essentially same point
            [5.0, 6.0]  # different point
        ]
        
        unique_pts = unique_fixed_points(candidates; tol=1e-6)
        @test length(unique_pts) == 2
    end
    
    @testset "Empty input" begin
        @test isempty(cluster_points(Vector{Float64}[]))
        @test isempty(unique_fixed_points(Vector{Float64}[]))
    end
end

@testset "NLsolve Fixed Point Finding" begin
    
    @testset "Simple linear system" begin
        # dx/dt = -x has fixed point at origin
        ds = DynamicalSystem(x -> -x, 2)
        guesses = [[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5]]
        
        fps = find_fixed_points_nlsolve(ds, guesses)
        
        @test length(fps) == 1
        @test norm(fps[1]) < 1e-8
    end
    
    @testset "Multiple fixed points" begin
        # dx/dt = x(1-x²) has fixed points at x = -1, 0, 1
        ds = DynamicalSystem(x -> [x[1] * (1 - x[1]^2)], 1)
        guesses = [[-0.9], [0.1], [0.9], [-1.5], [1.5]]
        
        fps = find_fixed_points_nlsolve(ds, guesses)
        
        # Should find all three fixed points
        @test length(fps) == 3
        fp_values = sort([fp[1] for fp in fps])
        @test fp_values ≈ [-1.0, 0.0, 1.0] atol=1e-6
    end
    
    @testset "2D system with multiple fixed points" begin
        # System with fixed points at (0,0), (1,0), (0,1)
        ds = DynamicalSystem(x -> [x[1] * (1 - x[1] - x[2]), x[2] * (1 - x[1] - x[2])], 2)
        guesses = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
        
        fps = find_fixed_points_nlsolve(ds, guesses)
        
        # Should find at least the origin
        @test any(norm(fp) < 1e-6 for fp in fps)
    end
end

@testset "ODE Fixed Point Finding" begin
    
    @testset "Stable fixed point" begin
        # Simple attractor at origin
        ds = DynamicalSystem(x -> -x, 2)
        initial_conditions = [[1.0, 1.0], [-1.0, 2.0], [3.0, -1.0]]
        
        fps = find_fixed_points_ode(ds, initial_conditions; 
                                    tspan=(0.0, 100.0),
                                    stationarity_tol=1e-8)
        
        @test length(fps) == 1
        @test norm(fps[1]) < 1e-6
    end
    
    @testset "Multiple stable fixed points" begin
        # Bistable system: dx/dt = x - x³ has stable points at ±1
        ds = DynamicalSystem(x -> [x[1] - x[1]^3], 1)
        initial_conditions = [[-2.0], [-0.5], [0.5], [2.0]]
        
        fps = find_fixed_points_ode(ds, initial_conditions;
                                    tspan=(0.0, 100.0))
        
        # Should find both stable fixed points at ±1
        @test length(fps) == 2
        fp_values = sort([fp[1] for fp in fps])
        @test fp_values ≈ [-1.0, 1.0] atol=1e-4
    end
end

@testset "Backward Integration for Unstable Fixed Points" begin
    
    @testset "Find unstable fixed point" begin
        # dx/dt = x - x³ has unstable fixed point at 0
        ds = DynamicalSystem(x -> [x[1] - x[1]^3], 1)
        initial_conditions = [[0.1], [-0.1], [0.5], [-0.5]]
        
        fps = find_unstable_fixed_points_ode(ds, initial_conditions;
                                             tspan=(0.0, 100.0))
        
        # Should find the unstable fixed point at 0
        @test any(abs(fp[1]) < 1e-4 for fp in fps)
    end
    
    @testset "2D system with saddle - NLsolve" begin
        # System with saddle at origin: dx/dt = x, dy/dt = -y
        # Use NLsolve which is more reliable for saddles
        ds = DynamicalSystem(x -> [x[1], -x[2]], 2)
        guesses = [[0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]]
        
        fps = find_fixed_points_nlsolve(ds, guesses)
        
        # Should find the saddle at origin
        @test any(norm(fp) < 1e-4 for fp in fps)
    end
end

@testset "High-Level find_fixed_points" begin
    
    @testset "Simple 1D system" begin
        # dx/dt = x(1-x)(x-0.5) has fixed points at 0, 0.5, 1
        ds = DynamicalSystem(x -> [x[1] * (1 - x[1]) * (x[1] - 0.5)], 1)
        bounds = ((-0.5, 1.5),)
        
        # Use NLsolve method which is more reliable for this system
        fps = find_fixed_points(ds, bounds; n_samples=50, method=:nlsolve)
        
        @test length(fps) == 3
        locations = sort([fp.location[1] for fp in fps])
        @test locations ≈ [0.0, 0.5, 1.0] atol=1e-4
    end
    
    @testset "2D linear system" begin
        # Stable node at origin
        ds = DynamicalSystem(x -> [-x[1] - 0.5x[2], -0.5x[1] - x[2]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        # Use NLsolve method
        fps = find_fixed_points(ds, bounds; n_samples=30, method=:nlsolve)
        
        @test length(fps) == 1
        @test norm(fps[1].location) < 1e-6
        @test fps[1].is_stable
        @test fps[1].is_hyperbolic
    end
    
    @testset "Classification is correct" begin
        # Saddle point at origin
        ds = DynamicalSystem(x -> [x[1], -x[2]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        # Use NLsolve method
        fps = find_fixed_points(ds, bounds; n_samples=30, method=:nlsolve)
        
        @test length(fps) == 1
        @test fps[1].type == SADDLE
        @test !fps[1].is_stable
        @test fps[1].is_hyperbolic
    end
    
    @testset "Dimension mismatch throws" begin
        ds = DynamicalSystem(x -> -x, 2)
        bounds_3d = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        
        @test_throws DimensionMismatch find_fixed_points(ds, bounds_3d)
    end
end

@testset "Fixed Point Collection Functions" begin
    
    @testset "all_fixed_points_hyperbolic" begin
        fp1 = FixedPoint([0.0, 0.0], [ComplexF64(-1.0), ComplexF64(-2.0)], STABLE_NODE, true, true)
        fp2 = FixedPoint([1.0, 0.0], [ComplexF64(1.0), ComplexF64(-1.0)], SADDLE, false, true)
        fp3 = FixedPoint([0.0, 1.0], [ComplexF64(0.0, 1.0), ComplexF64(0.0, -1.0)], CENTER, false, false)
        
        @test all_fixed_points_hyperbolic([fp1, fp2])
        @test !all_fixed_points_hyperbolic([fp1, fp2, fp3])
        @test all_fixed_points_hyperbolic(FixedPoint[])  # empty case
    end
    
    @testset "count_fixed_point_types" begin
        fp1 = FixedPoint([0.0], [ComplexF64(-1.0)], STABLE_NODE, true, true)
        fp2 = FixedPoint([1.0], [ComplexF64(-2.0)], STABLE_NODE, true, true)
        fp3 = FixedPoint([0.5], [ComplexF64(1.0)], UNSTABLE_NODE, false, true)
        
        counts = count_fixed_point_types([fp1, fp2, fp3])
        
        @test counts[STABLE_NODE] == 2
        @test counts[UNSTABLE_NODE] == 1
    end
    
    @testset "stable_fixed_points and unstable_fixed_points" begin
        fp_stable = FixedPoint([0.0], [ComplexF64(-1.0)], STABLE_NODE, true, true)
        fp_unstable = FixedPoint([1.0], [ComplexF64(1.0)], UNSTABLE_NODE, false, true)
        fp_saddle = FixedPoint([0.5], [ComplexF64(-1.0), ComplexF64(1.0)], SADDLE, false, true)
        
        all_fps = [fp_stable, fp_unstable, fp_saddle]
        
        stable = stable_fixed_points(all_fps)
        @test length(stable) == 1
        @test stable[1].type == STABLE_NODE
        
        unstable = unstable_fixed_points(all_fps)
        @test length(unstable) == 2
        @test all(!fp.is_stable for fp in unstable)
    end
end

@testset "Integration Test: Lorenz System" begin
    # Test with the Lorenz system which has known fixed points
    function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
        return [σ * (x[2] - x[1]),
                x[1] * (ρ - x[3]) - x[2],
                x[1] * x[2] - β * x[3]]
    end
    
    ds = DynamicalSystem(lorenz, 3)
    
    # For ρ = 28 > 24.74, there are three fixed points:
    # Origin (0, 0, 0) - saddle
    # Two symmetric points at (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)
    
    # Use NLsolve with good initial guesses near known fixed points
    β = 8/3
    ρ = 28.0
    c = sqrt(β * (ρ - 1))  # ≈ 8.485
    
    guesses = [
        [0.0, 0.0, 0.0],      # Origin
        [c, c, ρ-1],           # C+ 
        [-c, -c, ρ-1],         # C-
        [0.1, 0.1, 0.1],       # Near origin
        [c+1, c+1, ρ],         # Near C+
        [-c-1, -c-1, ρ]        # Near C-
    ]
    
    fps_locations = find_fixed_points_nlsolve(ds, guesses)
    
    # Should find all three fixed points
    @test length(fps_locations) >= 3
    
    # Check that origin is found
    @test any(norm(fp) < 0.1 for fp in fps_locations)
    
    # Classify the fixed points
    fps = [classify_fixed_point(ds, fp; verify=true, verification_tol=1e-6) for fp in fps_locations]
    
    # All should be hyperbolic for ρ = 28
    @test all(fp.is_hyperbolic for fp in fps)
end
