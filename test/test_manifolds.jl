# test_manifolds.jl
# Tests for manifold computation and transversality analysis

@testset "Manifolds" begin
    
    @testset "ManifoldType Enum" begin
        @test STABLE_MANIFOLD isa ManifoldType
        @test UNSTABLE_MANIFOLD isa ManifoldType
        @test STABLE_MANIFOLD != UNSTABLE_MANIFOLD
    end
    
    @testset "Manifold Construction" begin
        points = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]
        tangent = [[1.0, 0.0]]
        fp = [0.0, 0.0]
        
        m = Manifold(points, STABLE_MANIFOLD, 1, fp, tangent)
        
        @test m.manifold_type == STABLE_MANIFOLD
        @test m.dimension == 1
        @test m.fixed_point == fp
        @test length(m.points) == 3
        @test length(m.tangent_vectors) == 1
    end
    
    @testset "ManifoldIntersection Construction" begin
        point = [0.5, 0.5]
        t1 = [[1.0, 0.0]]
        t2 = [[0.0, 1.0]]
        
        int = ManifoldIntersection(point, t1, t2, true, 0)
        
        @test int.point == point
        @test int.is_transverse == true
        @test int.codimension_deficit == 0
    end
    
    @testset "Eigenspace Computation - 2D Saddle" begin
        # Simple saddle: ẋ = x, ẏ = -y
        # Eigenvalues: 1 (unstable), -1 (stable)
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        x0 = [0.0, 0.0]
        
        stable_eigs, stable_vecs = compute_stable_eigenspace(saddle, x0)
        unstable_eigs, unstable_vecs = compute_unstable_eigenspace(saddle, x0)
        
        @test length(stable_eigs) == 1
        @test length(unstable_eigs) == 1
        @test real(stable_eigs[1]) < 0
        @test real(unstable_eigs[1]) > 0
        @test length(stable_vecs) == 1
        @test length(unstable_vecs) == 1
        
        # Stable eigenvector should be along y-axis
        @test abs(stable_vecs[1][1]) < 0.1 || abs(stable_vecs[1][2]) > 0.9
        
        # Unstable eigenvector should be along x-axis
        @test abs(unstable_vecs[1][1]) > 0.9 || abs(unstable_vecs[1][2]) < 0.1
    end
    
    @testset "Eigenspace Computation - Stable Node" begin
        # Stable node: ẋ = -x, ẏ = -2y
        stable = DynamicalSystem(x -> [-x[1], -2x[2]], 2)
        x0 = [0.0, 0.0]
        
        stable_eigs, stable_vecs = compute_stable_eigenspace(stable, x0)
        unstable_eigs, unstable_vecs = compute_unstable_eigenspace(stable, x0)
        
        # All eigenvalues are stable
        @test length(stable_eigs) == 2
        @test length(unstable_eigs) == 0
        @test all(real.(stable_eigs) .< 0)
        @test length(stable_vecs) == 2
        @test length(unstable_vecs) == 0
    end
    
    @testset "Eigenspace Computation - Unstable Focus" begin
        # Unstable focus: ẋ = x - y, ẏ = x + y (eigenvalues 1 ± i)
        focus = DynamicalSystem(x -> [x[1] - x[2], x[1] + x[2]], 2)
        x0 = [0.0, 0.0]
        
        stable_eigs, stable_vecs = compute_stable_eigenspace(focus, x0)
        unstable_eigs, unstable_vecs = compute_unstable_eigenspace(focus, x0)
        
        # All eigenvalues unstable (Re > 0)
        @test length(stable_eigs) == 0
        @test length(unstable_eigs) == 2
        @test length(stable_vecs) == 0
        # For complex conjugate pair, we get 2 real basis vectors
        @test length(unstable_vecs) == 2
    end
    
    @testset "Extract Real Basis - Real Eigenvalues" begin
        # Two real eigenvalues
        eigenvalues = [-1.0 + 0im, 2.0 + 0im]
        eigenvectors = [1.0 0.0; 0.0 1.0] .+ 0im
        
        basis = extract_real_basis(eigenvectors, eigenvalues)
        
        @test length(basis) == 2
        @test all(v -> eltype(v) <: Real, basis)
    end
    
    @testset "Extract Real Basis - Complex Conjugate Pair" begin
        # Complex conjugate eigenvalues
        λ = 1.0 + 2.0im
        eigenvalues = [λ, conj(λ)]
        v = [1.0 + 0.5im, 0.5 - 0.5im]
        eigenvectors = hcat(v, conj(v))
        
        basis = extract_real_basis(eigenvectors, eigenvalues)
        
        # Should get 2 real basis vectors (real and imaginary parts)
        @test length(basis) == 2
        @test all(v -> eltype(v) <: Real, basis)
    end
    
    @testset "2D Saddle - Stable Manifold" begin
        # Simple saddle at origin: ẋ = x, ẏ = -y
        # Stable manifold is the y-axis, unstable manifold is the x-axis
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        
        # Create a FixedPoint
        fp = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        stable = compute_stable_manifold(saddle, fp; epsilon=0.1, max_extent=1.0)
        
        @test stable.manifold_type == STABLE_MANIFOLD
        @test stable.dimension == 1
        @test length(stable.points) > 2
        
        # Points should be approximately on the y-axis (x ≈ 0)
        for p in stable.points
            @test abs(p[1]) < 0.2  # x coordinate should be small
        end
    end
    
    @testset "2D Saddle - Unstable Manifold" begin
        # Simple saddle at origin: ẋ = x, ẏ = -y
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        unstable = compute_unstable_manifold(saddle, fp; epsilon=0.1, max_extent=1.0)
        
        @test unstable.manifold_type == UNSTABLE_MANIFOLD
        @test unstable.dimension == 1
        @test length(unstable.points) > 2
        
        # Points should be approximately on the x-axis (y ≈ 0)
        for p in unstable.points
            @test abs(p[2]) < 0.2  # y coordinate should be small
        end
    end
    
    @testset "Compute Both Manifolds" begin
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        stable, unstable = compute_manifolds(saddle, fp; epsilon=0.1, max_extent=1.0)
        
        @test stable.manifold_type == STABLE_MANIFOLD
        @test unstable.manifold_type == UNSTABLE_MANIFOLD
        @test stable.dimension == 1
        @test unstable.dimension == 1
    end
    
    @testset "Stable Node - Full Stable Manifold" begin
        # Stable node: all trajectories converge
        stable_node = DynamicalSystem(x -> [-x[1], -2x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [-1.0, -2.0], STABLE_NODE, true, true)
        
        stable = compute_stable_manifold(stable_node, fp; epsilon=0.1, max_extent=1.0)
        unstable = compute_unstable_manifold(stable_node, fp; epsilon=0.1, max_extent=1.0)
        
        @test stable.dimension == 2  # Full 2D stable manifold
        @test unstable.dimension == 0  # No unstable manifold
        @test length(stable.points) > 0
        @test length(unstable.points) == 0
    end
    
    @testset "Transversality Check - Orthogonal Manifolds" begin
        # Two 1D manifolds in 2D, intersecting orthogonally
        t1 = [[1.0, 0.0]]  # Along x-axis
        t2 = [[0.0, 1.0]]  # Along y-axis
        
        is_trans, deficit = check_transversality_at_point(t1, t2, 2)
        
        @test is_trans == true
        @test deficit == 0
    end
    
    @testset "Transversality Check - Parallel Manifolds" begin
        # Two 1D manifolds in 2D, parallel (non-transverse)
        t1 = [[1.0, 0.0]]
        t2 = [[1.0, 0.0]]
        
        is_trans, deficit = check_transversality_at_point(t1, t2, 2)
        
        @test is_trans == false
        @test deficit > 0
    end
    
    @testset "Transversality Check - 2D Manifolds in 3D" begin
        # Two 2D manifolds in 3D, transverse intersection (1D line)
        t1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # xy-plane
        t2 = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]  # xz-plane
        
        is_trans, deficit = check_transversality_at_point(t1, t2, 3)
        
        @test is_trans == true
        @test deficit == 0
    end
    
    @testset "Transversality Check - Non-transverse in 3D" begin
        # Two 2D manifolds in 3D, same plane (non-transverse)
        t1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # xy-plane
        t2 = [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]  # also xy-plane
        
        is_trans, deficit = check_transversality_at_point(t1, t2, 3)
        
        @test is_trans == false
        @test deficit > 0
    end
    
    @testset "Find Manifold Intersections - Simple Case" begin
        # Create two manifolds that clearly intersect
        points1 = [[0.0, y] for y in -1.0:0.1:1.0]  # Vertical line at x=0
        points2 = [[x, 0.0] for x in -1.0:0.1:1.0]  # Horizontal line at y=0
        
        tangent1 = [[0.0, 1.0]]
        tangent2 = [[1.0, 0.0]]
        fp1 = [0.0, -1.0]
        fp2 = [-1.0, 0.0]
        
        m1 = Manifold(points1, STABLE_MANIFOLD, 1, fp1, tangent1)
        m2 = Manifold(points2, UNSTABLE_MANIFOLD, 1, fp2, tangent2)
        
        intersections = find_manifold_intersections(m1, m2; distance_tol=0.15)
        
        @test length(intersections) >= 1
        
        # Intersection should be near origin
        @test any(int -> norm(int.point) < 0.2, intersections)
        
        # Should be transverse
        @test all(int -> int.is_transverse, intersections)
    end
    
    @testset "Separatrices in 2D" begin
        # Simple saddle at origin
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        stable_branches, unstable_branches = compute_separatrices(saddle, fp; 
                                                                   epsilon=0.1, max_extent=1.0)
        
        @test length(stable_branches) == 2
        @test length(unstable_branches) == 2
        
        # Each branch should have multiple points
        for branch in stable_branches
            @test length(branch) > 2
        end
        for branch in unstable_branches
            @test length(branch) > 2
        end
    end
    
    @testset "Manifold to Coordinates - 2D" begin
        points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        m = Manifold(points, STABLE_MANIFOLD, 1, [0.0, 0.0], [[1.0, 0.0]])
        
        xs, ys = manifold_to_coordinates(m)
        
        @test xs == [1.0, 3.0, 5.0]
        @test ys == [2.0, 4.0, 6.0]
    end
    
    @testset "Manifold to Coordinates - 3D" begin
        points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        m = Manifold(points, UNSTABLE_MANIFOLD, 1, [0.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])
        
        xs, ys, zs = manifold_to_coordinates(m)
        
        @test xs == [1.0, 4.0]
        @test ys == [2.0, 5.0]
        @test zs == [3.0, 6.0]
    end
    
    @testset "Manifold to Coordinates - Empty" begin
        m = Manifold(Vector{Float64}[], STABLE_MANIFOLD, 0, [0.0, 0.0], Vector{Float64}[])
        
        xs, ys = manifold_to_coordinates(m)
        
        @test isempty(xs)
        @test isempty(ys)
    end
    
    @testset "3D Saddle Point Manifolds" begin
        # 3D system with saddle at origin
        # ẋ = x, ẏ = -y, ż = -2z
        # 1D unstable (x), 2D stable (y-z plane)
        saddle_3d = DynamicalSystem(x -> [x[1], -x[2], -2x[3]], 3)
        
        _, stable_vecs = compute_stable_eigenspace(saddle_3d, [0.0, 0.0, 0.0])
        _, unstable_vecs = compute_unstable_eigenspace(saddle_3d, [0.0, 0.0, 0.0])
        
        @test length(stable_vecs) == 2  # 2D stable manifold
        @test length(unstable_vecs) == 1  # 1D unstable manifold
        
        # Create fixed point and compute manifolds
        fp = FixedPoint([0.0, 0.0, 0.0], [1.0, -1.0, -2.0], SADDLE, true, true)
        
        stable = compute_stable_manifold(saddle_3d, fp; epsilon=0.1, max_extent=1.0)
        unstable = compute_unstable_manifold(saddle_3d, fp; epsilon=0.1, max_extent=1.0)
        
        @test stable.dimension == 2
        @test unstable.dimension == 1
    end
    
    @testset "Homoclinic Orbit Detection - No Homoclinic" begin
        # Simple linear saddle - no homoclinic connection
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        has_hom, points = has_homoclinic_orbit(saddle, fp; 
                                                epsilon=0.1, max_extent=1.0, distance_tol=0.2)
        
        @test has_hom == false
        @test isempty(points)
    end
    
    @testset "Heteroclinic Connection Detection" begin
        # Simple test that the function works correctly
        # Using a linear saddle - no actual heteroclinic possible
        saddle_sys = DynamicalSystem(x -> [x[1], -x[2]], 2)
        
        # Saddle at origin
        saddle0 = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        # Another "fixed point" far away (for testing purposes)
        saddle1 = FixedPoint([5.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        # Test that the function runs and returns valid results
        has_het, points = has_heteroclinic_orbit(saddle_sys, saddle0, saddle1;
                                                   epsilon=0.1, max_extent=1.0, 
                                                   distance_tol=0.5)
        
        # Just verify it returns the right types
        @test has_het isa Bool
        @test points isa Vector
    end
    
    @testset "all_manifolds_transverse - Single Saddle" begin
        # Simple linear saddle - trivially transverse (no non-trivial intersections)
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        result = all_manifolds_transverse(saddle, [fp]; epsilon=0.1, max_extent=1.0)
        
        @test result == true
    end
    
    @testset "check_transversality - Single Saddle" begin
        # Simple linear saddle
        saddle_sys = DynamicalSystem(x -> [x[1], -x[2]], 2)
        saddle0 = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        # Quick check with minimal parameters
        all_trans, intersections = check_transversality(saddle_sys, [saddle0];
                                                          epsilon=0.1, max_extent=1.0, 
                                                          distance_tol=0.2)
        
        # Single saddle with no homoclinic orbit - should be transverse
        @test all_trans == true
        @test intersections isa Vector
    end
    
    @testset "Estimate Tangent Space" begin
        # Create a manifold along x-axis
        points = [[t, 0.0] for t in 0.0:0.1:1.0]
        m = Manifold(points, STABLE_MANIFOLD, 1, [0.0, 0.0], [[1.0, 0.0]])
        
        tangent = estimate_tangent_space(m, [0.5, 0.0]; radius=0.3)
        
        @test length(tangent) == 1
        # Tangent should be approximately along x-axis
        @test abs(tangent[1][1]) > 0.9 || abs(tangent[1][2]) < 0.1
    end
    
    @testset "Local Manifold Computation" begin
        # Simple saddle
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = [0.0, 0.0]
        tangent_vec = [[0.0, 1.0]]  # y-direction (stable)
        
        points = compute_local_manifold(saddle, fp, tangent_vec, STABLE_MANIFOLD;
                                         epsilon=0.1, n_points_per_direction=5,
                                         integration_time=1.0)
        
        @test length(points) > 1
        # Points should stay near y-axis
        for p in points
            @test abs(p[1]) < 0.5
        end
    end
    
    @testset "Grow 1D Manifold" begin
        # Simple unstable direction
        saddle = DynamicalSystem(x -> [x[1], -x[2]], 2)
        fp = [0.0, 0.0]
        direction = [1.0, 0.0]  # x-direction (unstable)
        
        points = grow_manifold_1d(saddle, fp, direction, UNSTABLE_MANIFOLD;
                                   epsilon=0.1, max_arclength=1.0)
        
        @test length(points) > 2
        # Points should stay near x-axis
        for p in points
            @test abs(p[2]) < 0.3
        end
    end
    
    @testset "Count Heteroclinic Connections" begin
        # Simple linear saddle
        saddle_sys = DynamicalSystem(x -> [x[1], -x[2]], 2)
        saddle0 = FixedPoint([0.0, 0.0], [1.0, -1.0], SADDLE, true, true)
        
        connections = count_heteroclinic_connections(saddle_sys, [saddle0];
                                                      epsilon=0.1, max_extent=1.0,
                                                      distance_tol=0.2)
        
        # Single saddle - no heteroclinic connections possible
        @test connections isa Dict
        @test isempty(connections)
    end
    
    @testset "Stable Focus - Only Stable Manifold" begin
        # Stable focus: spiral in
        focus = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
        fp = FixedPoint([0.0, 0.0], [-1.0 + 1.0im, -1.0 - 1.0im], STABLE_FOCUS, true, true)
        
        stable = compute_stable_manifold(focus, fp; epsilon=0.1, max_extent=1.0)
        unstable = compute_unstable_manifold(focus, fp; epsilon=0.1, max_extent=1.0)
        
        @test stable.dimension == 2  # Full plane
        @test unstable.dimension == 0  # No unstable directions
    end
    
end
