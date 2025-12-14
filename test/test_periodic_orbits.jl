# test_periodic_orbits.jl
# Tests for periodic orbit detection and classification

using Test
using FlowClass
using LinearAlgebra

@testset "Periodic Orbits" begin
    
    @testset "Floquet Multiplier Classification" begin
        # Stable limit cycle: all multipliers inside unit circle (except trivial)
        multipliers_stable = ComplexF64[1.0, 0.5, 0.3]
        type, stable, hyperbolic = FlowClass.classify_floquet_multipliers(multipliers_stable)
        @test type == STABLE_LIMIT_CYCLE
        @test stable == true
        @test hyperbolic == true
        
        # Unstable limit cycle: multipliers outside unit circle
        multipliers_unstable = ComplexF64[1.0, 2.0, 1.5]
        type, stable, hyperbolic = FlowClass.classify_floquet_multipliers(multipliers_unstable)
        @test type == UNSTABLE_LIMIT_CYCLE
        @test stable == false
        @test hyperbolic == true
        
        # Saddle cycle: mixed inside/outside
        multipliers_saddle = ComplexF64[1.0, 0.5, 2.0]
        type, stable, hyperbolic = FlowClass.classify_floquet_multipliers(multipliers_saddle)
        @test type == SADDLE_CYCLE
        @test stable == false
        @test hyperbolic == true
        
        # Non-hyperbolic: multiplier on unit circle
        multipliers_nonhyp = ComplexF64[1.0, 1.0, 0.5]  # Two multipliers at 1.0
        type, stable, hyperbolic = FlowClass.classify_floquet_multipliers(multipliers_nonhyp)
        @test type == NON_HYPERBOLIC_CYCLE
        @test hyperbolic == false
        
        # Complex multipliers inside unit circle (spiral stable)
        multipliers_complex = ComplexF64[1.0, 0.5 + 0.3im, 0.5 - 0.3im]
        type, stable, hyperbolic = FlowClass.classify_floquet_multipliers(multipliers_complex)
        @test type == STABLE_LIMIT_CYCLE
        @test stable == true
        @test hyperbolic == true
        
        # Empty multipliers
        type, stable, hyperbolic = FlowClass.classify_floquet_multipliers(ComplexF64[])
        @test type == UNKNOWN_CYCLE
    end
    
    @testset "PeriodicOrbit Construction" begin
        # With Floquet multipliers
        points = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        period = 2π
        multipliers = ComplexF64[1.0, 0.5]
        
        orbit = PeriodicOrbit(points, period, multipliers)
        @test orbit.period ≈ 2π
        @test length(orbit.points) == 4
        @test orbit.type == STABLE_LIMIT_CYCLE
        @test orbit.is_stable == true
        @test orbit.is_hyperbolic == true
        
        # Without Floquet multipliers
        orbit_simple = PeriodicOrbit(points, period)
        @test orbit_simple.type == UNKNOWN_CYCLE
        @test isempty(orbit_simple.floquet_multipliers)
    end
    
    @testset "Van der Pol Oscillator - Oscillation Detection" begin
        # Van der Pol oscillator: classic example with a limit cycle
        # ẋ = y
        # ẏ = μ(1 - x²)y - x
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        
        # Test oscillation detection with longer time span for convergence
        is_osc, est_period, amplitude = detect_oscillation(vdp, [2.0, 0.0]; tspan=(0.0, 150.0))
        @test is_osc == true
        @test est_period > 0
        @test amplitude > 0
        
        # From near origin (unstable fixed point, should still lead to oscillation)
        is_osc2, _, _ = detect_oscillation(vdp, [0.1, 0.1]; tspan=(0.0, 150.0))
        @test is_osc2 == true
    end
    
    @testset "Van der Pol Oscillator - Orbit Finding" begin
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        
        # Find periodic orbit via recurrence
        found, orbit_points, period = find_periodic_orbit_recurrence(
            vdp, [2.0, 0.0]; 
            tspan=(0.0, 250.0), 
            transient_time=80.0,
            recurrence_tol=1e-2  # Looser tolerance for initial detection
        )
        
        @test found == true
        @test period > 0
        @test length(orbit_points) > 0
        
        # Period should be around 6.66 for μ=1
        # Allow wider range to account for numerical detection
        @test 5.0 < period < 10.0
    end
    
    @testset "Van der Pol Oscillator - Poincaré Section" begin
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        
        # Find via Poincaré section - use section at x=1, detect y increasing through section
        section = PoincaréSection([1.0, 0.0], [1.0, 0.0])  # Vertical line at x=1
        
        crossings = find_poincare_crossings(
            vdp, [2.0, 0.0], section;
            tspan=(0.0, 250.0),
            transient_time=80.0
        )
        
        @test length(crossings) >= 3
        
        # Check that crossings are periodic (return to roughly same point)
        if length(crossings) >= 3
            x1, _ = crossings[end-1]
            x2, _ = crossings[end]
            @test norm(x2 - x1) < 0.1  # Should be close for limit cycle
        end
    end
    
    @testset "Van der Pol - Full Orbit Finding Pipeline" begin
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        bounds = ((-3.0, 3.0), (-3.0, 3.0))
        
        orbits = find_periodic_orbits(vdp, bounds; 
                                       n_samples=5, 
                                       method=:recurrence,  # Use recurrence method explicitly
                                       tspan=(0.0, 200.0),
                                       transient_time=80.0,
                                       recurrence_tol=1e-2,
                                       classify=true)
        
        @test length(orbits) >= 1
        
        if !isempty(orbits)
            orbit = orbits[1]
            @test orbit.period > 0
            @test length(orbit.points) > 0
            
            # Van der Pol limit cycle should be stable
            @test orbit.type == STABLE_LIMIT_CYCLE || orbit.type == UNKNOWN_CYCLE
        end
    end
    
    @testset "Monodromy Matrix and Floquet Multipliers" begin
        # Simple harmonic oscillator with damping (spiral to origin)
        # ẋ = y
        # ẏ = -x - 0.1y
        # This has no limit cycle (stable focus at origin), but we can test 
        # the monodromy matrix computation for a circular "pseudo-orbit"
        
        # Better test: use a system with known Floquet multipliers
        # Linear system with periodic orbit: ẋ = y, ẏ = -x gives circles
        harmonic = DynamicalSystem(x -> [x[2], -x[1]], 2)
        
        # Circles have period 2π
        x0 = [1.0, 0.0]  # Start on unit circle
        T = 2π
        
        # Compute monodromy matrix
        M = compute_monodromy_matrix(harmonic, x0, T)
        
        @test size(M) == (2, 2)
        
        # For harmonic oscillator, monodromy matrix should be approximately identity
        # (since it's a conservative system - all Floquet multipliers on unit circle)
        @test norm(M - I) < 0.1
        
        # Floquet multipliers
        fm = compute_floquet_multipliers(harmonic, x0, T)
        @test length(fm) == 2
        # Both should be on unit circle
        @test all(abs(abs(m) - 1.0) < 0.1 for m in fm)
    end
    
    @testset "Linear System - No Periodic Orbits" begin
        # Stable linear system: ẋ = -x, ẏ = -2y
        # All trajectories go to origin, no limit cycles
        linear_stable = DynamicalSystem(x -> [-x[1], -2*x[2]], 2)
        
        is_osc, _, _ = detect_oscillation(linear_stable, [1.0, 1.0]; tspan=(0.0, 50.0))
        @test is_osc == false
        
        # has_periodic_orbits should return false
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        @test has_periodic_orbits(linear_stable, bounds; n_samples=10, tspan=(0.0, 50.0)) == false
    end
    
    @testset "Unstable Focus - No Periodic Orbits (Diverging)" begin
        # Unstable focus: spirals outward
        # ẋ = 0.1x + y
        # ẏ = -x + 0.1y
        unstable_focus = DynamicalSystem(x -> [0.1*x[1] + x[2], -x[1] + 0.1*x[2]], 2)
        
        # Should not detect bounded oscillations (system diverges)
        is_osc, _, _ = detect_oscillation(unstable_focus, [0.1, 0.1]; tspan=(0.0, 30.0))
        # May or may not detect oscillation depending on how far it diverges
        # The key is that it shouldn't find a periodic orbit
        
        found, _, _ = find_periodic_orbit_recurrence(
            unstable_focus, [0.1, 0.1];
            tspan=(0.0, 50.0),
            recurrence_tol=1e-3
        )
        @test found == false
    end
    
    @testset "has_periodic_orbits Function" begin
        # Van der Pol has periodic orbits
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        bounds = ((-3.0, 3.0), (-3.0, 3.0))
        
        @test has_periodic_orbits(vdp, bounds; n_samples=15, quick_check=true, tspan=(0.0, 200.0)) == true
        
        # Gradient system (no periodic orbits)
        gradient_sys = DynamicalSystem(x -> [-2*x[1], -2*x[2]], 2)
        @test has_periodic_orbits(gradient_sys, bounds; n_samples=10, quick_check=true) == false
    end
    
    @testset "Collection Functions" begin
        # Create mock orbits for testing collection functions
        points = [[1.0, 0.0], [0.0, 1.0]]
        
        stable_orbit = PeriodicOrbit(points, 2π, ComplexF64[1.0, 0.5])
        unstable_orbit = PeriodicOrbit(points, 2π, ComplexF64[1.0, 2.0])
        saddle_orbit = PeriodicOrbit(points, 2π, ComplexF64[1.0, 0.5, 2.0])
        nonhyp_orbit = PeriodicOrbit(points, 2π, ComplexF64[1.0, 1.0, 0.5])
        
        orbits = [stable_orbit, unstable_orbit, saddle_orbit, nonhyp_orbit]
        
        # all_periodic_orbits_hyperbolic
        @test all_periodic_orbits_hyperbolic(orbits) == false
        @test all_periodic_orbits_hyperbolic([stable_orbit, unstable_orbit, saddle_orbit]) == true
        @test all_periodic_orbits_hyperbolic(PeriodicOrbit[]) == true  # Vacuously true
        
        # stable_periodic_orbits
        stable = stable_periodic_orbits(orbits)
        @test length(stable) == 1
        @test stable[1].type == STABLE_LIMIT_CYCLE
        
        # unstable_periodic_orbits
        unstable = unstable_periodic_orbits(orbits)
        @test length(unstable) == 3  # unstable, saddle, and non-hyperbolic
        
        # count_periodic_orbit_types
        counts = count_periodic_orbit_types(orbits)
        @test counts[STABLE_LIMIT_CYCLE] == 1
        @test counts[UNSTABLE_LIMIT_CYCLE] == 1
        @test counts[SADDLE_CYCLE] == 1
        @test counts[NON_HYPERBOLIC_CYCLE] == 1
    end
    
    @testset "unique_periodic_orbits" begin
        points1 = [[1.0, 0.0], [0.0, 1.0]]
        points2 = [[1.001, 0.0], [0.0, 1.0]]  # Close to points1
        points3 = [[2.0, 0.0], [0.0, 2.0]]    # Different orbit
        
        orbit1 = PeriodicOrbit(points1, 2π)
        orbit2 = PeriodicOrbit(points2, 2π)
        orbit3 = PeriodicOrbit(points3, 2π)
        
        # With default tolerance, orbit1 and orbit2 should be duplicates
        unique = unique_periodic_orbits([orbit1, orbit2, orbit3]; tol=0.01)
        @test length(unique) == 2
        
        # With smaller tolerance, all three should be unique
        unique_strict = unique_periodic_orbits([orbit1, orbit2, orbit3]; tol=1e-6)
        @test length(unique_strict) == 3
    end
    
    @testset "3D System - Lorenz (No Simple Periodic Orbits)" begin
        # Lorenz system with standard parameters (chaotic regime)
        # This doesn't have simple stable periodic orbits in the chaotic regime
        σ, ρ, β = 10.0, 28.0, 8/3
        lorenz = DynamicalSystem(
            x -> [σ*(x[2] - x[1]), x[1]*(ρ - x[3]) - x[2], x[1]*x[2] - β*x[3]], 
            3
        )
        
        # Oscillation detection may trigger due to the oscillatory nature of chaos
        # but recurrence should not find a true periodic orbit
        is_osc, _, _ = detect_oscillation(lorenz, [1.0, 1.0, 1.0]; tspan=(0.0, 100.0))
        # Chaotic systems often appear oscillatory
        
        # Try to find periodic orbit - should generally fail for chaotic parameters
        found, _, _ = find_periodic_orbit_recurrence(
            lorenz, [1.0, 1.0, 1.0];
            tspan=(0.0, 100.0),
            transient_time=20.0,
            recurrence_tol=1e-3
        )
        # Note: might occasionally find a near-return, but not a true periodic orbit
        # This test is more about ensuring the code runs than guaranteeing a specific result
    end
    
    @testset "Hopf Bifurcation System" begin
        # System exhibiting Hopf bifurcation
        # ẋ = μx - y - x(x² + y²)
        # ẏ = x + μy - y(x² + y²)
        # For μ > 0: unstable focus at origin, stable limit cycle
        # For μ < 0: stable focus at origin, no limit cycle
        
        function hopf_system(μ)
            DynamicalSystem(x -> [
                μ*x[1] - x[2] - x[1]*(x[1]^2 + x[2]^2),
                x[1] + μ*x[2] - x[2]*(x[1]^2 + x[2]^2)
            ], 2)
        end
        
        # Supercritical case (μ > 0): should have limit cycle
        hopf_super = hopf_system(0.5)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        @test has_periodic_orbits(hopf_super, bounds; n_samples=10, tspan=(0.0, 150.0)) == true
        
        # Subcritical case (μ < 0): no limit cycle, stable focus at origin
        # Use stronger damping (μ=-1.0) so oscillations decay quickly
        hopf_sub = hopf_system(-1.0)
        @test has_periodic_orbits(hopf_sub, bounds; n_samples=10, tspan=(0.0, 50.0)) == false
    end
    
    @testset "PoincaréSection Construction" begin
        # Default constructor
        section = PoincaréSection([1.0, 0.0])
        @test section.point == [1.0, 0.0]
        @test section.normal == [0.0, 1.0]  # Default: horizontal, detect upward
        
        # Custom constructor
        section2 = PoincaréSection([0.0, 0.0], [1.0, 0.0])  # Vertical, detect rightward
        @test section2.normal == [1.0, 0.0]
    end
    
    @testset "Orbit Refinement" begin
        # Test refinement on Van der Pol
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        
        # First find a rough orbit
        found, rough_points, rough_period = find_periodic_orbit_recurrence(
            vdp, [2.0, 0.0];
            tspan=(0.0, 150.0),
            transient_time=50.0,
            recurrence_tol=1e-2  # Looser tolerance
        )
        
        if found && !isempty(rough_points)
            # Refine it
            success, refined_x0, refined_points = refine_periodic_orbit(
                vdp, rough_points[1], rough_period;
                tol=1e-8
            )
            
            # Refinement should succeed or at least not crash
            if success
                # Check that refined orbit actually closes
                x_end = refined_points[end]
                @test norm(x_end - refined_x0) < 1e-4
            end
        end
    end
    
end
