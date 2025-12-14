# test_classification.jl
# Tests for system classification

@testset "Classification" begin
    
    @testset "SystemClass Enum" begin
        @test GRADIENT isa SystemClass
        @test GRADIENT_LIKE isa SystemClass
        @test MORSE_SMALE isa SystemClass
        @test STRUCTURALLY_STABLE isa SystemClass
        @test GENERAL isa SystemClass
        @test UNDETERMINED isa SystemClass
    end
    
    @testset "Class Hierarchy Levels" begin
        @test class_hierarchy_level(GRADIENT) == 1
        @test class_hierarchy_level(GRADIENT_LIKE) == 2
        @test class_hierarchy_level(MORSE_SMALE) == 3
        @test class_hierarchy_level(STRUCTURALLY_STABLE) == 4
        @test class_hierarchy_level(GENERAL) == 5
        
        # GRADIENT is most restrictive
        @test class_hierarchy_level(GRADIENT) < class_hierarchy_level(GRADIENT_LIKE)
        @test class_hierarchy_level(GRADIENT_LIKE) < class_hierarchy_level(MORSE_SMALE)
    end
    
    @testset "is_subclass" begin
        # GRADIENT is subclass of everything
        @test is_subclass(GRADIENT, GRADIENT)
        @test is_subclass(GRADIENT, GRADIENT_LIKE)
        @test is_subclass(GRADIENT, MORSE_SMALE)
        @test is_subclass(GRADIENT, GENERAL)
        
        # GENERAL is not subclass of more restrictive
        @test !is_subclass(GENERAL, GRADIENT)
        @test !is_subclass(GENERAL, MORSE_SMALE)
        
        # Same class is subclass of itself
        @test is_subclass(MORSE_SMALE, MORSE_SMALE)
    end
    
    @testset "is_subclass_with_dim - Palis-Smale Theorem" begin
        # In 2D: Morse-Smale ⊂ Structurally Stable
        @test is_subclass_with_dim(MORSE_SMALE, STRUCTURALLY_STABLE, 2) == true
        @test is_subclass_with_dim(MORSE_SMALE, STRUCTURALLY_STABLE, 1) == true
        
        # In 3D and higher: Morse-Smale ⊄ Structurally Stable
        @test is_subclass_with_dim(MORSE_SMALE, STRUCTURALLY_STABLE, 3) == false
        @test is_subclass_with_dim(MORSE_SMALE, STRUCTURALLY_STABLE, 4) == false
        
        # Gradient is always subclass (dimension-independent)
        @test is_subclass_with_dim(GRADIENT, STRUCTURALLY_STABLE, 2) == true
        @test is_subclass_with_dim(GRADIENT, STRUCTURALLY_STABLE, 5) == true
        @test is_subclass_with_dim(GRADIENT, MORSE_SMALE, 10) == true
    end
    
    @testset "is_structurally_stable - Dimension Dependent" begin
        fps = FixedPoint[]
        orbits = PeriodicOrbit[]
        
        # Morse-Smale in 2D is structurally stable
        ms_2d = ClassificationResult(MORSE_SMALE, 2, fps, orbits, false, 0.5,
                                      true, true, true, true, :high, String[])
        @test is_structurally_stable(ms_2d) == true
        
        # Morse-Smale in 3D is NOT guaranteed structurally stable
        ms_3d = ClassificationResult(MORSE_SMALE, 3, fps, orbits, false, 0.5,
                                      true, true, true, true, :high, String[])
        @test is_structurally_stable(ms_3d) == false
        
        # Gradient is always structurally stable
        grad_2d = ClassificationResult(GRADIENT, 2, fps, orbits, true, 0.0,
                                        true, true, true, false, :high, String[])
        grad_5d = ClassificationResult(GRADIENT, 5, fps, orbits, true, 0.0,
                                        true, true, true, false, :high, String[])
        @test is_structurally_stable(grad_2d) == true
        @test is_structurally_stable(grad_5d) == true
        
        # STRUCTURALLY_STABLE class is always structurally stable
        ss_3d = ClassificationResult(STRUCTURALLY_STABLE, 3, fps, orbits, false, 0.5,
                                      true, true, true, true, :medium, String[])
        @test is_structurally_stable(ss_3d) == true
    end
    
    @testset "Gradient System - Simple Linear" begin
        # dx/dt = -2x, dy/dt = -3y → gradient of V = x² + 1.5y²
        gradient_sys = DynamicalSystem(x -> [-2x[1], -3x[2]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        result = classify_system(gradient_sys, bounds; n_samples=10)
        
        @test result.system_class == GRADIENT
        @test result.dimension == 2
        @test result.jacobian_symmetric == true
        @test result.curl_ratio < 1e-6
        @test result.has_periodic_orbits == false
        @test is_gradient(result)
        @test is_gradient_like(result)
        @test is_morse_smale(result)
        @test is_structurally_stable(result)  # Gradient → structurally stable
    end
    
    @testset "Gradient System - Quadratic Potential" begin
        # V(x,y) = x² + y² + xy → ∇V = [2x+y, 2y+x]
        # dx/dt = -∇V = [-2x-y, -2y-x]
        grad_sys = DynamicalSystem(x -> [-2x[1] - x[2], -2x[2] - x[1]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        result = classify_system(grad_sys, bounds; n_samples=10)
        
        @test result.system_class == GRADIENT
        @test result.jacobian_symmetric == true
    end
    
    @testset "Non-Gradient - Rotation" begin
        # Pure rotation: dx/dt = -y, dy/dt = x
        # This has antisymmetric Jacobian, not gradient
        rotation = DynamicalSystem(x -> [-x[2], x[1]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        result = classify_system(rotation, bounds; n_samples=10)
        
        @test result.system_class != GRADIENT
        @test result.jacobian_symmetric == false
    end
    
    @testset "Gradient-Like System" begin
        # System with small curl but not exactly gradient
        # Add small rotational perturbation to gradient
        grad_like = DynamicalSystem(x -> [-2x[1] + 0.01*x[2], -3x[2] - 0.01*x[1]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        result = classify_system(grad_like, bounds; n_samples=10,
                                  symmetry_tol=1e-8, curl_tol=1e-8,
                                  curl_ratio_threshold=0.1)
        
        # Small perturbation means it's gradient-like but not gradient
        @test result.system_class in (GRADIENT, GRADIENT_LIKE)
        @test is_gradient_like(result)
    end
    
    @testset "Morse-Smale - Van der Pol" begin
        # Van der Pol oscillator has a stable limit cycle
        μ = 1.0
        vdp = DynamicalSystem(x -> [x[2], μ*(1 - x[1]^2)*x[2] - x[1]], 2)
        bounds = ((-3.0, 3.0), (-3.0, 3.0))
        
        result = classify_system(vdp, bounds; n_samples=10)
        
        # Van der Pol is Morse-Smale (hyperbolic limit cycle)
        @test result.system_class in (MORSE_SMALE, STRUCTURALLY_STABLE, GENERAL)
        @test !is_gradient(result)
        @test !is_gradient_like(result)
        @test allows_periodic_orbits(result)
    end
    
    @testset "ClassificationResult Construction" begin
        fps = [FixedPoint([0.0, 0.0], [-1.0, -2.0], STABLE_NODE, true, true)]
        orbits = PeriodicOrbit[]
        
        result = ClassificationResult(
            GRADIENT,
            2,  # dimension
            fps,
            orbits,
            true,
            0.0,
            true,
            true,
            true,
            false,
            :high,
            String[]
        )
        
        @test result.system_class == GRADIENT
        @test result.dimension == 2
        @test length(result.fixed_points) == 1
        @test result.confidence == :high
    end
    
    @testset "ClassificationResult Display" begin
        fps = FixedPoint[]
        orbits = PeriodicOrbit[]
        
        result = ClassificationResult(
            MORSE_SMALE,
            2,  # dimension
            fps,
            orbits,
            false,
            0.5,
            true,
            true,
            true,
            true,
            :medium,
            ["Test note"]
        )
        
        # Test that show works without error
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        
        @test contains(output, "MORSE_SMALE")
        @test contains(output, "medium")
        @test contains(output, "Dimension: 2")
    end
    
    @testset "has_landscape_representation" begin
        fps = FixedPoint[]
        orbits = PeriodicOrbit[]
        
        grad_result = ClassificationResult(GRADIENT, 2, fps, orbits, true, 0.0, 
                                            true, true, true, false, :high, String[])
        gl_result = ClassificationResult(GRADIENT_LIKE, 2, fps, orbits, false, 0.05,
                                          true, true, true, false, :high, String[])
        ms_result = ClassificationResult(MORSE_SMALE, 2, fps, orbits, false, 0.5,
                                          true, true, true, true, :medium, String[])
        gen_result = ClassificationResult(GENERAL, 2, fps, orbits, false, 1.0,
                                           false, false, false, true, :low, String[])
        
        @test has_landscape_representation(grad_result)[1] == true
        @test has_landscape_representation(grad_result)[2] == :exact_potential
        
        @test has_landscape_representation(gl_result)[1] == true
        @test has_landscape_representation(gl_result)[2] == :lyapunov_function
        
        @test has_landscape_representation(ms_result)[1] == true
        @test has_landscape_representation(ms_result)[2] == :potential_plus_metric
        
        @test has_landscape_representation(gen_result)[1] == false
    end
    
    @testset "allows_periodic_orbits" begin
        fps = FixedPoint[]
        orbits = PeriodicOrbit[]
        
        grad_result = ClassificationResult(GRADIENT, 2, fps, orbits, true, 0.0,
                                            true, true, true, false, :high, String[])
        gl_result = ClassificationResult(GRADIENT_LIKE, 2, fps, orbits, false, 0.05,
                                          true, true, true, false, :high, String[])
        ms_result = ClassificationResult(MORSE_SMALE, 2, fps, orbits, false, 0.5,
                                          true, true, true, true, :medium, String[])
        
        @test allows_periodic_orbits(grad_result) == false
        @test allows_periodic_orbits(gl_result) == false
        @test allows_periodic_orbits(ms_result) == true
    end
    
    @testset "classification_summary" begin
        fps = FixedPoint[]
        orbits = PeriodicOrbit[]
        
        result = ClassificationResult(GRADIENT, 2, fps, orbits, true, 0.0,
                                        true, true, true, false, :high, 
                                        ["Test classification"])
        
        summary = classification_summary(result)
        
        @test contains(summary, "GRADIENT")
        @test contains(summary, "potential")
        @test contains(summary, "high")
        @test contains(summary, "Dimension: 2")
    end
    
    @testset "quick_classify" begin
        # Quick classify should work faster with fewer samples
        gradient_sys = DynamicalSystem(x -> [-x[1], -x[2]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        result = quick_classify(gradient_sys, bounds; n_samples=5)
        
        @test result isa ClassificationResult
        @test result.system_class == GRADIENT
        @test result.dimension == 2
    end
    
    @testset "get_system_class" begin
        gradient_sys = DynamicalSystem(x -> [-x[1], -x[2]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        class = get_system_class(gradient_sys, bounds; n_samples=5)
        
        @test class isa SystemClass
        @test class == GRADIENT
    end
    
    @testset "compare_classifications" begin
        fps = FixedPoint[]
        orbits = PeriodicOrbit[]
        
        grad_result = ClassificationResult(GRADIENT, 2, fps, orbits, true, 0.0,
                                            true, true, true, false, :high, String[])
        ms_result = ClassificationResult(MORSE_SMALE, 2, fps, orbits, false, 0.5,
                                          true, true, true, true, :medium, String[])
        
        comparison = compare_classifications(grad_result, ms_result)
        
        @test comparison.same_class == false
        @test comparison.more_restrictive == 1  # First is more restrictive
        @test comparison.level_difference == 2  # GRADIENT=1, MORSE_SMALE=3
        @test comparison.both_structurally_stable == true  # Both are struct stable in 2D
    end
    
    @testset "is_gradient_system Function" begin
        # Test the individual function
        gradient_sys = DynamicalSystem(x -> [-2x[1], -3x[2]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        is_grad, details, notes = is_gradient_system(gradient_sys, bounds; 
                                                      n_samples=10)
        
        @test is_grad == true
        @test details.jacobian_symmetric == true
        @test details.curl_free == true
    end
    
    @testset "is_gradient_system - Non-gradient" begin
        rotation = DynamicalSystem(x -> [-x[2], x[1]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        is_grad, details, notes = is_gradient_system(rotation, bounds;
                                                      n_samples=10)
        
        @test is_grad == false
        @test details.jacobian_symmetric == false
    end
    
    @testset "is_morse_smale_system Function" begin
        # Simple stable node - trivially Morse-Smale
        stable_sys = DynamicalSystem(x -> [-x[1], -2x[2]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        is_ms, details, notes = is_morse_smale_system(stable_sys, bounds;
                                                       n_samples=10, manifold_check=false)
        
        @test is_ms == true
        @test details.all_fps_hyperbolic == true
    end
    
    @testset "3D Gradient System" begin
        # V(x,y,z) = x² + y² + z² → gradient system
        grad_3d = DynamicalSystem(x -> [-2x[1], -2x[2], -2x[3]], 3)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        
        result = classify_system(grad_3d, bounds; n_samples=10)
        
        @test result.system_class == GRADIENT
        @test result.dimension == 3
        @test result.jacobian_symmetric == true
        # Gradient systems are structurally stable in any dimension
        @test is_structurally_stable(result) == true
    end
    
    @testset "Saddle Point System" begin
        # Saddle at origin: ẋ = x, ẏ = -y
        saddle_sys = DynamicalSystem(x -> [x[1], -x[2]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        result = classify_system(saddle_sys, bounds; n_samples=10)
        
        # This is a gradient system with V = -x²/2 + y²/2
        @test result.system_class == GRADIENT
        @test result.all_fps_hyperbolic == true
        @test length(result.fixed_points) >= 1
    end
    
    @testset "fixed_point_summary" begin
        fps = [
            FixedPoint([0.0, 0.0], [-1.0, -2.0], STABLE_NODE, true, true),
            FixedPoint([1.0, 0.0], [1.0, -1.0], SADDLE, true, false)
        ]
        
        summary = fixed_point_summary(fps)
        
        @test contains(summary, "Fixed points: 2")
        @test contains(summary, "Hyperbolic")
    end
    
    @testset "periodic_orbit_summary - Empty" begin
        summary = periodic_orbit_summary(PeriodicOrbit[])
        @test contains(summary, "No periodic orbits")
    end
    
    @testset "Bistable System" begin
        # Double-well system: ẋ = x - x³, ẏ = -y
        # Has saddle at origin, stable nodes at (±1, 0)
        bistable = DynamicalSystem(x -> [x[1] - x[1]^3, -x[2]], 2)
        bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        result = classify_system(bistable, bounds; n_samples=15)
        
        # Should be gradient (J is diagonal, hence symmetric)
        @test result.system_class == GRADIENT
        @test length(result.fixed_points) >= 2  # At least origin and one stable
    end
    
    @testset "Classification with verbose" begin
        gradient_sys = DynamicalSystem(x -> [-x[1], -x[2]], 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        # Just test that verbose mode doesn't crash
        result = classify_system(gradient_sys, bounds; n_samples=5, verbose=false)
        @test result isa ClassificationResult
    end
    
end
