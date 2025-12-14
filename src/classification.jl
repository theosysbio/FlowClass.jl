# classification.jl
# High-level classification of dynamical systems into the hierarchy:
# Gradient ⊂ Gradient-like ⊂ Morse-Smale ⊂ Structurally Stable ⊂ General
#
# IMPORTANT: Morse-Smale ⟹ Structurally Stable holds only in dimensions 1 and 2
# (Palis-Smale theorem). In dimension ≥ 3, Morse-Smale systems are NOT necessarily
# structurally stable.

#=============================================================================
                            System Class Enum
=============================================================================#

"""
    SystemClass

Enumeration of dynamical system classes, ordered from most restrictive to most general.

- `GRADIENT`: dx/dt = -∇V(x) for some potential V. Symmetric Jacobian, zero curl, no periodic orbits.
- `GRADIENT_LIKE`: Has a global Lyapunov function. Small curl, no periodic orbits.
- `MORSE_SMALE`: Hyperbolic fixed points and periodic orbits, transverse manifold intersections.
  Note: Morse-Smale implies structural stability only in dimensions 1 and 2.
- `STRUCTURALLY_STABLE`: Qualitative behaviour unchanged under small perturbations.
- `GENERAL`: No special structure guaranteed. May exhibit chaos, strange attractors, etc.
- `UNDETERMINED`: Classification could not be completed (e.g., numerical issues).
"""
@enum SystemClass begin
    GRADIENT
    GRADIENT_LIKE
    MORSE_SMALE
    STRUCTURALLY_STABLE
    GENERAL
    UNDETERMINED
end

#=============================================================================
                        Classification Result Type
=============================================================================#

"""
    ClassificationResult

Detailed result of classifying a dynamical system.

# Fields
- `system_class::SystemClass`: The determined class
- `dimension::Int`: Dimension of the system (important for hierarchy interpretation)
- `fixed_points::Vector{FixedPoint}`: All detected fixed points
- `periodic_orbits::Vector{PeriodicOrbit}`: All detected periodic orbits
- `jacobian_symmetric::Bool`: Whether the Jacobian is symmetric (gradient condition)
- `curl_ratio::Float64`: Ratio of curl to gradient component magnitude
- `all_fps_hyperbolic::Bool`: Whether all fixed points are hyperbolic
- `all_orbits_hyperbolic::Bool`: Whether all periodic orbits are hyperbolic
- `manifolds_transverse::Bool`: Whether all manifold intersections are transverse
- `has_periodic_orbits::Bool`: Whether any periodic orbits were detected
- `confidence::Symbol`: Confidence level (:high, :medium, :low)
- `notes::Vector{String}`: Explanatory notes about the classification
"""
struct ClassificationResult
    system_class::SystemClass
    dimension::Int
    fixed_points::Vector{FixedPoint}
    periodic_orbits::Vector{PeriodicOrbit}
    jacobian_symmetric::Bool
    curl_ratio::Float64
    all_fps_hyperbolic::Bool
    all_orbits_hyperbolic::Bool
    manifolds_transverse::Bool
    has_periodic_orbits::Bool
    confidence::Symbol
    notes::Vector{String}
end

function Base.show(io::IO, cr::ClassificationResult)
    println(io, "ClassificationResult:")
    println(io, "  System class: ", cr.system_class)
    println(io, "  Dimension: ", cr.dimension)
    println(io, "  Confidence: ", cr.confidence)
    println(io, "  Fixed points: ", length(cr.fixed_points))
    println(io, "  Periodic orbits: ", length(cr.periodic_orbits))
    println(io, "  Jacobian symmetric: ", cr.jacobian_symmetric)
    println(io, "  Curl/gradient ratio: ", round(cr.curl_ratio, digits=4))
    println(io, "  All FPs hyperbolic: ", cr.all_fps_hyperbolic)
    println(io, "  All orbits hyperbolic: ", cr.all_orbits_hyperbolic)
    println(io, "  Manifolds transverse: ", cr.manifolds_transverse)
    if !isempty(cr.notes)
        println(io, "  Notes:")
        for note in cr.notes
            println(io, "    - ", note)
        end
    end
end

#=============================================================================
                        Individual Classification Tests
=============================================================================#

"""
    is_gradient_system(ds::DynamicalSystem, bounds; n_samples=20, 
                       symmetry_tol=1e-6, curl_tol=1e-6)

Test if a system is a gradient system (dx/dt = -∇V).

A gradient system has:
1. Symmetric Jacobian everywhere: J = Jᵀ
2. Zero curl everywhere: ∇ × F = 0
3. No periodic orbits (implied by 1 & 2)

Returns `(is_gradient, details)` where details is a NamedTuple with test results.
"""
function is_gradient_system(ds::DynamicalSystem, bounds::Tuple;
                            n_samples::Int=20,
                            symmetry_tol::Float64=1e-6,
                            curl_tol::Float64=1e-6)
    
    notes = String[]
    
    # Sample points for testing
    samples = latin_hypercube_sample(bounds, n_samples)
    
    # Test 1: Jacobian symmetry
    symmetric = true
    max_asym = 0.0
    for x in samples
        J = compute_jacobian(ds, x)
        if !is_jacobian_symmetric(J; atol=symmetry_tol)
            symmetric = false
            asym = jacobian_symmetry_error(J)
            max_asym = max(max_asym, asym)
        end
    end
    
    if !symmetric
        push!(notes, "Jacobian not symmetric (max error: $(round(max_asym, digits=6)))")
    end
    
    # Test 2: Zero curl
    curl_free = true
    max_curl = 0.0
    for x in samples
        if !is_curl_free(ds, x; atol=curl_tol)
            curl_free = false
            c = curl_magnitude(ds, x)
            max_curl = max(max_curl, c)
        end
    end
    
    if !curl_free
        push!(notes, "Non-zero curl detected (max: $(round(max_curl, digits=6)))")
    end
    
    # If Jacobian is symmetric AND curl is zero, this IS a gradient system
    # Gradient systems mathematically cannot have periodic orbits, so skip that check
    is_grad = symmetric && curl_free
    
    details = (
        jacobian_symmetric = symmetric,
        curl_free = curl_free,
        has_periodic_orbits = false,  # Impossible for gradient systems
        max_asymmetry = max_asym,
        max_curl = max_curl
    )
    
    return (is_grad, details, notes)
end

"""
    is_gradient_like_system(ds::DynamicalSystem, bounds; n_samples=20,
                            curl_ratio_threshold=0.1)

Test if a system is gradient-like (has a global Lyapunov function).

A gradient-like system has:
1. Nearly symmetric Jacobian (small antisymmetric part)
2. Small curl relative to gradient component
3. No periodic orbits
4. All fixed points hyperbolic

Returns `(is_gradient_like, details)`.
"""
function is_gradient_like_system(ds::DynamicalSystem, bounds::Tuple;
                                  n_samples::Int=20,
                                  curl_ratio_threshold::Float64=0.1)
    
    notes = String[]
    
    # Sample points for testing
    samples = latin_hypercube_sample(bounds, n_samples)
    
    # Test 1: Curl to gradient ratio
    ratios = Float64[]
    for x in samples
        r = curl_to_gradient_ratio(ds, x)
        if !isnan(r) && !isinf(r)
            push!(ratios, r)
        end
    end
    
    mean_ratio = isempty(ratios) ? Inf : mean(ratios)
    max_ratio = isempty(ratios) ? Inf : maximum(ratios)
    
    small_curl = max_ratio < curl_ratio_threshold
    
    if !small_curl
        push!(notes, "Curl/gradient ratio too large (max: $(round(max_ratio, digits=4)))")
    end
    
    # Test 2: No periodic orbits - use a quick check
    # For gradient-like systems, if curl is small, orbits are unlikely
    # Skip expensive orbit search if curl is very small
    has_orbits = false
    if max_ratio > 0.01  # Only check for orbits if there's meaningful curl
        has_orbits = has_periodic_orbits(ds, bounds; n_samples=min(n_samples, 5))
        if has_orbits
            push!(notes, "Periodic orbits detected - not gradient-like")
        end
    end
    
    # Test 3: All fixed points hyperbolic
    fps = find_fixed_points(ds, bounds; n_samples=n_samples)
    all_hyp = all_fixed_points_hyperbolic(fps)
    
    if !all_hyp
        push!(notes, "Non-hyperbolic fixed points detected")
    end
    
    is_grad_like = small_curl && !has_orbits && all_hyp
    
    details = (
        mean_curl_ratio = mean_ratio,
        max_curl_ratio = max_ratio,
        has_periodic_orbits = has_orbits,
        all_fps_hyperbolic = all_hyp,
        fixed_points = fps
    )
    
    return (is_grad_like, details, notes)
end

"""
    is_morse_smale_system(ds::DynamicalSystem, bounds; n_samples=20, manifold_check=true)

Test if a system is Morse-Smale.

A Morse-Smale system has:
1. Finitely many fixed points, all hyperbolic
2. Finitely many periodic orbits, all hyperbolic
3. All stable/unstable manifolds intersect transversely
4. All trajectories converge to a fixed point or periodic orbit

Returns `(is_morse_smale, details)`.
"""
function is_morse_smale_system(ds::DynamicalSystem, bounds::Tuple;
                                n_samples::Int=20,
                                manifold_check::Bool=true)
    
    notes = String[]
    
    # Test 1: Find and check fixed points
    fps = find_fixed_points(ds, bounds; n_samples=n_samples)
    all_fps_hyp = all_fixed_points_hyperbolic(fps)
    
    if !all_fps_hyp
        push!(notes, "Non-hyperbolic fixed points detected")
    end
    
    # Test 2: Find and check periodic orbits
    orbits = find_periodic_orbits(ds, bounds; n_samples=min(n_samples, 10), classify=true)
    all_orbits_hyp = all_periodic_orbits_hyperbolic(orbits)
    
    if !all_orbits_hyp && !isempty(orbits)
        push!(notes, "Non-hyperbolic periodic orbits detected")
    end
    
    # Test 3: Transversality of manifolds (only for saddles)
    manifolds_trans = true
    if manifold_check
        saddles = filter(fp -> fp.type == SADDLE || fp.type == SADDLE_FOCUS, fps)
        if !isempty(saddles)
            manifolds_trans = all_manifolds_transverse(ds, saddles; 
                                                        epsilon=0.1, max_extent=1.0)
            if !manifolds_trans
                push!(notes, "Non-transverse manifold intersections detected")
            end
        end
    end
    
    is_ms = all_fps_hyp && all_orbits_hyp && manifolds_trans
    
    details = (
        fixed_points = fps,
        periodic_orbits = orbits,
        all_fps_hyperbolic = all_fps_hyp,
        all_orbits_hyperbolic = all_orbits_hyp,
        manifolds_transverse = manifolds_trans
    )
    
    return (is_ms, details, notes)
end

"""
    is_structurally_stable(ds::DynamicalSystem, bounds; n_samples=20)

Test if a system is structurally stable.

Structural stability requires:
1. All fixed points hyperbolic
2. All periodic orbits hyperbolic
3. Transverse manifold intersections
4. No saddle connections forming cycles

This is similar to Morse-Smale but allows more complex omega-limit sets.

Returns `(is_stable, details)`.
"""
function is_structurally_stable(ds::DynamicalSystem, bounds::Tuple;
                                 n_samples::Int=20)
    
    notes = String[]
    
    # For now, structural stability check is similar to Morse-Smale
    # A more complete check would require additional analysis
    
    fps = find_fixed_points(ds, bounds; n_samples=n_samples)
    all_fps_hyp = all_fixed_points_hyperbolic(fps)
    
    if !all_fps_hyp
        push!(notes, "Non-hyperbolic fixed points violate structural stability")
    end
    
    # Check for periodic orbits
    has_orbits = has_periodic_orbits(ds, bounds; n_samples=min(n_samples, 10))
    
    orbits = PeriodicOrbit[]
    all_orbits_hyp = true
    if has_orbits
        orbits = find_periodic_orbits(ds, bounds; n_samples=min(n_samples, 10), classify=true)
        all_orbits_hyp = all_periodic_orbits_hyperbolic(orbits)
        if !all_orbits_hyp
            push!(notes, "Non-hyperbolic periodic orbits violate structural stability")
        end
    end
    
    is_stable = all_fps_hyp && all_orbits_hyp
    
    details = (
        fixed_points = fps,
        periodic_orbits = orbits,
        all_fps_hyperbolic = all_fps_hyp,
        all_orbits_hyperbolic = all_orbits_hyp
    )
    
    return (is_stable, details, notes)
end

#=============================================================================
                        Main Classification Function
=============================================================================#

"""
    classify_system(ds::DynamicalSystem, bounds; n_samples=20, symmetry_tol=1e-6,
                    curl_tol=1e-6, curl_ratio_threshold=0.1, check_manifolds=true, 
                    verbose=false)

Classify a dynamical system into the hierarchy:
Gradient ⊂ Gradient-like ⊂ Morse-Smale ⊂ Structurally Stable ⊂ General

The classification proceeds from most restrictive to most general,
returning the most specific class that the system belongs to.

# Arguments
- `ds::DynamicalSystem`: The system to classify
- `bounds::Tuple`: Search bounds as tuple of (min, max) pairs per dimension

# Keyword Arguments
- `n_samples::Int=20`: Number of sample points for testing
- `symmetry_tol::Float64=1e-6`: Tolerance for Jacobian symmetry
- `curl_tol::Float64=1e-6`: Tolerance for curl being zero
- `curl_ratio_threshold::Float64=0.1`: Max curl/gradient ratio for gradient-like
- `check_manifolds::Bool=true`: Whether to check manifold transversality
- `verbose::Bool=false`: Print progress information

# Returns
A `ClassificationResult` containing the classification and supporting details.

# Example
```julia
# Simple gradient system
ds = DynamicalSystem(x -> [-2x[1], -3x[2]], 2)
result = classify_system(ds, ((-2.0, 2.0), (-2.0, 2.0)))
result.system_class  # GRADIENT
```
"""
function classify_system(ds::DynamicalSystem, bounds::Tuple;
                          n_samples::Int=20,
                          symmetry_tol::Float64=1e-6,
                          curl_tol::Float64=1e-6,
                          curl_ratio_threshold::Float64=0.1,
                          check_manifolds::Bool=true,
                          verbose::Bool=false)
    
    all_notes = String[]
    confidence = :high
    dim = dimension(ds)
    
    verbose && println("Classifying $(dim)D system...")
    
    # Step 1: Test for gradient system
    verbose && println("  Testing gradient conditions...")
    is_grad, grad_details, grad_notes = is_gradient_system(ds, bounds;
                                                            n_samples=n_samples,
                                                            symmetry_tol=symmetry_tol,
                                                            curl_tol=curl_tol)
    append!(all_notes, grad_notes)
    
    if is_grad
        verbose && println("  → GRADIENT system")
        fps = find_fixed_points(ds, bounds; n_samples=n_samples)
        
        return ClassificationResult(
            GRADIENT,
            dim,
            fps,
            PeriodicOrbit[],
            true,
            0.0,
            all_fixed_points_hyperbolic(fps),
            true,  # No orbits to be non-hyperbolic
            true,  # N/A for gradient
            false,
            confidence,
            all_notes
        )
    end
    
    # Step 2: Test for gradient-like system
    verbose && println("  Testing gradient-like conditions...")
    is_grad_like, gl_details, gl_notes = is_gradient_like_system(ds, bounds;
                                                                   n_samples=n_samples,
                                                                   curl_ratio_threshold=curl_ratio_threshold)
    append!(all_notes, gl_notes)
    
    if is_grad_like
        verbose && println("  → GRADIENT_LIKE system")
        
        return ClassificationResult(
            GRADIENT_LIKE,
            dim,
            gl_details.fixed_points,
            PeriodicOrbit[],
            grad_details.jacobian_symmetric,
            gl_details.max_curl_ratio,
            gl_details.all_fps_hyperbolic,
            true,
            true,
            false,
            confidence,
            all_notes
        )
    end
    
    # Step 3: Test for Morse-Smale system
    verbose && println("  Testing Morse-Smale conditions...")
    is_ms, ms_details, ms_notes = is_morse_smale_system(ds, bounds;
                                                         n_samples=n_samples,
                                                         manifold_check=check_manifolds)
    append!(all_notes, ms_notes)
    
    # Compute curl ratio if not already done
    samples = latin_hypercube_sample(bounds, min(n_samples, 10))
    ratios = [curl_to_gradient_ratio(ds, x) for x in samples]
    ratios = filter(r -> !isnan(r) && !isinf(r), ratios)
    curl_ratio = isempty(ratios) ? NaN : mean(ratios)
    
    if is_ms
        verbose && println("  → MORSE_SMALE system")
        
        # Important: Morse-Smale implies structural stability only in dim < 3
        if dim >= 3
            push!(all_notes, "Morse-Smale does NOT imply structural stability in dimension ≥ 3 (Palis-Smale theorem)")
        else
            push!(all_notes, "Morse-Smale implies structural stability in dimension < 3 (Palis-Smale theorem)")
        end
        
        return ClassificationResult(
            MORSE_SMALE,
            dim,
            ms_details.fixed_points,
            ms_details.periodic_orbits,
            grad_details.jacobian_symmetric,
            curl_ratio,
            ms_details.all_fps_hyperbolic,
            ms_details.all_orbits_hyperbolic,
            ms_details.manifolds_transverse,
            !isempty(ms_details.periodic_orbits),
            confidence,
            all_notes
        )
    end
    
    # Step 4: Test for structural stability
    verbose && println("  Testing structural stability...")
    is_stable, stable_details, stable_notes = is_structurally_stable(ds, bounds;
                                                                       n_samples=n_samples)
    append!(all_notes, stable_notes)
    
    if is_stable
        verbose && println("  → STRUCTURALLY_STABLE system")
        confidence = :medium  # Less certain without full manifold analysis
        
        return ClassificationResult(
            STRUCTURALLY_STABLE,
            dim,
            stable_details.fixed_points,
            stable_details.periodic_orbits,
            grad_details.jacobian_symmetric,
            curl_ratio,
            stable_details.all_fps_hyperbolic,
            stable_details.all_orbits_hyperbolic,
            ms_details.manifolds_transverse,
            !isempty(stable_details.periodic_orbits),
            confidence,
            all_notes
        )
    end
    
    # Step 5: General system
    verbose && println("  → GENERAL system")
    push!(all_notes, "System does not fit more restrictive classes")
    
    # Gather what information we have
    fps = ms_details.fixed_points
    orbits = ms_details.periodic_orbits
    
    return ClassificationResult(
        GENERAL,
        dim,
        fps,
        orbits,
        grad_details.jacobian_symmetric,
        curl_ratio,
        all_fixed_points_hyperbolic(fps),
        all_periodic_orbits_hyperbolic(orbits),
        ms_details.manifolds_transverse,
        !isempty(orbits),
        :medium,
        all_notes
    )
end

#=============================================================================
                        Quick Classification Functions
=============================================================================#

"""
    quick_classify(ds::DynamicalSystem, bounds; n_samples=10)

Perform a quick classification with minimal computation.

Uses fewer samples and skips expensive manifold checks.
Good for initial exploration or large parameter sweeps.
"""
function quick_classify(ds::DynamicalSystem, bounds::Tuple; n_samples::Int=10)
    return classify_system(ds, bounds; 
                           n_samples=n_samples, 
                           check_manifolds=false)
end

"""
    get_system_class(ds::DynamicalSystem, bounds; n_samples=20)

Get just the system class without full details.

Returns a `SystemClass` enum value.
"""
function get_system_class(ds::DynamicalSystem, bounds::Tuple; n_samples::Int=20)
    result = classify_system(ds, bounds; n_samples=n_samples)
    return result.system_class
end

#=============================================================================
                        Classification Predicates
=============================================================================#

"""
    is_gradient(result::ClassificationResult)

Check if the classification result indicates a gradient system.
"""
is_gradient(result::ClassificationResult) = result.system_class == GRADIENT

"""
    is_gradient_like(result::ClassificationResult)

Check if the classification result indicates a gradient-like system.

Note: Gradient systems are also gradient-like.
"""
is_gradient_like(result::ClassificationResult) = 
    result.system_class in (GRADIENT, GRADIENT_LIKE)

"""
    is_morse_smale(result::ClassificationResult)

Check if the classification result indicates a Morse-Smale system.

Note: Gradient and gradient-like systems are also Morse-Smale.
"""
is_morse_smale(result::ClassificationResult) = 
    result.system_class in (GRADIENT, GRADIENT_LIKE, MORSE_SMALE)

"""
    is_structurally_stable(result::ClassificationResult)

Check if the classification result indicates a structurally stable system.

Important: Morse-Smale systems are structurally stable only in dimensions 1 and 2.
In dimension ≥ 3, a system classified as MORSE_SMALE is NOT guaranteed to be 
structurally stable. This function accounts for dimension.
"""
function is_structurally_stable(result::ClassificationResult)
    # Gradient and gradient-like are always structurally stable
    if result.system_class in (GRADIENT, GRADIENT_LIKE)
        return true
    end
    
    # Morse-Smale implies structural stability only in dim < 3
    if result.system_class == MORSE_SMALE
        return result.dimension < 3
    end
    
    # STRUCTURALLY_STABLE class
    if result.system_class == STRUCTURALLY_STABLE
        return true
    end
    
    return false
end

"""
    allows_periodic_orbits(result::ClassificationResult)

Check if the system class allows periodic orbits.

Gradient and gradient-like systems cannot have periodic orbits.
"""
allows_periodic_orbits(result::ClassificationResult) = 
    result.system_class ∉ (GRADIENT, GRADIENT_LIKE)

"""
    has_landscape_representation(result::ClassificationResult)

Check if the system has a valid landscape (potential) representation.

Gradient systems have an exact potential.
Gradient-like systems have a Lyapunov function.
Morse-Smale can have a potential + metric tensor representation.
"""
function has_landscape_representation(result::ClassificationResult)
    if result.system_class == GRADIENT
        return (true, :exact_potential, "V(x) such that dx/dt = -∇V")
    elseif result.system_class == GRADIENT_LIKE
        return (true, :lyapunov_function, "Lyapunov function exists")
    elseif result.system_class == MORSE_SMALE
        return (true, :potential_plus_metric, "V(x) + metric tensor G(x)")
    else
        return (false, :none, "No global landscape representation")
    end
end

#=============================================================================
                        Summary and Reporting Functions
=============================================================================#

"""
    classification_summary(result::ClassificationResult)

Generate a human-readable summary of the classification result.
"""
function classification_summary(result::ClassificationResult)
    lines = String[]
    
    push!(lines, "═" ^ 50)
    push!(lines, "DYNAMICAL SYSTEM CLASSIFICATION")
    push!(lines, "═" ^ 50)
    push!(lines, "")
    push!(lines, "Dimension: $(result.dimension)")
    push!(lines, "")
    
    class_descriptions = Dict(
        GRADIENT => "GRADIENT SYSTEM\nMost restrictive class. Has potential function V(x)\nsuch that dx/dt = -∇V(x). Detailed balance holds.",
        GRADIENT_LIKE => "GRADIENT-LIKE SYSTEM\nHas global Lyapunov function. Small rotational\ncomponent. No periodic orbits possible.",
        MORSE_SMALE => "MORSE-SMALE SYSTEM\nHyperbolic fixed points and periodic orbits with\ntransverse manifold intersections. Permits limit cycles.",
        STRUCTURALLY_STABLE => "STRUCTURALLY STABLE SYSTEM\nQualitative behaviour robust to perturbations.\nMay have complex attractors.",
        GENERAL => "GENERAL DYNAMICAL SYSTEM\nNo special structure guaranteed. May exhibit\nchaos, strange attractors, or other complex dynamics.",
        UNDETERMINED => "UNDETERMINED\nClassification could not be completed."
    )
    
    push!(lines, class_descriptions[result.system_class])
    push!(lines, "")
    push!(lines, "─" ^ 50)
    push!(lines, "PROPERTIES")
    push!(lines, "─" ^ 50)
    push!(lines, "  Dimension: $(result.dimension)")
    push!(lines, "  Fixed points found: $(length(result.fixed_points))")
    push!(lines, "  Periodic orbits found: $(length(result.periodic_orbits))")
    push!(lines, "  Jacobian symmetric: $(result.jacobian_symmetric)")
    push!(lines, "  Curl/gradient ratio: $(round(result.curl_ratio, digits=4))")
    push!(lines, "  All FPs hyperbolic: $(result.all_fps_hyperbolic)")
    push!(lines, "  All orbits hyperbolic: $(result.all_orbits_hyperbolic)")
    push!(lines, "  Manifolds transverse: $(result.manifolds_transverse)")
    push!(lines, "")
    push!(lines, "─" ^ 50)
    push!(lines, "IMPLICATIONS")
    push!(lines, "─" ^ 50)
    
    has_landscape, type, desc = has_landscape_representation(result)
    push!(lines, "  Landscape representation: $(has_landscape ? "Yes ($type)" : "No")")
    push!(lines, "  Allows periodic orbits: $(allows_periodic_orbits(result))")
    
    # Structural stability depends on dimension for Morse-Smale systems
    struct_stable = is_structurally_stable(result)
    if result.system_class == MORSE_SMALE
        if result.dimension < 3
            push!(lines, "  Structurally stable: Yes (Palis-Smale theorem, dim < 3)")
        else
            push!(lines, "  Structurally stable: Not guaranteed (dim ≥ 3)")
        end
    else
        push!(lines, "  Structurally stable: $(struct_stable)")
    end
    
    push!(lines, "  Long-term behaviour: $(result.system_class in (GRADIENT, GRADIENT_LIKE) ? 
                                           "All trajectories → fixed points" :
                                           result.system_class == MORSE_SMALE ?
                                           "All trajectories → fixed points or limit cycles" :
                                           "May have complex omega-limit sets")")
    
    if !isempty(result.notes)
        push!(lines, "")
        push!(lines, "─" ^ 50)
        push!(lines, "NOTES")
        push!(lines, "─" ^ 50)
        for note in result.notes
            push!(lines, "  • $note")
        end
    end
    
    push!(lines, "")
    push!(lines, "═" ^ 50)
    push!(lines, "Confidence: $(result.confidence)")
    push!(lines, "═" ^ 50)
    
    return join(lines, "\n")
end

"""
    print_classification(result::ClassificationResult)

Print the classification summary to stdout.
"""
function print_classification(result::ClassificationResult)
    println(classification_summary(result))
end

#=============================================================================
                    Fixed Point Summary for Classification
=============================================================================#

"""
    fixed_point_summary(fps::Vector{FixedPoint})

Generate a summary of fixed points for classification reporting.
"""
function fixed_point_summary(fps::Vector{FixedPoint})
    if isempty(fps)
        return "No fixed points found"
    end
    
    type_counts = count_fixed_point_types(fps)
    
    lines = String[]
    push!(lines, "Fixed points: $(length(fps))")
    for (type, count) in type_counts
        push!(lines, "  - $type: $count")
    end
    
    n_hyperbolic = count(fp -> fp.is_hyperbolic, fps)
    n_stable = count(fp -> fp.is_stable, fps)
    push!(lines, "  Hyperbolic: $n_hyperbolic / $(length(fps))")
    push!(lines, "  Stable: $n_stable / $(length(fps))")
    
    return join(lines, "\n")
end

"""
    periodic_orbit_summary(orbits::Vector{PeriodicOrbit})

Generate a summary of periodic orbits for classification reporting.
"""
function periodic_orbit_summary(orbits::Vector{PeriodicOrbit})
    if isempty(orbits)
        return "No periodic orbits found"
    end
    
    type_counts = count_periodic_orbit_types(orbits)
    
    lines = String[]
    push!(lines, "Periodic orbits: $(length(orbits))")
    for (type, count) in type_counts
        push!(lines, "  - $type: $count")
    end
    
    n_hyperbolic = count(o -> o.is_hyperbolic, orbits)
    n_stable = count(o -> o.is_stable, orbits)
    push!(lines, "  Hyperbolic: $n_hyperbolic / $(length(orbits))")
    push!(lines, "  Stable: $n_stable / $(length(orbits))")
    
    periods = [o.period for o in orbits]
    push!(lines, "  Periods: $(round.(periods, digits=3))")
    
    return join(lines, "\n")
end

#=============================================================================
                    Comparison and Hierarchy Functions
=============================================================================#

"""
    class_hierarchy_level(class::SystemClass)

Return the hierarchy level of a system class (lower = more restrictive).

Note: This returns a simple ordering that does NOT account for dimension.
In dimension ≥ 3, MORSE_SMALE and STRUCTURALLY_STABLE are not comparable
via this ordering alone. Use `is_subclass_with_dim` for dimension-aware checks.
"""
function class_hierarchy_level(class::SystemClass)
    levels = Dict(
        GRADIENT => 1,
        GRADIENT_LIKE => 2,
        MORSE_SMALE => 3,
        STRUCTURALLY_STABLE => 4,
        GENERAL => 5,
        UNDETERMINED => 6
    )
    return levels[class]
end

"""
    is_subclass(class1::SystemClass, class2::SystemClass)

Check if class1 is a subclass of (more restrictive than) class2.

For example, GRADIENT is a subclass of GRADIENT_LIKE.

Warning: This function does NOT account for dimension. In dimension ≥ 3, 
MORSE_SMALE is NOT a subclass of STRUCTURALLY_STABLE. Use `is_subclass_with_dim`
for dimension-aware comparisons.
"""
function is_subclass(class1::SystemClass, class2::SystemClass)
    return class_hierarchy_level(class1) <= class_hierarchy_level(class2)
end

"""
    is_subclass_with_dim(class1::SystemClass, class2::SystemClass, dim::Int)

Check if class1 is a subclass of class2, accounting for dimension.

The key dimension-dependent relationship:
- dim < 3: MORSE_SMALE ⊂ STRUCTURALLY_STABLE (Palis-Smale theorem)
- dim ≥ 3: MORSE_SMALE and STRUCTURALLY_STABLE are not comparable

# Examples
```julia
is_subclass_with_dim(MORSE_SMALE, STRUCTURALLY_STABLE, 2)  # true
is_subclass_with_dim(MORSE_SMALE, STRUCTURALLY_STABLE, 3)  # false
is_subclass_with_dim(GRADIENT, MORSE_SMALE, 5)             # true (always)
```
"""
function is_subclass_with_dim(class1::SystemClass, class2::SystemClass, dim::Int)
    # Gradient and Gradient-like are always subclasses of everything above them
    if class1 in (GRADIENT, GRADIENT_LIKE)
        return class_hierarchy_level(class1) <= class_hierarchy_level(class2)
    end
    
    # The key case: MORSE_SMALE → STRUCTURALLY_STABLE only in dim < 3
    if class1 == MORSE_SMALE && class2 == STRUCTURALLY_STABLE
        return dim < 3
    end
    
    # For other comparisons, use standard hierarchy
    return class_hierarchy_level(class1) <= class_hierarchy_level(class2)
end

"""
    compare_classifications(result1::ClassificationResult, result2::ClassificationResult)

Compare two classification results.

Returns a NamedTuple with comparison information.
Note: Uses dimension from result1 for dimension-dependent comparisons.
"""
function compare_classifications(result1::ClassificationResult, result2::ClassificationResult)
    level1 = class_hierarchy_level(result1.system_class)
    level2 = class_hierarchy_level(result2.system_class)
    
    return (
        same_class = result1.system_class == result2.system_class,
        more_restrictive = level1 < level2 ? 1 : (level1 > level2 ? 2 : 0),
        level_difference = abs(level1 - level2),
        both_allow_orbits = allows_periodic_orbits(result1) && allows_periodic_orbits(result2),
        both_have_landscape = has_landscape_representation(result1)[1] && 
                             has_landscape_representation(result2)[1],
        both_structurally_stable = is_structurally_stable(result1) && is_structurally_stable(result2)
    )
end
