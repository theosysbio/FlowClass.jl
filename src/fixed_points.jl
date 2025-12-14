"""
    fixed_points.jl

Functions for finding and classifying fixed points (stationary points) of dynamical systems.

Provides two complementary approaches:
1. Root finding using NLsolve.jl (fast, can find unstable points, needs good guesses)
2. ODE integration to stationarity (robust for stable points, uses clustering)

Also includes backward integration to find unstable fixed points.
"""

using LinearAlgebra
using NLsolve
using OrdinaryDiffEq
using Statistics

#=============================================================================
    Fixed Point Type
=============================================================================#

"""
    FixedPointType

Enumeration of fixed point types based on eigenvalue analysis.
"""
@enum FixedPointType begin
    STABLE_NODE          # All eigenvalues real and negative
    UNSTABLE_NODE        # All eigenvalues real and positive
    SADDLE               # Real eigenvalues of mixed sign
    STABLE_FOCUS         # Complex eigenvalues with negative real parts
    UNSTABLE_FOCUS       # Complex eigenvalues with positive real parts
    SADDLE_FOCUS         # Mixed: some complex, mixed stability
    CENTER               # Pure imaginary eigenvalues (non-hyperbolic)
    NON_HYPERBOLIC       # At least one eigenvalue with zero real part
    UNKNOWN              # Classification failed
end


"""
    FixedPoint

Represents a fixed point of a dynamical system with its classification.

# Fields
- `location::Vector{Float64}`: The coordinates of the fixed point
- `eigenvalues::Vector{ComplexF64}`: Eigenvalues of the Jacobian at this point
- `type::FixedPointType`: Classification of the fixed point
- `is_stable::Bool`: Whether the fixed point is stable (all Re(λ) < 0)
- `is_hyperbolic::Bool`: Whether all eigenvalues have non-zero real parts
"""
struct FixedPoint
    location::Vector{Float64}
    eigenvalues::Vector{ComplexF64}
    type::FixedPointType
    is_stable::Bool
    is_hyperbolic::Bool
end

function Base.show(io::IO, fp::FixedPoint)
    print(io, "FixedPoint($(fp.type), stable=$(fp.is_stable), hyperbolic=$(fp.is_hyperbolic))")
end

function Base.show(io::IO, ::MIME"text/plain", fp::FixedPoint)
    println(io, "FixedPoint")
    println(io, "  Location: $(fp.location)")
    println(io, "  Type: $(fp.type)")
    println(io, "  Stable: $(fp.is_stable)")
    println(io, "  Hyperbolic: $(fp.is_hyperbolic)")
    print(io, "  Eigenvalues: $(fp.eigenvalues)")
end


#=============================================================================
    Eigenvalue Analysis and Classification
=============================================================================#

"""
    compute_eigenvalues(ds::DynamicalSystem, x::AbstractVector)

Compute the eigenvalues of the Jacobian at point x.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `x::AbstractVector`: The point at which to evaluate

# Returns
- `Vector{ComplexF64}`: The eigenvalues of the Jacobian
"""
function compute_eigenvalues(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    eigs = eigvals(J)
    # Convert to ComplexF64 to ensure consistent type
    return ComplexF64.(eigs)
end


"""
    is_hyperbolic(eigenvalues::AbstractVector{<:Complex}; tol::Real=1e-10)

Check if all eigenvalues have non-zero real parts (hyperbolic condition).

# Arguments
- `eigenvalues`: Vector of eigenvalues
- `tol::Real=1e-10`: Tolerance for considering real part as zero

# Returns
- `Bool`: `true` if all eigenvalues have |Re(λ)| > tol
"""
function is_hyperbolic(eigenvalues::AbstractVector{<:Complex}; tol::Real=1e-10)
    return all(abs(real(λ)) > tol for λ in eigenvalues)
end


"""
    is_hyperbolic(ds::DynamicalSystem, x::AbstractVector; tol::Real=1e-10)

Check if the fixed point at x is hyperbolic.
"""
function is_hyperbolic(ds::DynamicalSystem, x::AbstractVector; tol::Real=1e-10)
    eigenvalues = compute_eigenvalues(ds, x)
    return is_hyperbolic(eigenvalues; tol=tol)
end


"""
    is_stable(eigenvalues::AbstractVector{<:Complex}; tol::Real=1e-10)

Check if all eigenvalues have negative real parts (asymptotic stability).

# Arguments
- `eigenvalues`: Vector of eigenvalues
- `tol::Real=1e-10`: Tolerance for stability boundary

# Returns
- `Bool`: `true` if all Re(λ) < -tol
"""
function is_stable(eigenvalues::AbstractVector{<:Complex}; tol::Real=1e-10)
    return all(real(λ) < -tol for λ in eigenvalues)
end


"""
    is_stable(ds::DynamicalSystem, x::AbstractVector; tol::Real=1e-10)

Check if the fixed point at x is stable.
"""
function is_stable(ds::DynamicalSystem, x::AbstractVector; tol::Real=1e-10)
    eigenvalues = compute_eigenvalues(ds, x)
    return is_stable(eigenvalues; tol=tol)
end


"""
    classify_fixed_point_type(eigenvalues::AbstractVector{<:Complex}; tol::Real=1e-10)

Classify the type of fixed point based on its eigenvalues.

# Arguments
- `eigenvalues`: Vector of eigenvalues of the Jacobian at the fixed point
- `tol::Real=1e-10`: Tolerance for numerical comparisons

# Returns
- `FixedPointType`: The classification of the fixed point
"""
function classify_fixed_point_type(eigenvalues::AbstractVector{<:Complex}; tol::Real=1e-10)
    n = length(eigenvalues)
    
    real_parts = real.(eigenvalues)
    imag_parts = imag.(eigenvalues)
    
    # Check for non-hyperbolic (zero real parts)
    has_zero_real = any(abs(r) ≤ tol for r in real_parts)
    
    if has_zero_real
        # Check if pure imaginary (center)
        if all(abs(r) ≤ tol for r in real_parts) && any(abs(i) > tol for i in imag_parts)
            return CENTER
        end
        return NON_HYPERBOLIC
    end
    
    # Count positive and negative real parts
    n_positive = count(r -> r > tol, real_parts)
    n_negative = count(r -> r < -tol, real_parts)
    
    # Check if any complex eigenvalues (non-zero imaginary parts)
    has_complex = any(abs(i) > tol for i in imag_parts)
    
    if !has_complex
        # All real eigenvalues
        if n_negative == n
            return STABLE_NODE
        elseif n_positive == n
            return UNSTABLE_NODE
        else
            return SADDLE
        end
    else
        # Has complex eigenvalues
        if n_negative == n
            return STABLE_FOCUS
        elseif n_positive == n
            return UNSTABLE_FOCUS
        else
            return SADDLE_FOCUS
        end
    end
end


"""
    classify_fixed_point(ds::DynamicalSystem, x::AbstractVector; 
                         tol::Real=1e-10, verify::Bool=true, 
                         verification_tol::Real=1e-8)

Classify a fixed point of a dynamical system.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `x::AbstractVector`: The location of the (candidate) fixed point
- `tol::Real=1e-10`: Tolerance for eigenvalue analysis
- `verify::Bool=true`: Whether to verify that x is actually a fixed point
- `verification_tol::Real=1e-8`: Tolerance for fixed point verification

# Returns
- `FixedPoint`: A struct containing the location, eigenvalues, and classification

# Throws
- `ArgumentError`: If verify=true and x is not a fixed point within tolerance
"""
function classify_fixed_point(ds::DynamicalSystem, x::AbstractVector; 
                               tol::Real=1e-10, verify::Bool=true,
                               verification_tol::Real=1e-8)
    # Verify it's actually a fixed point
    if verify
        f_x = ds(x)
        if norm(f_x) > verification_tol
            throw(ArgumentError(
                "Point is not a fixed point: ‖F(x)‖ = $(norm(f_x)) > $verification_tol"
            ))
        end
    end
    
    eigenvalues = compute_eigenvalues(ds, x)
    fp_type = classify_fixed_point_type(eigenvalues; tol=tol)
    fp_stable = is_stable(eigenvalues; tol=tol)
    fp_hyperbolic = is_hyperbolic(eigenvalues; tol=tol)
    
    return FixedPoint(collect(Float64, x), eigenvalues, fp_type, fp_stable, fp_hyperbolic)
end


#=============================================================================
    Sampling Methods
=============================================================================#

"""
    latin_hypercube_sample(bounds::NTuple{N, Tuple{Real, Real}}, n::Int) where N

Generate n points in N dimensions using Latin Hypercube Sampling.

This provides better coverage of the space than random sampling.

# Arguments
- `bounds`: Tuple of (min, max) pairs for each dimension
- `n::Int`: Number of points to generate

# Returns
- `Vector{Vector{Float64}}`: n points, each of dimension N

# Examples
```julia
bounds = ((-2.0, 2.0), (-2.0, 2.0))  # 2D box
points = latin_hypercube_sample(bounds, 100)
```
"""
function latin_hypercube_sample(bounds::NTuple{N, Tuple{Real, Real}}, n::Int) where N
    n > 0 || throw(ArgumentError("Number of samples must be positive"))
    
    points = Vector{Vector{Float64}}(undef, n)
    
    for i in 1:n
        points[i] = Vector{Float64}(undef, N)
    end
    
    for d in 1:N
        lo, hi = bounds[d]
        lo < hi || throw(ArgumentError("Lower bound must be less than upper bound in dimension $d"))
        
        # Create n intervals and randomly sample one point from each
        perm = randperm(n)
        for i in 1:n
            # Interval for this sample
            interval_lo = lo + (perm[i] - 1) * (hi - lo) / n
            interval_hi = lo + perm[i] * (hi - lo) / n
            points[i][d] = interval_lo + rand() * (interval_hi - interval_lo)
        end
    end
    
    return points
end


"""
    latin_hypercube_sample(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Real, Real}}, n::Int) where N

Generate Latin Hypercube samples compatible with a dynamical system.

Validates that the bounds dimension matches the system dimension.
"""
function latin_hypercube_sample(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Real, Real}}, n::Int) where N
    N == ds.dim || throw(DimensionMismatch(
        "Bounds dimension ($N) doesn't match system dimension ($(ds.dim))"
    ))
    return latin_hypercube_sample(bounds, n)
end


#=============================================================================
    Clustering
=============================================================================#

"""
    cluster_points(points::Vector{<:AbstractVector}; distance_tol::Real=1e-6)

Cluster nearby points using a simple distance-based algorithm.

Points within `distance_tol` of each other are grouped together.
Returns the centroid of each cluster.

# Arguments
- `points`: Vector of points to cluster
- `distance_tol::Real=1e-6`: Maximum distance between points in same cluster

# Returns
- `Vector{Vector{Float64}}`: Cluster centroids
"""
function cluster_points(points::Vector{<:AbstractVector}; distance_tol::Real=1e-6)
    isempty(points) && return Vector{Float64}[]
    
    n = length(points)
    dim = length(points[1])
    
    # Track which cluster each point belongs to
    cluster_id = zeros(Int, n)
    n_clusters = 0
    
    for i in 1:n
        if cluster_id[i] == 0
            # Start a new cluster
            n_clusters += 1
            cluster_id[i] = n_clusters
            
            # Find all points close to this one
            for j in (i+1):n
                if cluster_id[j] == 0
                    if norm(points[i] - points[j]) < distance_tol
                        cluster_id[j] = n_clusters
                    end
                end
            end
        end
    end
    
    # Compute centroids
    centroids = [zeros(dim) for _ in 1:n_clusters]
    counts = zeros(Int, n_clusters)
    
    for i in 1:n
        c = cluster_id[i]
        centroids[c] .+= points[i]
        counts[c] += 1
    end
    
    for c in 1:n_clusters
        centroids[c] ./= counts[c]
    end
    
    return centroids
end


"""
    unique_fixed_points(candidates::Vector{<:AbstractVector}; tol::Real=1e-6)

Remove duplicate fixed point candidates by clustering.

This is useful after finding fixed points from multiple initial conditions.

# Arguments
- `candidates`: Vector of candidate fixed point locations
- `tol::Real=1e-6`: Tolerance for considering points as identical

# Returns
- `Vector{Vector{Float64}}`: Unique fixed point locations
"""
function unique_fixed_points(candidates::Vector{<:AbstractVector}; tol::Real=1e-6)
    return cluster_points(candidates; distance_tol=tol)
end


#=============================================================================
    Root Finding Approach (NLsolve)
=============================================================================#

"""
    find_fixed_points_nlsolve(ds::DynamicalSystem, initial_guesses::Vector{<:AbstractVector};
                               ftol::Real=1e-10, iterations::Int=1000,
                               method::Symbol=:trust_region,
                               unique_tol::Real=1e-6)

Find fixed points using nonlinear root finding (NLsolve.jl).

This method can find both stable and unstable fixed points, but requires
good initial guesses and may fail to converge for some systems.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `initial_guesses`: Vector of starting points for the solver
- `ftol::Real=1e-10`: Function tolerance for convergence
- `iterations::Int=1000`: Maximum iterations per solve
- `method::Symbol=:trust_region`: NLsolve method (:trust_region or :newton)
- `unique_tol::Real=1e-6`: Tolerance for deduplicating found fixed points

# Returns
- `Vector{Vector{Float64}}`: Locations of found fixed points (deduplicated)

# Examples
```julia
ds = DynamicalSystem(x -> [x[1]*(1-x[1]) - x[1]*x[2], x[2]*(x[1]-1)], 2)
guesses = [[0.1, 0.1], [0.9, 0.1], [0.5, 0.5]]
fps = find_fixed_points_nlsolve(ds, guesses)
```
"""
function find_fixed_points_nlsolve(ds::DynamicalSystem, initial_guesses::Vector{<:AbstractVector};
                                    ftol::Real=1e-10, iterations::Int=1000,
                                    method::Symbol=:trust_region,
                                    unique_tol::Real=1e-6)
    candidates = Vector{Float64}[]
    
    for x0 in initial_guesses
        length(x0) == ds.dim || throw(DimensionMismatch(
            "Initial guess dimension ($(length(x0))) doesn't match system dimension ($(ds.dim))"
        ))
        
        try
            # Define the function for NLsolve
            function f!(F, x)
                result = ds.f(x)
                for i in 1:length(F)
                    F[i] = result[i]
                end
            end
            
            # Solve
            result = nlsolve(f!, collect(Float64, x0); 
                            ftol=ftol, iterations=iterations, method=method)
            
            if converged(result)
                push!(candidates, result.zero)
            end
        catch e
            # Solver failed for this initial guess, continue to next
            @debug "NLsolve failed for initial guess $x0: $e"
        end
    end
    
    # Deduplicate
    if isempty(candidates)
        return Vector{Float64}[]
    end
    
    return unique_fixed_points(candidates; tol=unique_tol)
end


#=============================================================================
    ODE Integration Approach
=============================================================================#

"""
    StationarityCallback

Callback for ODE solver to terminate when solution becomes stationary.
"""
struct StationarityCallback
    tol::Float64
    check_interval::Float64
end

"""
    find_fixed_points_ode(ds::DynamicalSystem, initial_conditions::Vector{<:AbstractVector};
                          tspan::Tuple{Real,Real}=(0.0, 1000.0),
                          stationarity_tol::Real=1e-8,
                          solver=Tsit5(),
                          unique_tol::Real=1e-6,
                          dt_check::Real=1.0,
                          abstol::Real=1e-10,
                          reltol::Real=1e-8)

Find stable fixed points by integrating the ODE until stationarity.

This method is more robust than root finding for stable fixed points,
but cannot find unstable fixed points. Use `find_unstable_fixed_points_ode`
for those.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `initial_conditions`: Vector of starting points
- `tspan::Tuple{Real,Real}=(0.0, 1000.0)`: Time span for integration
- `stationarity_tol::Real=1e-8`: Tolerance for considering solution stationary
- `solver=Tsit5()`: ODE solver to use
- `unique_tol::Real=1e-6`: Tolerance for deduplicating found fixed points
- `dt_check::Real=1.0`: Interval for checking stationarity
- `abstol::Real=1e-10`: Absolute tolerance for ODE solver
- `reltol::Real=1e-8`: Relative tolerance for ODE solver

# Returns
- `Vector{Vector{Float64}}`: Locations of found stable fixed points (deduplicated)
"""
function find_fixed_points_ode(ds::DynamicalSystem, initial_conditions::Vector{<:AbstractVector};
                                tspan::Tuple{Real,Real}=(0.0, 1000.0),
                                stationarity_tol::Real=1e-8,
                                solver=Tsit5(),
                                unique_tol::Real=1e-6,
                                dt_check::Real=1.0,
                                abstol::Real=1e-10,
                                reltol::Real=1e-8)
    
    candidates = Vector{Float64}[]
    
    # Define ODE function
    function ode_f!(du, u, p, t)
        result = ds.f(u)
        for i in 1:length(du)
            du[i] = result[i]
        end
    end
    
    # Termination condition: solution is stationary
    function condition(u, t, integrator)
        # Check if derivative is small
        du = similar(u)
        ode_f!(du, u, nothing, t)
        return norm(du) < stationarity_tol
    end
    
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))
    
    for x0 in initial_conditions
        length(x0) == ds.dim || throw(DimensionMismatch(
            "Initial condition dimension ($(length(x0))) doesn't match system dimension ($(ds.dim))"
        ))
        
        try
            prob = ODEProblem(ode_f!, collect(Float64, x0), tspan)
            sol = solve(prob, solver; callback=cb, abstol=abstol, reltol=reltol,
                       save_everystep=false, save_start=false)
            
            # Check if final state is actually stationary
            final_state = sol.u[end]
            du = similar(final_state)
            ode_f!(du, final_state, nothing, sol.t[end])
            
            if norm(du) < stationarity_tol
                push!(candidates, final_state)
            end
        catch e
            @debug "ODE integration failed for initial condition $x0: $e"
        end
    end
    
    if isempty(candidates)
        return Vector{Float64}[]
    end
    
    return unique_fixed_points(candidates; tol=unique_tol)
end


"""
    find_unstable_fixed_points_ode(ds::DynamicalSystem, initial_conditions::Vector{<:AbstractVector};
                                    tspan::Tuple{Real,Real}=(0.0, 1000.0),
                                    stationarity_tol::Real=1e-8,
                                    solver=Tsit5(),
                                    unique_tol::Real=1e-6,
                                    abstol::Real=1e-10,
                                    reltol::Real=1e-8)

Find unstable fixed points by integrating the reversed ODE (backward in time).

For the reversed system dx/dt = -F(x), unstable fixed points of the original
system become stable fixed points, allowing them to be found via forward
integration of the reversed system.

# Arguments
Same as `find_fixed_points_ode`.

# Returns
- `Vector{Vector{Float64}}`: Locations of found unstable fixed points (deduplicated)

# Note
This finds fixed points that are unstable in at least one direction. Saddle points
may or may not be found depending on the initial conditions.
"""
function find_unstable_fixed_points_ode(ds::DynamicalSystem, initial_conditions::Vector{<:AbstractVector};
                                         tspan::Tuple{Real,Real}=(0.0, 1000.0),
                                         stationarity_tol::Real=1e-8,
                                         solver=Tsit5(),
                                         unique_tol::Real=1e-6,
                                         abstol::Real=1e-10,
                                         reltol::Real=1e-8)
    
    # Create reversed system: dx/dt = -F(x)
    reversed_f = x -> -ds.f(x)
    ds_reversed = DynamicalSystem(reversed_f, ds.dim)
    
    return find_fixed_points_ode(ds_reversed, initial_conditions;
                                  tspan=tspan,
                                  stationarity_tol=stationarity_tol,
                                  solver=solver,
                                  unique_tol=unique_tol,
                                  abstol=abstol,
                                  reltol=reltol)
end


#=============================================================================
    High-Level Interface
=============================================================================#

"""
    find_fixed_points(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Real, Real}};
                      n_samples::Int=100,
                      method::Symbol=:auto,
                      find_unstable::Bool=true,
                      unique_tol::Real=1e-6,
                      nlsolve_ftol::Real=1e-10,
                      ode_tspan::Tuple{Real,Real}=(0.0, 1000.0),
                      ode_stationarity_tol::Real=1e-8,
                      verification_tol::Real=1e-8) where N

Find fixed points of a dynamical system within the specified bounds.

This is the main high-level interface that combines multiple methods:
1. ODE integration (forward) for stable fixed points
2. ODE integration (backward) for unstable fixed points (if find_unstable=true)
3. NLsolve refinement for higher precision

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `bounds`: Tuple of (min, max) pairs for each dimension
- `n_samples::Int=100`: Number of initial conditions to try
- `method::Symbol=:auto`: Method to use (:auto, :ode, :nlsolve, :both)
- `find_unstable::Bool=true`: Whether to also search for unstable fixed points
- `unique_tol::Real=1e-6`: Tolerance for deduplicating fixed points
- `nlsolve_ftol::Real=1e-10`: Function tolerance for NLsolve
- `ode_tspan::Tuple{Real,Real}=(0.0, 1000.0)`: Time span for ODE integration
- `ode_stationarity_tol::Real=1e-8`: Tolerance for ODE stationarity
- `verification_tol::Real=1e-8`: Tolerance for verifying fixed points

# Returns
- `Vector{FixedPoint}`: Found fixed points with their classifications

# Examples
```julia
# Simple 2D system
ds = DynamicalSystem(x -> [x[1] - x[1]^3, -x[2]], 2)
bounds = ((-2.0, 2.0), (-2.0, 2.0))
fps = find_fixed_points(ds, bounds)

# 3D Lorenz system
function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
    return [σ*(x[2]-x[1]), x[1]*(ρ-x[3])-x[2], x[1]*x[2]-β*x[3]]
end
ds = DynamicalSystem(lorenz, 3)
bounds = ((-30.0, 30.0), (-30.0, 30.0), (0.0, 50.0))
fps = find_fixed_points(ds, bounds; n_samples=200)
```
"""
function find_fixed_points(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Real, Real}};
                           n_samples::Int=100,
                           method::Symbol=:auto,
                           find_unstable::Bool=true,
                           unique_tol::Real=1e-6,
                           nlsolve_ftol::Real=1e-10,
                           ode_tspan::Tuple{Real,Real}=(0.0, 1000.0),
                           ode_stationarity_tol::Real=1e-8,
                           verification_tol::Real=1e-8) where N
    
    N == ds.dim || throw(DimensionMismatch(
        "Bounds dimension ($N) doesn't match system dimension ($(ds.dim))"
    ))
    
    # Generate initial conditions
    initial_conditions = latin_hypercube_sample(bounds, n_samples)
    
    all_candidates = Vector{Float64}[]
    
    if method in (:auto, :ode, :both)
        # Find stable fixed points via ODE
        stable_fps = find_fixed_points_ode(ds, initial_conditions;
                                           tspan=ode_tspan,
                                           stationarity_tol=ode_stationarity_tol,
                                           unique_tol=unique_tol)
        append!(all_candidates, stable_fps)
        
        # Find unstable fixed points via backward ODE
        if find_unstable
            unstable_fps = find_unstable_fixed_points_ode(ds, initial_conditions;
                                                          tspan=ode_tspan,
                                                          stationarity_tol=ode_stationarity_tol,
                                                          unique_tol=unique_tol)
            append!(all_candidates, unstable_fps)
        end
    end
    
    if method in (:auto, :nlsolve, :both)
        # Also try NLsolve from initial conditions
        nlsolve_fps = find_fixed_points_nlsolve(ds, initial_conditions;
                                                 ftol=nlsolve_ftol,
                                                 unique_tol=unique_tol)
        append!(all_candidates, nlsolve_fps)
    end
    
    # Deduplicate all candidates
    unique_candidates = unique_fixed_points(all_candidates; tol=unique_tol)
    
    # Verify and classify each candidate
    fixed_points = FixedPoint[]
    
    for candidate in unique_candidates
        # Verify it's a fixed point
        f_x = ds(candidate)
        if norm(f_x) < verification_tol
            # Refine with NLsolve for higher precision
            try
                refined = find_fixed_points_nlsolve(ds, [candidate]; 
                                                    ftol=nlsolve_ftol,
                                                    unique_tol=unique_tol/10)
                if !isempty(refined)
                    candidate = refined[1]
                end
            catch
                # Keep original if refinement fails
            end
            
            # Classify
            try
                fp = classify_fixed_point(ds, candidate; 
                                          verify=false,  # Already verified
                                          tol=1e-10)
                push!(fixed_points, fp)
            catch e
                @debug "Failed to classify fixed point at $candidate: $e"
            end
        end
    end
    
    return fixed_points
end


"""
    all_fixed_points_hyperbolic(fps::Vector{FixedPoint})

Check if all fixed points in a collection are hyperbolic.

This is a necessary condition for Morse-Smale and structurally stable systems.

# Arguments
- `fps::Vector{FixedPoint}`: Collection of fixed points

# Returns
- `Bool`: `true` if all fixed points are hyperbolic
"""
function all_fixed_points_hyperbolic(fps::Vector{FixedPoint})
    return all(fp -> fp.is_hyperbolic, fps)
end


"""
    count_fixed_point_types(fps::Vector{FixedPoint})

Count the number of each type of fixed point.

# Returns
- `Dict{FixedPointType, Int}`: Counts for each type
"""
function count_fixed_point_types(fps::Vector{FixedPoint})
    counts = Dict{FixedPointType, Int}()
    for fp in fps
        counts[fp.type] = get(counts, fp.type, 0) + 1
    end
    return counts
end


"""
    stable_fixed_points(fps::Vector{FixedPoint})

Filter to only stable fixed points.

# Returns
- `Vector{FixedPoint}`: Only the stable fixed points
"""
function stable_fixed_points(fps::Vector{FixedPoint})
    return filter(fp -> fp.is_stable, fps)
end


"""
    unstable_fixed_points(fps::Vector{FixedPoint})

Filter to only unstable fixed points.

# Returns
- `Vector{FixedPoint}`: Only the unstable fixed points
"""
function unstable_fixed_points(fps::Vector{FixedPoint})
    return filter(fp -> !fp.is_stable, fps)
end
