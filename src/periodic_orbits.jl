# periodic_orbits.jl
# Detection, analysis, and classification of periodic orbits (limit cycles)

"""
Classification types for periodic orbits based on Floquet multipliers.
"""
@enum PeriodicOrbitType begin
    STABLE_LIMIT_CYCLE      # All Floquet multipliers inside unit circle (except trivial one)
    UNSTABLE_LIMIT_CYCLE    # At least one Floquet multiplier outside unit circle, none on
    SADDLE_CYCLE            # Floquet multipliers both inside and outside unit circle
    NON_HYPERBOLIC_CYCLE    # At least one non-trivial Floquet multiplier on unit circle
    UNKNOWN_CYCLE           # Could not determine classification
end

"""
    PeriodicOrbit

Represents a periodic orbit (limit cycle) of a dynamical system.

# Fields
- `points::Vector{Vector{Float64}}`: Sampled points along the orbit
- `period::Float64`: The period T of the orbit
- `floquet_multipliers::Vector{ComplexF64}`: Eigenvalues of the monodromy matrix
- `type::PeriodicOrbitType`: Classification of the orbit
- `is_stable::Bool`: Whether the orbit is asymptotically stable
- `is_hyperbolic::Bool`: Whether all non-trivial Floquet multipliers are away from unit circle
"""
struct PeriodicOrbit
    points::Vector{Vector{Float64}}
    period::Float64
    floquet_multipliers::Vector{ComplexF64}
    type::PeriodicOrbitType
    is_stable::Bool
    is_hyperbolic::Bool
end

# Constructor with automatic classification
function PeriodicOrbit(points::Vector{Vector{Float64}}, period::Float64, 
                       floquet_multipliers::Vector{<:Number})
    fm = ComplexF64.(floquet_multipliers)
    orbit_type, stable, hyperbolic = classify_floquet_multipliers(fm)
    PeriodicOrbit(points, period, fm, orbit_type, stable, hyperbolic)
end

# Simple constructor without Floquet analysis
function PeriodicOrbit(points::Vector{Vector{Float64}}, period::Float64)
    PeriodicOrbit(points, period, ComplexF64[], UNKNOWN_CYCLE, false, false)
end

#=============================================================================
                        Floquet Analysis
=============================================================================#

"""
    classify_floquet_multipliers(multipliers; tol=1e-6)

Classify a periodic orbit based on its Floquet multipliers.

For a periodic orbit of an n-dimensional system, the monodromy matrix has n eigenvalues
(Floquet multipliers). One multiplier is always +1 (trivial, along the orbit direction).
The remaining n-1 determine stability:
- All |μ| < 1: Stable limit cycle
- All |μ| > 1 (except trivial): Unstable limit cycle  
- Mixed: Saddle cycle
- Any |μ| = 1 (non-trivial): Non-hyperbolic

Returns: (type, is_stable, is_hyperbolic)
"""
function classify_floquet_multipliers(multipliers::Vector{ComplexF64}; tol::Float64=1e-6)
    if isempty(multipliers)
        return (UNKNOWN_CYCLE, false, false)
    end
    
    # Find non-trivial multipliers (exclude the one closest to +1)
    mags = abs.(multipliers)
    phases = angle.(multipliers)
    
    # Identify trivial multiplier: closest to +1 (magnitude 1, phase 0)
    distances_to_one = abs.(multipliers .- 1.0)
    trivial_idx = argmin(distances_to_one)
    
    # Get non-trivial multipliers
    non_trivial_indices = setdiff(1:length(multipliers), [trivial_idx])
    
    if isempty(non_trivial_indices)
        # Only the trivial multiplier exists (1D system - shouldn't happen for limit cycles)
        return (UNKNOWN_CYCLE, false, false)
    end
    
    non_trivial_mags = mags[non_trivial_indices]
    
    # Check for non-hyperbolicity: any multiplier on unit circle
    on_unit_circle = any(abs(m - 1.0) < tol for m in non_trivial_mags)
    
    if on_unit_circle
        return (NON_HYPERBOLIC_CYCLE, false, false)
    end
    
    # Classify based on magnitudes
    inside = all(m < 1.0 - tol for m in non_trivial_mags)
    outside = all(m > 1.0 + tol for m in non_trivial_mags)
    
    if inside
        return (STABLE_LIMIT_CYCLE, true, true)
    elseif outside
        return (UNSTABLE_LIMIT_CYCLE, false, true)
    else
        return (SADDLE_CYCLE, false, true)
    end
end

"""
    compute_monodromy_matrix(ds::DynamicalSystem, x0::Vector{Float64}, period::Float64;
                             solver=Tsit5(), reltol=1e-8, abstol=1e-10)

Compute the monodromy matrix for a periodic orbit.

The monodromy matrix M is defined such that a perturbation δx(0) evolves to δx(T) = M·δx(0)
after one period T. It's computed by integrating the variational equation:
    dΦ/dt = J(x(t))·Φ, Φ(0) = I

where J is the Jacobian and Φ(T) = M is the monodromy matrix.
"""
function compute_monodromy_matrix(ds::DynamicalSystem, x0::Vector{Float64}, period::Float64;
                                   solver=Tsit5(), reltol::Float64=1e-8, abstol::Float64=1e-10)
    n = dimension(ds)
    
    # State: [x; vec(Φ)] where Φ is n×n identity initially
    # Total dimension: n + n²
    
    function variational_ode!(du, u, p, t)
        x = u[1:n]
        Φ = reshape(u[n+1:end], n, n)
        
        # Dynamics
        dx = ds.f(x)
        
        # Jacobian at current point
        J = compute_jacobian(ds, x)
        
        # Variational equation: dΦ/dt = J·Φ
        dΦ = J * Φ
        
        du[1:n] .= dx
        du[n+1:end] .= vec(dΦ)
    end
    
    # Initial condition: x0 and identity matrix
    u0 = vcat(x0, vec(Matrix{Float64}(I, n, n)))
    
    # Integrate for one period
    prob = ODEProblem(variational_ode!, u0, (0.0, period))
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, save_everystep=false)
    
    # Extract monodromy matrix from final state
    u_final = sol.u[end]
    M = reshape(u_final[n+1:end], n, n)
    
    return M
end

"""
    compute_floquet_multipliers(ds::DynamicalSystem, x0::Vector{Float64}, period::Float64; kwargs...)

Compute the Floquet multipliers for a periodic orbit starting at x0 with given period.

Returns the eigenvalues of the monodromy matrix, sorted by magnitude (descending).
"""
function compute_floquet_multipliers(ds::DynamicalSystem, x0::Vector{Float64}, period::Float64; kwargs...)
    M = compute_monodromy_matrix(ds, x0, period; kwargs...)
    multipliers = eigvals(M)
    # Sort by magnitude (descending) so trivial multiplier ~1 is first
    sorted_indices = sortperm(abs.(multipliers), rev=true)
    return ComplexF64.(multipliers[sorted_indices])
end

"""
    compute_floquet_multipliers(ds::DynamicalSystem, orbit::PeriodicOrbit; kwargs...)

Compute Floquet multipliers from an existing PeriodicOrbit object.
"""
function compute_floquet_multipliers(ds::DynamicalSystem, orbit::PeriodicOrbit; kwargs...)
    if isempty(orbit.points)
        error("PeriodicOrbit has no points")
    end
    return compute_floquet_multipliers(ds, orbit.points[1], orbit.period; kwargs...)
end

#=============================================================================
                        Oscillation Detection
=============================================================================#

"""
    detect_oscillation(ds::DynamicalSystem, x0::Vector{Float64};
                       tspan=(0.0, 100.0), transient_fraction=0.5,
                       min_oscillations=3, relative_period_tol=0.3,
                       solver=Tsit5(), kwargs...)

Detect if a trajectory starting from x0 exhibits sustained oscillatory behaviour.

Returns `(is_oscillating, estimated_period, amplitude)` where:
- `is_oscillating`: true if sustained oscillations detected
- `estimated_period`: rough estimate of oscillation period (or NaN)
- `amplitude`: typical oscillation amplitude (or NaN)

The method integrates the trajectory, discards a transient, then analyses the
remaining trajectory for zero-crossings (relative to mean) to detect periodicity.
"""
function detect_oscillation(ds::DynamicalSystem, x0::Vector{Float64};
                            tspan::Tuple{Float64,Float64}=(0.0, 100.0),
                            transient_fraction::Float64=0.5,
                            min_oscillations::Int=3,
                            relative_period_tol::Float64=0.3,
                            solver=Tsit5(),
                            reltol::Float64=1e-6, abstol::Float64=1e-8)
    
    # Integrate trajectory with finer time resolution
    dt = 0.005 * (tspan[2] - tspan[1])
    prob = ODEProblem((u, p, t) -> ds.f(u), x0, tspan)
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, saveat=dt)
    
    if sol.retcode != ReturnCode.Success
        return (false, NaN, NaN)
    end
    
    # Discard transient
    n_points = length(sol.t)
    start_idx = max(1, round(Int, transient_fraction * n_points))
    
    t_post = sol.t[start_idx:end]
    
    if length(t_post) < 20
        return (false, NaN, NaN)
    end
    
    # Try each coordinate and use the one with clearest oscillation
    n_dim = dimension(ds)
    best_result = (false, NaN, NaN)
    
    for coord in 1:n_dim
        x_post = [sol.u[i][coord] for i in start_idx:n_points]
        
        # Check for variability (not converged to fixed point)
        x_mean = mean(x_post)
        x_std = std(x_post)
        x_range = maximum(x_post) - minimum(x_post)
        
        if x_std < 1e-6 * (abs(x_mean) + 1e-10)
            # Converged to fixed point, no oscillation in this coordinate
            continue
        end
        
        # Count zero-crossings relative to mean
        centered = x_post .- x_mean
        crossings = Int[]
        
        for i in 2:length(centered)
            if centered[i-1] * centered[i] < 0
                push!(crossings, i)
            end
        end
        
        n_crossings = length(crossings)
        
        # Need at least 2*min_oscillations crossings for min_oscillations complete cycles
        if n_crossings < 2 * min_oscillations
            continue
        end
        
        # Estimate period from crossing intervals (each period has 2 crossings)
        crossing_times = t_post[crossings]
        half_periods = diff(crossing_times)
        
        if isempty(half_periods)
            continue
        end
        
        estimated_period = 2 * median(half_periods)  # Use median for robustness
        
        # Check regularity: coefficient of variation of half-periods
        period_cv = std(half_periods) / mean(half_periods)
        if period_cv > relative_period_tol
            # Irregular oscillations (possibly quasi-periodic or chaotic)
            continue
        end
        
        amplitude = x_range / 2  # Half the range as amplitude estimate
        
        # This coordinate shows oscillation
        return (true, estimated_period, amplitude)
    end
    
    return best_result
end

#=============================================================================
                    Periodic Orbit Finding via Recurrence
=============================================================================#

"""
    find_periodic_orbit_recurrence(ds::DynamicalSystem, x0::Vector{Float64};
                                   tspan=(0.0, 200.0), transient_time=50.0,
                                   recurrence_tol=1e-4, min_period=0.1,
                                   solver=Tsit5(), kwargs...)

Find a periodic orbit by detecting trajectory recurrence.

Starting from x0, integrates the trajectory and looks for points where the
trajectory returns close to a previously visited point after at least `min_period` time.

Returns `(found, orbit_points, period)` where:
- `found`: whether a periodic orbit was detected
- `orbit_points`: sampled points along one period of the orbit
- `period`: the detected period
"""
function find_periodic_orbit_recurrence(ds::DynamicalSystem, x0::Vector{Float64};
                                        tspan::Tuple{Float64,Float64}=(0.0, 200.0),
                                        transient_time::Float64=50.0,
                                        recurrence_tol::Float64=1e-4,
                                        min_period::Float64=0.1,
                                        solver=Tsit5(),
                                        reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    # Integrate trajectory with fine time resolution
    saveat = min(0.01, min_period / 10)
    prob = ODEProblem((u, p, t) -> ds.f(u), x0, tspan)
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, saveat=saveat)
    
    if sol.retcode != ReturnCode.Success
        return (false, Vector{Float64}[], NaN)
    end
    
    # Find start index after transient
    transient_idx = findfirst(t -> t >= transient_time, sol.t)
    if isnothing(transient_idx) || transient_idx >= length(sol.t) - 10
        return (false, Vector{Float64}[], NaN)
    end
    
    times = sol.t[transient_idx:end]
    points = sol.u[transient_idx:end]
    n = length(times)
    
    # Strategy: for each reference point i, find the FIRST j where trajectory returns
    # This finds the shortest period
    best_period = Inf
    best_i = 0
    best_j = 0
    
    # Start from a point after initial transient has settled
    # Use points spaced apart to avoid trivial matches
    step = max(1, n ÷ 100)  # Check ~100 reference points
    
    for i in 1:step:min(n÷2, n-10)
        ref_point = points[i]
        ref_time = times[i]
        
        # Search forward for first return
        for j in (i+1):n
            dt = times[j] - ref_time
            
            if dt < min_period
                continue
            end
            
            dist = norm(points[j] - ref_point)
            
            if dist < recurrence_tol
                # Found a recurrence - check if this is the shortest period so far
                if dt < best_period
                    best_period = dt
                    best_i = i
                    best_j = j
                end
                break  # Move to next reference point (we found first return for this one)
            end
            
            # Don't search too far for a single reference point
            if dt > 3 * best_period && best_period < Inf
                break
            end
        end
    end
    
    if best_period == Inf
        return (false, Vector{Float64}[], NaN)
    end
    
    # Extract one period worth of points
    period_indices = best_i:best_j
    orbit_points = [copy(points[k]) for k in period_indices]
    
    return (true, orbit_points, best_period)
end

"""
    refine_periodic_orbit(ds::DynamicalSystem, x0_guess::Vector{Float64}, T_guess::Float64;
                          max_iterations=50, tol=1e-10, solver=Tsit5(), kwargs...)

Refine a periodic orbit estimate using a shooting method.

Given an initial guess (x0_guess, T_guess) for a point on a periodic orbit and its period,
uses Newton-like iteration to find a true periodic orbit satisfying φ(x₀, T) = x₀.

Note: This solves for x₀ only, keeping T fixed (the period from recurrence detection).
For more robust refinement, one could also optimise T, but that requires care with
the trivial solution and is more complex.

Returns `(success, refined_x0, refined_points)`.
"""
function refine_periodic_orbit(ds::DynamicalSystem, x0_guess::Vector{Float64}, T_guess::Float64;
                               max_iterations::Int=50, tol::Float64=1e-10,
                               solver=Tsit5(), reltol::Float64=1e-10, abstol::Float64=1e-12)
    
    n = dimension(ds)
    x0 = copy(x0_guess)
    T = T_guess
    
    for iter in 1:max_iterations
        # Integrate for one period and get monodromy matrix
        u0 = vcat(x0, vec(Matrix{Float64}(I, n, n)))
        
        function variational_ode!(du, u, p, t)
            x = u[1:n]
            Φ = reshape(u[n+1:end], n, n)
            dx = ds.f(x)
            J = compute_jacobian(ds, x)
            dΦ = J * Φ
            du[1:n] .= dx
            du[n+1:end] .= vec(dΦ)
        end
        
        prob = ODEProblem(variational_ode!, u0, (0.0, T))
        sol = solve(prob, solver; reltol=reltol, abstol=abstol, save_everystep=false)
        
        if sol.retcode != ReturnCode.Success
            return (false, x0_guess, Vector{Float64}[])
        end
        
        u_final = sol.u[end]
        x_T = u_final[1:n]
        M = reshape(u_final[n+1:end], n, n)
        
        # Residual: F(x₀) = φ(x₀, T) - x₀
        residual = x_T - x0
        
        if norm(residual) < tol
            # Converged! Get orbit points
            prob_orbit = ODEProblem((u, p, t) -> ds.f(u), x0, (0.0, T))
            sol_orbit = solve(prob_orbit, solver; reltol=reltol, abstol=abstol, saveat=T/100)
            orbit_points = [copy(u) for u in sol_orbit.u]
            return (true, x0, orbit_points)
        end
        
        # Newton step: δx = -(M - I)⁻¹ · residual
        # But M - I is singular (has eigenvalue 0 along orbit direction)
        # Use pseudo-inverse or regularisation
        A = M - I
        
        # SVD-based pseudo-inverse for robustness
        try
            δx = pinv(A) * residual
            x0 = x0 - δx
        catch
            return (false, x0_guess, Vector{Float64}[])
        end
    end
    
    return (false, x0_guess, Vector{Float64}[])
end

#=============================================================================
                    Poincaré Section Method (2D systems)
=============================================================================#

"""
    PoincaréSection

Defines a Poincaré section for a 2D system as a line from `point` in direction `normal`.
The section is crossed when the trajectory passes through the line in the positive normal direction.
"""
struct PoincaréSection
    point::Vector{Float64}    # A point on the section
    normal::Vector{Float64}   # Normal vector (crossing direction)
end

# Default constructor: horizontal line through a point
function PoincaréSection(point::Vector{Float64})
    PoincaréSection(point, [0.0, 1.0])  # Horizontal section, detect upward crossings
end

"""
    find_poincare_crossings(ds::DynamicalSystem, x0::Vector{Float64}, section::PoincaréSection;
                            tspan=(0.0, 200.0), transient_time=50.0, max_crossings=100,
                            solver=Tsit5(), kwargs...)

Find crossings of a trajectory with a Poincaré section.

Returns a vector of (crossing_point, crossing_time) tuples for each crossing
in the positive normal direction.
"""
function find_poincare_crossings(ds::DynamicalSystem, x0::Vector{Float64}, section::PoincaréSection;
                                  tspan::Tuple{Float64,Float64}=(0.0, 200.0),
                                  transient_time::Float64=50.0,
                                  max_crossings::Int=100,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    if dimension(ds) != 2
        error("Poincaré section method currently only supports 2D systems")
    end
    
    # Integrate trajectory with fine time resolution (no dense output)
    dt = 0.001 * (tspan[2] - tspan[1])
    prob = ODEProblem((u, p, t) -> ds.f(u), x0, tspan)
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, saveat=dt)
    
    if sol.retcode != ReturnCode.Success
        return Tuple{Vector{Float64}, Float64}[]
    end
    
    crossings = Tuple{Vector{Float64}, Float64}[]
    
    # Section equation: (x - section.point) · section.normal = 0
    function section_value(x)
        return dot(x - section.point, section.normal)
    end
    
    # Scan for crossings
    for i in 2:length(sol.t)
        if sol.t[i] < transient_time
            continue
        end
        
        s_prev = section_value(sol.u[i-1])
        s_curr = section_value(sol.u[i])
        
        # Check for crossing (sign change) in positive direction
        if s_prev < 0 && s_curr >= 0
            # Linear interpolation for crossing time and point
            frac = -s_prev / (s_curr - s_prev)
            t_cross = sol.t[i-1] + frac * (sol.t[i] - sol.t[i-1])
            x_cross = sol.u[i-1] + frac * (sol.u[i] - sol.u[i-1])
            
            push!(crossings, (copy(x_cross), t_cross))
            
            if length(crossings) >= max_crossings
                break
            end
        end
    end
    
    return crossings
end

"""
    find_periodic_orbit_poincare(ds::DynamicalSystem, x0::Vector{Float64};
                                  section_point=nothing, 
                                  tspan=(0.0, 500.0), transient_time=100.0,
                                  crossing_tol=1e-4, min_crossings=5,
                                  solver=Tsit5(), kwargs...)

Find a periodic orbit in a 2D system using the Poincaré section method.

Detects if successive crossings of a Poincaré section return to the same point,
indicating a periodic orbit.

Returns `(found, orbit_points, period)`.
"""
function find_periodic_orbit_poincare(ds::DynamicalSystem, x0::Vector{Float64};
                                       section_point::Union{Nothing, Vector{Float64}}=nothing,
                                       tspan::Tuple{Float64,Float64}=(0.0, 500.0),
                                       transient_time::Float64=100.0,
                                       crossing_tol::Float64=1e-4,
                                       min_crossings::Int=5,
                                       solver=Tsit5(),
                                       reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    if dimension(ds) != 2
        error("Poincaré section method currently only supports 2D systems")
    end
    
    # Default section through x0 (or provided point)
    sp = isnothing(section_point) ? x0 : section_point
    section = PoincaréSection(sp)
    
    # Find crossings
    crossings = find_poincare_crossings(ds, x0, section;
                                        tspan=tspan, transient_time=transient_time,
                                        solver=solver, reltol=reltol, abstol=abstol)
    
    if length(crossings) < min_crossings
        return (false, Vector{Float64}[], NaN)
    end
    
    # Check if consecutive crossings return to same point (periodic orbit)
    for i in 2:length(crossings)
        x1, t1 = crossings[i-1]
        x2, t2 = crossings[i]
        
        if norm(x2 - x1) < crossing_tol
            # Found periodic orbit!
            period = t2 - t1
            
            # Get orbit points for one period
            prob = ODEProblem((u, p, t) -> ds.f(u), x1, (0.0, period))
            sol = solve(prob, solver; reltol=reltol, abstol=abstol, saveat=period/100)
            orbit_points = [copy(u) for u in sol.u]
            
            return (true, orbit_points, period)
        end
    end
    
    return (false, Vector{Float64}[], NaN)
end

#=============================================================================
                        High-Level Interface
=============================================================================#

"""
    classify_periodic_orbit(ds::DynamicalSystem, orbit_points::Vector{Vector{Float64}}, 
                            period::Float64; kwargs...)

Classify a periodic orbit by computing its Floquet multipliers.

Returns a fully characterised `PeriodicOrbit` object.
"""
function classify_periodic_orbit(ds::DynamicalSystem, orbit_points::Vector{Vector{Float64}}, 
                                  period::Float64; kwargs...)
    if isempty(orbit_points)
        return PeriodicOrbit(orbit_points, period)
    end
    
    x0 = orbit_points[1]
    
    try
        floquet = compute_floquet_multipliers(ds, x0, period; kwargs...)
        return PeriodicOrbit(orbit_points, period, floquet)
    catch
        return PeriodicOrbit(orbit_points, period)
    end
end

"""
    find_periodic_orbits(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Float64, Float64}};
                         n_samples=20, method=:auto,
                         tspan=(0.0, 200.0), transient_time=50.0,
                         recurrence_tol=1e-4, refine=true,
                         unique_tol=1e-3, classify=true,
                         solver=Tsit5(), kwargs...) where N

Find periodic orbits in a dynamical system within the given bounds.

# Arguments
- `ds`: The dynamical system
- `bounds`: Domain bounds as tuple of (min, max) for each dimension

# Keyword Arguments
- `n_samples`: Number of initial conditions to try (via Latin hypercube sampling)
- `method`: Detection method - `:auto`, `:recurrence`, `:poincare` (2D only)
- `tspan`: Time span for integration
- `transient_time`: Time to discard as transient
- `recurrence_tol`: Tolerance for detecting recurrence
- `refine`: Whether to refine found orbits using shooting method
- `unique_tol`: Tolerance for identifying duplicate orbits
- `classify`: Whether to compute Floquet multipliers for classification

# Returns
Vector of `PeriodicOrbit` objects.
"""
function find_periodic_orbits(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Float64, Float64}};
                               n_samples::Int=20,
                               method::Symbol=:auto,
                               tspan::Tuple{Float64,Float64}=(0.0, 200.0),
                               transient_time::Float64=50.0,
                               recurrence_tol::Float64=1e-4,
                               refine::Bool=true,
                               unique_tol::Float64=1e-3,
                               classify::Bool=true,
                               solver=Tsit5(),
                               reltol::Float64=1e-8, abstol::Float64=1e-10) where N
    
    if N != dimension(ds)
        error("Bounds dimension ($N) must match system dimension ($(dimension(ds)))")
    end
    
    # Generate initial conditions
    initial_conditions = latin_hypercube_sample(ds, bounds, n_samples)
    
    # Choose method
    actual_method = method
    if method == :auto
        actual_method = (N == 2) ? :poincare : :recurrence
    end
    
    # Collect candidate orbits
    candidates = PeriodicOrbit[]
    
    for x0 in initial_conditions
        found = false
        orbit_points = Vector{Float64}[]
        period = NaN
        
        if actual_method == :poincare && N == 2
            found, orbit_points, period = find_periodic_orbit_poincare(
                ds, x0; tspan=tspan, transient_time=transient_time,
                crossing_tol=recurrence_tol, solver=solver, reltol=reltol, abstol=abstol)
        else
            # First check for oscillation
            is_osc, est_period, _ = detect_oscillation(ds, x0; tspan=tspan, solver=solver)
            
            if is_osc
                found, orbit_points, period = find_periodic_orbit_recurrence(
                    ds, x0; tspan=tspan, transient_time=transient_time,
                    recurrence_tol=recurrence_tol, solver=solver, reltol=reltol, abstol=abstol)
            end
        end
        
        if found && !isempty(orbit_points)
            # Optionally refine
            if refine
                success, refined_x0, refined_points = refine_periodic_orbit(
                    ds, orbit_points[1], period; solver=solver, reltol=reltol, abstol=abstol)
                if success
                    orbit_points = refined_points
                end
            end
            
            # Classify if requested
            if classify
                orbit = classify_periodic_orbit(ds, orbit_points, period; solver=solver)
            else
                orbit = PeriodicOrbit(orbit_points, period)
            end
            
            push!(candidates, orbit)
        end
    end
    
    # Remove duplicates
    orbits = unique_periodic_orbits(candidates; tol=unique_tol)
    
    return orbits
end

"""
    unique_periodic_orbits(orbits::Vector{PeriodicOrbit}; tol=1e-3)

Remove duplicate periodic orbits based on their representative points.

Two orbits are considered duplicates if a point on one orbit is within `tol` 
of a point on the other orbit.
"""
function unique_periodic_orbits(orbits::Vector{PeriodicOrbit}; tol::Float64=1e-3)
    if isempty(orbits)
        return PeriodicOrbit[]
    end
    
    unique = PeriodicOrbit[orbits[1]]
    
    for orbit in orbits[2:end]
        is_duplicate = false
        
        if !isempty(orbit.points)
            x_rep = orbit.points[1]
            
            for existing in unique
                if !isempty(existing.points)
                    # Check if representative point is close to any point on existing orbit
                    for x_exist in existing.points
                        if norm(x_rep - x_exist) < tol
                            is_duplicate = true
                            break
                        end
                    end
                end
                if is_duplicate
                    break
                end
            end
        end
        
        if !is_duplicate
            push!(unique, orbit)
        end
    end
    
    return unique
end

#=============================================================================
                        Query Functions
=============================================================================#

"""
    has_periodic_orbits(ds::DynamicalSystem, bounds; n_samples=30, 
                        quick_check=true, kwargs...)

Check whether a dynamical system appears to have periodic orbits within the given bounds.

This is a convenience function for classification: systems with periodic orbits
cannot be gradient or gradient-like.

# Arguments
- `quick_check`: If true, returns as soon as any periodic orbit is found
"""
function has_periodic_orbits(ds::DynamicalSystem, bounds::NTuple{N, Tuple{Float64, Float64}};
                              n_samples::Int=30,
                              quick_check::Bool=true,
                              tspan::Tuple{Float64,Float64}=(0.0, 200.0),
                              transient_time::Float64=60.0,
                              recurrence_tol::Float64=1e-2,
                              kwargs...) where N
    
    if quick_check
        # Just need to find one
        initial_conditions = latin_hypercube_sample(ds, bounds, n_samples)
        
        for x0 in initial_conditions
            is_osc, _, _ = detect_oscillation(ds, x0; tspan=tspan, transient_fraction=0.4)
            if is_osc
                # Verify with recurrence
                found, _, _ = find_periodic_orbit_recurrence(
                    ds, x0; 
                    tspan=tspan, 
                    transient_time=transient_time,
                    recurrence_tol=recurrence_tol
                )
                if found
                    return true
                end
            end
        end
        return false
    else
        orbits = find_periodic_orbits(ds, bounds; n_samples=n_samples, tspan=tspan, 
                                       transient_time=transient_time, recurrence_tol=recurrence_tol, kwargs...)
        return !isempty(orbits)
    end
end

"""
    all_periodic_orbits_hyperbolic(orbits::Vector{PeriodicOrbit})

Check if all periodic orbits in the collection are hyperbolic.

This is a necessary condition for Morse-Smale systems.
Returns `true` if the collection is empty (vacuously true).
"""
function all_periodic_orbits_hyperbolic(orbits::Vector{PeriodicOrbit})
    return all(orbit -> orbit.is_hyperbolic, orbits)
end

"""
    stable_periodic_orbits(orbits::Vector{PeriodicOrbit})

Filter to keep only stable periodic orbits.
"""
function stable_periodic_orbits(orbits::Vector{PeriodicOrbit})
    return filter(orbit -> orbit.is_stable, orbits)
end

"""
    unstable_periodic_orbits(orbits::Vector{PeriodicOrbit})

Filter to keep only unstable (including saddle) periodic orbits.
"""
function unstable_periodic_orbits(orbits::Vector{PeriodicOrbit})
    return filter(orbit -> !orbit.is_stable, orbits)
end

"""
    count_periodic_orbit_types(orbits::Vector{PeriodicOrbit})

Count the number of orbits of each type.
"""
function count_periodic_orbit_types(orbits::Vector{PeriodicOrbit})
    counts = Dict{PeriodicOrbitType, Int}()
    for orbit in orbits
        counts[orbit.type] = get(counts, orbit.type, 0) + 1
    end
    return counts
end
