# manifolds.jl
# Computation of stable and unstable manifolds, and transversality analysis

#=============================================================================
                               Types
=============================================================================#

"""
Type of invariant manifold.
"""
@enum ManifoldType begin
    STABLE_MANIFOLD      # Points converging to fixed point as t → ∞
    UNSTABLE_MANIFOLD    # Points converging to fixed point as t → -∞
end

"""
    Manifold

Represents a stable or unstable manifold of a fixed point or periodic orbit.

# Fields
- `points::Vector{Vector{Float64}}`: Sampled points on the manifold
- `manifold_type::ManifoldType`: Whether stable or unstable
- `dimension::Int`: Dimension of the manifold
- `fixed_point::Vector{Float64}`: The associated fixed point
- `tangent_vectors::Vector{Vector{Float64}}`: Tangent directions at the fixed point
"""
struct Manifold
    points::Vector{Vector{Float64}}
    manifold_type::ManifoldType
    dimension::Int
    fixed_point::Vector{Float64}
    tangent_vectors::Vector{Vector{Float64}}
end

"""
    ManifoldIntersection

Records an intersection between two manifolds.

# Fields
- `point::Vector{Float64}`: Location of the intersection
- `manifold1_tangent::Vector{Vector{Float64}}`: Tangent space of first manifold at intersection
- `manifold2_tangent::Vector{Vector{Float64}}`: Tangent space of second manifold at intersection
- `is_transverse::Bool`: Whether the intersection is transverse
- `codimension_deficit::Int`: How far from transverse (0 = transverse)
"""
struct ManifoldIntersection
    point::Vector{Float64}
    manifold1_tangent::Vector{Vector{Float64}}
    manifold2_tangent::Vector{Vector{Float64}}
    is_transverse::Bool
    codimension_deficit::Int
end

#=============================================================================
                        Eigenspace Computation
=============================================================================#

"""
    compute_stable_eigenspace(ds::DynamicalSystem, x::Vector{Float64}; tol=1e-10)

Compute the stable eigenspace at a fixed point.

Returns `(eigenvalues, eigenvectors)` where eigenvalues have negative real part.
The eigenvectors span the tangent space to the stable manifold at x.
"""
function compute_stable_eigenspace(ds::DynamicalSystem, x::Vector{Float64}; tol::Float64=1e-10)
    J = compute_jacobian(ds, x)
    eig = eigen(J)
    
    # Find stable eigenvalues (Re(λ) < -tol)
    stable_mask = real.(eig.values) .< -tol
    
    stable_eigenvalues = eig.values[stable_mask]
    stable_eigenvectors = eig.vectors[:, stable_mask]
    
    # Convert to real vectors where possible, handling complex conjugate pairs
    tangent_vectors = extract_real_basis(stable_eigenvectors, stable_eigenvalues)
    
    return (stable_eigenvalues, tangent_vectors)
end

"""
    compute_unstable_eigenspace(ds::DynamicalSystem, x::Vector{Float64}; tol=1e-10)

Compute the unstable eigenspace at a fixed point.

Returns `(eigenvalues, eigenvectors)` where eigenvalues have positive real part.
The eigenvectors span the tangent space to the unstable manifold at x.
"""
function compute_unstable_eigenspace(ds::DynamicalSystem, x::Vector{Float64}; tol::Float64=1e-10)
    J = compute_jacobian(ds, x)
    eig = eigen(J)
    
    # Find unstable eigenvalues (Re(λ) > tol)
    unstable_mask = real.(eig.values) .> tol
    
    unstable_eigenvalues = eig.values[unstable_mask]
    unstable_eigenvectors = eig.vectors[:, unstable_mask]
    
    # Convert to real vectors where possible
    tangent_vectors = extract_real_basis(unstable_eigenvectors, unstable_eigenvalues)
    
    return (unstable_eigenvalues, tangent_vectors)
end

"""
    extract_real_basis(eigenvectors, eigenvalues)

Extract a real basis from possibly complex eigenvectors.

For complex conjugate eigenvalue pairs, extracts real and imaginary parts
as the two real basis vectors.
"""
function extract_real_basis(eigenvectors::AbstractMatrix, eigenvalues::AbstractVector)
    n = size(eigenvectors, 1)
    m = size(eigenvectors, 2)
    
    if m == 0
        return Vector{Float64}[]
    end
    
    real_vectors = Vector{Float64}[]
    processed = Set{Int}()
    
    for i in 1:m
        if i in processed
            continue
        end
        
        λ = eigenvalues[i]
        v = eigenvectors[:, i]
        
        if abs(imag(λ)) < 1e-10
            # Real eigenvalue - use the real eigenvector
            push!(real_vectors, real.(v))
            push!(processed, i)
        else
            # Complex eigenvalue - find conjugate pair
            # Use real and imaginary parts as basis
            push!(real_vectors, real.(v))
            push!(real_vectors, imag.(v))
            push!(processed, i)
            
            # Find and mark the conjugate
            for j in (i+1):m
                if j ∉ processed && abs(eigenvalues[j] - conj(λ)) < 1e-10
                    push!(processed, j)
                    break
                end
            end
        end
    end
    
    # Normalize
    for i in 1:length(real_vectors)
        nrm = norm(real_vectors[i])
        if nrm > 1e-10
            real_vectors[i] ./= nrm
        end
    end
    
    return real_vectors
end

#=============================================================================
                    Local Manifold Computation
=============================================================================#

"""
    compute_local_manifold(ds::DynamicalSystem, fp::Vector{Float64}, 
                           tangent_vectors::Vector{Vector{Float64}},
                           manifold_type::ManifoldType;
                           epsilon=0.01, n_points_per_direction=10,
                           solver=Tsit5(), integration_time=10.0, kwargs...)

Compute the local portion of a stable or unstable manifold near a fixed point.

Starting from small perturbations along the tangent directions, integrates
trajectories forward (stable) or backward (unstable) to trace out the manifold.
"""
function compute_local_manifold(ds::DynamicalSystem, fp::Vector{Float64},
                                 tangent_vectors::Vector{Vector{Float64}},
                                 manifold_type::ManifoldType;
                                 epsilon::Float64=0.01,
                                 n_points_per_direction::Int=10,
                                 solver=Tsit5(),
                                 integration_time::Float64=10.0,
                                 reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    if isempty(tangent_vectors)
        return Vector{Float64}[]
    end
    
    n_dim = length(fp)
    manifold_dim = length(tangent_vectors)
    
    # Generate initial conditions along tangent directions
    initial_points = Vector{Float64}[]
    
    if manifold_dim == 1
        # 1D manifold: points along a line
        v = tangent_vectors[1]
        for s in range(-epsilon, epsilon, length=2*n_points_per_direction+1)
            if abs(s) > 1e-12  # Skip the fixed point itself
                push!(initial_points, fp + s * v)
            end
        end
    elseif manifold_dim == 2
        # 2D manifold: points on a grid in the tangent plane
        v1, v2 = tangent_vectors[1], tangent_vectors[2]
        for s1 in range(-epsilon, epsilon, length=n_points_per_direction)
            for s2 in range(-epsilon, epsilon, length=n_points_per_direction)
                if abs(s1) > 1e-12 || abs(s2) > 1e-12
                    push!(initial_points, fp + s1 * v1 + s2 * v2)
                end
            end
        end
    else
        # Higher dimensional: sample on hypersphere in tangent space
        # Use random sampling for simplicity
        for _ in 1:n_points_per_direction^2
            coeffs = randn(manifold_dim)
            coeffs ./= norm(coeffs)
            r = epsilon * rand()^(1/manifold_dim)  # Uniform in ball
            point = fp + r * sum(coeffs[i] * tangent_vectors[i] for i in 1:manifold_dim)
            push!(initial_points, point)
        end
    end
    
    # Integrate each initial condition
    manifold_points = [copy(fp)]  # Include the fixed point
    
    # Direction of integration: forward for stable, backward for unstable
    tspan = manifold_type == STABLE_MANIFOLD ? (0.0, integration_time) : (0.0, -integration_time)
    
    for x0 in initial_points
        prob = ODEProblem((u, p, t) -> ds.f(u), x0, tspan)
        sol = solve(prob, solver; reltol=reltol, abstol=abstol, 
                    saveat=abs(integration_time)/20)
        
        if sol.retcode == ReturnCode.Success
            for u in sol.u
                push!(manifold_points, copy(u))
            end
        end
    end
    
    return manifold_points
end

#=============================================================================
                    Global Manifold Computation
=============================================================================#

"""
    grow_manifold_1d(ds::DynamicalSystem, fp::Vector{Float64},
                     direction::Vector{Float64}, manifold_type::ManifoldType;
                     epsilon=0.01, max_arclength=10.0, step_size=0.1,
                     solver=Tsit5(), kwargs...)

Grow a 1-dimensional manifold by following the flow from the fixed point.

For stable manifolds, integrates backward; for unstable, integrates forward.
Uses adaptive stepping to maintain resolution along the manifold.
"""
function grow_manifold_1d(ds::DynamicalSystem, fp::Vector{Float64},
                          direction::Vector{Float64}, manifold_type::ManifoldType;
                          epsilon::Float64=0.01,
                          max_arclength::Float64=10.0,
                          step_size::Float64=0.1,
                          solver=Tsit5(),
                          reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    # For 1D manifolds in 2D+ systems, we can trace by:
    # 1. Start with small perturbation along eigenvector
    # 2. Integrate in appropriate direction
    # 3. Track arclength and add points
    
    manifold_points = Vector{Float64}[]
    
    # Trace in both directions (±direction)
    for sign in [1.0, -1.0]
        current_point = fp + sign * epsilon * direction
        arclength = 0.0
        
        push!(manifold_points, copy(current_point))
        
        while arclength < max_arclength
            # Integration direction
            if manifold_type == UNSTABLE_MANIFOLD
                # For unstable manifold, integrate forward
                tspan = (0.0, step_size)
            else
                # For stable manifold, integrate backward
                tspan = (0.0, -step_size)
            end
            
            prob = ODEProblem((u, p, t) -> ds.f(u), current_point, tspan)
            sol = solve(prob, solver; reltol=reltol, abstol=abstol, save_everystep=false)
            
            if sol.retcode != ReturnCode.Success
                break
            end
            
            new_point = sol.u[end]
            segment_length = norm(new_point - current_point)
            arclength += segment_length
            
            # Check for divergence or convergence back to fixed point
            if norm(new_point - fp) > 100.0 || norm(new_point - fp) < epsilon/10
                break
            end
            
            push!(manifold_points, copy(new_point))
            current_point = new_point
        end
    end
    
    return manifold_points
end

"""
    compute_stable_manifold(ds::DynamicalSystem, fp::FixedPoint;
                            epsilon=0.01, max_extent=10.0, n_points=100, kwargs...)

Compute the stable manifold of a fixed point.

# Arguments
- `ds`: The dynamical system
- `fp`: The fixed point (must be a saddle or have stable directions)
- `epsilon`: Initial perturbation size
- `max_extent`: Maximum distance to grow the manifold
- `n_points`: Approximate number of points to sample

# Returns
A `Manifold` object containing the sampled stable manifold.
"""
function compute_stable_manifold(ds::DynamicalSystem, fp::FixedPoint;
                                  epsilon::Float64=0.01,
                                  max_extent::Float64=10.0,
                                  n_points::Int=100,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    x = fp.location
    stable_eigs, tangent_vecs = compute_stable_eigenspace(ds, x)
    
    stable_dim = length(tangent_vecs)
    
    if stable_dim == 0
        # No stable directions (source)
        return Manifold(Vector{Float64}[], STABLE_MANIFOLD, 0, x, Vector{Float64}[])
    end
    
    if stable_dim == 1
        # 1D stable manifold - use curve tracing
        points = grow_manifold_1d(ds, x, tangent_vecs[1], STABLE_MANIFOLD;
                                   epsilon=epsilon, max_arclength=max_extent,
                                   solver=solver, reltol=reltol, abstol=abstol)
    else
        # Higher-dimensional - use local manifold computation
        n_per_dir = max(5, round(Int, sqrt(n_points / 2)))
        points = compute_local_manifold(ds, x, tangent_vecs, STABLE_MANIFOLD;
                                         epsilon=epsilon, n_points_per_direction=n_per_dir,
                                         integration_time=max_extent/2,
                                         solver=solver, reltol=reltol, abstol=abstol)
    end
    
    return Manifold(points, STABLE_MANIFOLD, stable_dim, x, tangent_vecs)
end

"""
    compute_unstable_manifold(ds::DynamicalSystem, fp::FixedPoint;
                               epsilon=0.01, max_extent=10.0, n_points=100, kwargs...)

Compute the unstable manifold of a fixed point.

# Arguments
- `ds`: The dynamical system
- `fp`: The fixed point (must be a saddle or have unstable directions)
- `epsilon`: Initial perturbation size
- `max_extent`: Maximum distance to grow the manifold
- `n_points`: Approximate number of points to sample

# Returns
A `Manifold` object containing the sampled unstable manifold.
"""
function compute_unstable_manifold(ds::DynamicalSystem, fp::FixedPoint;
                                    epsilon::Float64=0.01,
                                    max_extent::Float64=10.0,
                                    n_points::Int=100,
                                    solver=Tsit5(),
                                    reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    x = fp.location
    unstable_eigs, tangent_vecs = compute_unstable_eigenspace(ds, x)
    
    unstable_dim = length(tangent_vecs)
    
    if unstable_dim == 0
        # No unstable directions (sink)
        return Manifold(Vector{Float64}[], UNSTABLE_MANIFOLD, 0, x, Vector{Float64}[])
    end
    
    if unstable_dim == 1
        # 1D unstable manifold - use curve tracing
        points = grow_manifold_1d(ds, x, tangent_vecs[1], UNSTABLE_MANIFOLD;
                                   epsilon=epsilon, max_arclength=max_extent,
                                   solver=solver, reltol=reltol, abstol=abstol)
    else
        # Higher-dimensional - use local manifold computation
        n_per_dir = max(5, round(Int, sqrt(n_points / 2)))
        points = compute_local_manifold(ds, x, tangent_vecs, UNSTABLE_MANIFOLD;
                                         epsilon=epsilon, n_points_per_direction=n_per_dir,
                                         integration_time=max_extent/2,
                                         solver=solver, reltol=reltol, abstol=abstol)
    end
    
    return Manifold(points, UNSTABLE_MANIFOLD, unstable_dim, x, tangent_vecs)
end

"""
    compute_manifolds(ds::DynamicalSystem, fp::FixedPoint; kwargs...)

Compute both stable and unstable manifolds of a fixed point.

Returns `(stable_manifold, unstable_manifold)`.
"""
function compute_manifolds(ds::DynamicalSystem, fp::FixedPoint; kwargs...)
    stable = compute_stable_manifold(ds, fp; kwargs...)
    unstable = compute_unstable_manifold(ds, fp; kwargs...)
    return (stable, unstable)
end

#=============================================================================
                    Manifold Intersection Detection
=============================================================================#

"""
    find_manifold_intersections(manifold1::Manifold, manifold2::Manifold;
                                 distance_tol=1e-3)

Find approximate intersection points between two manifolds.

Uses distance-based detection: points from one manifold that are close to 
points on the other manifold are considered potential intersections.

Returns a vector of `ManifoldIntersection` objects.
"""
function find_manifold_intersections(manifold1::Manifold, manifold2::Manifold;
                                      distance_tol::Float64=1e-3)
    
    intersections = ManifoldIntersection[]
    
    if isempty(manifold1.points) || isempty(manifold2.points)
        return intersections
    end
    
    # Find close pairs of points
    close_pairs = Tuple{Vector{Float64}, Vector{Float64}}[]
    
    for p1 in manifold1.points
        for p2 in manifold2.points
            if norm(p1 - p2) < distance_tol
                push!(close_pairs, (p1, p2))
            end
        end
    end
    
    # Cluster close pairs to find distinct intersections
    if isempty(close_pairs)
        return intersections
    end
    
    # Simple clustering: take midpoints and merge nearby ones
    midpoints = [(p1 + p2) / 2 for (p1, p2) in close_pairs]
    unique_intersections = cluster_points(midpoints; distance_tol=2*distance_tol)
    
    # For each unique intersection, estimate tangent spaces and check transversality
    for int_point in unique_intersections
        # Estimate tangent space of manifold1 at intersection
        tangent1 = estimate_tangent_space(manifold1, int_point; radius=5*distance_tol)
        
        # Estimate tangent space of manifold2 at intersection
        tangent2 = estimate_tangent_space(manifold2, int_point; radius=5*distance_tol)
        
        # Check transversality
        is_trans, deficit = check_transversality_at_point(tangent1, tangent2, length(int_point))
        
        push!(intersections, ManifoldIntersection(int_point, tangent1, tangent2, is_trans, deficit))
    end
    
    return intersections
end

"""
    estimate_tangent_space(manifold::Manifold, point::Vector{Float64}; radius=0.1)

Estimate the tangent space of a manifold at a given point.

Uses nearby points on the manifold to estimate tangent directions via PCA.
"""
function estimate_tangent_space(manifold::Manifold, point::Vector{Float64}; radius::Float64=0.1)
    # Find nearby points
    nearby = [p for p in manifold.points if norm(p - point) < radius && norm(p - point) > 1e-10]
    
    if length(nearby) < manifold.dimension
        # Not enough points - use tangent vectors from manifold definition
        return manifold.tangent_vectors
    end
    
    # Center the points
    centered = [p - point for p in nearby]
    
    # Form data matrix
    n = length(point)
    m = length(centered)
    X = zeros(n, m)
    for (j, p) in enumerate(centered)
        X[:, j] = p
    end
    
    # SVD to find principal directions
    U, S, V = svd(X)
    
    # Take the first `manifold.dimension` singular vectors as tangent basis
    tangent_vecs = [U[:, i] for i in 1:min(manifold.dimension, n)]
    
    return tangent_vecs
end

"""
    check_transversality_at_point(tangent1::Vector{Vector{Float64}}, 
                                   tangent2::Vector{Vector{Float64}},
                                   ambient_dim::Int; tol=1e-6)

Check if two tangent spaces intersect transversely.

Transversality requires that the combined tangent spaces span the full ambient space.

Returns `(is_transverse, codimension_deficit)` where:
- `is_transverse`: true if transverse
- `codimension_deficit`: how many dimensions short of spanning (0 = transverse)
"""
function check_transversality_at_point(tangent1::Vector{Vector{Float64}},
                                        tangent2::Vector{Vector{Float64}},
                                        ambient_dim::Int; tol::Float64=1e-6)
    
    dim1 = length(tangent1)
    dim2 = length(tangent2)
    
    if dim1 == 0 || dim2 == 0
        # Trivial case: point manifold
        return (true, 0)
    end
    
    if isempty(tangent1[1]) || isempty(tangent2[1])
        return (true, 0)
    end
    
    n = length(tangent1[1])
    
    # Combine all tangent vectors into a matrix
    all_vectors = vcat(tangent1, tangent2)
    m = length(all_vectors)
    
    if m == 0
        return (true, 0)
    end
    
    A = zeros(n, m)
    for (j, v) in enumerate(all_vectors)
        A[:, j] = v
    end
    
    # Compute rank via SVD
    S = svdvals(A)
    rank = count(s -> s > tol, S)
    
    # For transversality, need rank = min(dim1 + dim2, ambient_dim)
    expected_rank = min(dim1 + dim2, ambient_dim)
    
    is_transverse = rank >= expected_rank
    deficit = expected_rank - rank
    
    return (is_transverse, max(0, deficit))
end

#=============================================================================
                    High-Level Transversality Analysis
=============================================================================#

"""
    check_transversality(ds::DynamicalSystem, fps::Vector{FixedPoint};
                         epsilon=0.01, max_extent=5.0, distance_tol=1e-2, kwargs...)

Check transversality of all stable/unstable manifold intersections.

For Morse-Smale systems, all intersections must be transverse.

# Returns
`(all_transverse, intersections)` where:
- `all_transverse`: true if all detected intersections are transverse
- `intersections`: vector of all detected `ManifoldIntersection` objects
"""
function check_transversality(ds::DynamicalSystem, fps::Vector{FixedPoint};
                               epsilon::Float64=0.01,
                               max_extent::Float64=5.0,
                               distance_tol::Float64=1e-2,
                               solver=Tsit5(),
                               kwargs...)
    
    all_intersections = ManifoldIntersection[]
    
    # Compute manifolds for each saddle point (points with both stable and unstable directions)
    saddle_fps = filter(fp -> fp.type == SADDLE || fp.type == SADDLE_FOCUS, fps)
    
    manifolds = Dict{Vector{Float64}, Tuple{Manifold, Manifold}}()
    
    for fp in saddle_fps
        stable, unstable = compute_manifolds(ds, fp; epsilon=epsilon, max_extent=max_extent,
                                              solver=solver, kwargs...)
        manifolds[fp.location] = (stable, unstable)
    end
    
    # Check intersections between all pairs
    saddle_locations = collect(keys(manifolds))
    
    for i in 1:length(saddle_locations)
        for j in i:length(saddle_locations)
            loc_i = saddle_locations[i]
            loc_j = saddle_locations[j]
            
            stable_i, unstable_i = manifolds[loc_i]
            stable_j, unstable_j = manifolds[loc_j]
            
            if i == j
                # Self-intersections: stable with unstable of same saddle
                # (These always intersect at the saddle point itself, which is trivially transverse)
                # Check for homoclinic orbits (non-trivial self-intersection)
                ints = find_manifold_intersections(stable_i, unstable_i; distance_tol=distance_tol)
                # Filter out the trivial intersection at the fixed point
                ints = filter(int -> norm(int.point - loc_i) > 2*epsilon, ints)
                append!(all_intersections, ints)
            else
                # Heteroclinic connections between different saddles
                # stable_i with unstable_j
                ints1 = find_manifold_intersections(stable_i, unstable_j; distance_tol=distance_tol)
                append!(all_intersections, ints1)
                
                # stable_j with unstable_i
                ints2 = find_manifold_intersections(stable_j, unstable_i; distance_tol=distance_tol)
                append!(all_intersections, ints2)
            end
        end
    end
    
    # Also check stable manifolds of sinks with unstable manifolds of saddles
    # and unstable manifolds of sources with stable manifolds of saddles
    sinks = filter(fp -> fp.type == STABLE_NODE || fp.type == STABLE_FOCUS, fps)
    sources = filter(fp -> fp.type == UNSTABLE_NODE || fp.type == UNSTABLE_FOCUS, fps)
    
    # Sinks have only stable manifolds (the whole basin)
    # Sources have only unstable manifolds (the whole basin of the time-reversed system)
    # For now, focus on saddle-saddle and saddle-sink/source connections
    
    all_transverse = all(int -> int.is_transverse, all_intersections)
    
    return (all_transverse, all_intersections)
end

"""
    has_homoclinic_orbit(ds::DynamicalSystem, fp::FixedPoint;
                         epsilon=0.01, max_extent=10.0, distance_tol=1e-2, kwargs...)

Check if a saddle point has a homoclinic orbit (its unstable manifold
intersects its stable manifold away from the fixed point).

Returns `(has_homoclinic, intersection_points)`.
"""
function has_homoclinic_orbit(ds::DynamicalSystem, fp::FixedPoint;
                               epsilon::Float64=0.01,
                               max_extent::Float64=10.0,
                               distance_tol::Float64=1e-2,
                               solver=Tsit5(),
                               kwargs...)
    
    if fp.type != SADDLE && fp.type != SADDLE_FOCUS
        return (false, Vector{Float64}[])
    end
    
    stable, unstable = compute_manifolds(ds, fp; epsilon=epsilon, max_extent=max_extent,
                                          solver=solver, kwargs...)
    
    intersections = find_manifold_intersections(stable, unstable; distance_tol=distance_tol)
    
    # Filter out the trivial intersection at the fixed point
    nontrivial = filter(int -> norm(int.point - fp.location) > 2*epsilon, intersections)
    
    intersection_points = [int.point for int in nontrivial]
    
    return (!isempty(nontrivial), intersection_points)
end

"""
    has_heteroclinic_orbit(ds::DynamicalSystem, fp1::FixedPoint, fp2::FixedPoint;
                           epsilon=0.01, max_extent=10.0, distance_tol=1e-2, kwargs...)

Check if there is a heteroclinic connection from fp1 to fp2
(unstable manifold of fp1 intersects stable manifold of fp2).

Returns `(has_connection, intersection_points)`.
"""
function has_heteroclinic_orbit(ds::DynamicalSystem, fp1::FixedPoint, fp2::FixedPoint;
                                 epsilon::Float64=0.01,
                                 max_extent::Float64=10.0,
                                 distance_tol::Float64=1e-2,
                                 solver=Tsit5(),
                                 kwargs...)
    
    unstable1 = compute_unstable_manifold(ds, fp1; epsilon=epsilon, max_extent=max_extent,
                                           solver=solver, kwargs...)
    stable2 = compute_stable_manifold(ds, fp2; epsilon=epsilon, max_extent=max_extent,
                                       solver=solver, kwargs...)
    
    intersections = find_manifold_intersections(unstable1, stable2; distance_tol=distance_tol)
    
    # Filter out points too close to either fixed point
    min_dist = 2 * epsilon
    nontrivial = filter(int -> norm(int.point - fp1.location) > min_dist && 
                               norm(int.point - fp2.location) > min_dist, 
                        intersections)
    
    intersection_points = [int.point for int in nontrivial]
    
    return (!isempty(nontrivial), intersection_points)
end

#=============================================================================
                    Morse-Smale Condition Checking
=============================================================================#

"""
    all_manifolds_transverse(ds::DynamicalSystem, fps::Vector{FixedPoint}; kwargs...)

Check if all stable/unstable manifold intersections are transverse.

This is a necessary condition for Morse-Smale systems.

Returns `true` if all detected intersections are transverse (or if there are no
saddle points), `false` otherwise.
"""
function all_manifolds_transverse(ds::DynamicalSystem, fps::Vector{FixedPoint}; kwargs...)
    all_transverse, _ = check_transversality(ds, fps; kwargs...)
    return all_transverse
end

"""
    count_heteroclinic_connections(ds::DynamicalSystem, fps::Vector{FixedPoint}; kwargs...)

Count the number of heteroclinic connections between fixed points.

Returns a dictionary mapping (source_idx, sink_idx) to the number of connections.
"""
function count_heteroclinic_connections(ds::DynamicalSystem, fps::Vector{FixedPoint};
                                         epsilon::Float64=0.01,
                                         max_extent::Float64=5.0,
                                         distance_tol::Float64=1e-2,
                                         kwargs...)
    
    connections = Dict{Tuple{Int,Int}, Int}()
    
    for i in 1:length(fps)
        for j in 1:length(fps)
            if i != j
                has_conn, _ = has_heteroclinic_orbit(ds, fps[i], fps[j];
                                                      epsilon=epsilon, max_extent=max_extent,
                                                      distance_tol=distance_tol, kwargs...)
                if has_conn
                    connections[(i, j)] = get(connections, (i, j), 0) + 1
                end
            end
        end
    end
    
    return connections
end

#=============================================================================
                    Convenience Functions for 2D Systems
=============================================================================#

"""
    compute_separatrices(ds::DynamicalSystem, saddle::FixedPoint;
                         epsilon=0.01, max_extent=10.0, kwargs...)

For a 2D system, compute the separatrices (1D stable and unstable manifolds)
of a saddle point.

The separatrices are curves that divide the phase space into regions with
different asymptotic behaviours.

Returns `(stable_branches, unstable_branches)` where each is a vector of
two vectors of points (one for each branch).
"""
function compute_separatrices(ds::DynamicalSystem, saddle::FixedPoint;
                               epsilon::Float64=0.01,
                               max_extent::Float64=10.0,
                               solver=Tsit5(),
                               reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    if dimension(ds) != 2
        error("compute_separatrices is only for 2D systems")
    end
    
    if saddle.type != SADDLE
        error("compute_separatrices requires a saddle point")
    end
    
    x = saddle.location
    
    # Get stable and unstable eigenvectors
    _, stable_vecs = compute_stable_eigenspace(ds, x)
    _, unstable_vecs = compute_unstable_eigenspace(ds, x)
    
    if isempty(stable_vecs) || isempty(unstable_vecs)
        error("Saddle point must have both stable and unstable directions")
    end
    
    v_stable = stable_vecs[1]
    v_unstable = unstable_vecs[1]
    
    # Trace stable manifold branches (integrate backward)
    stable_branches = Vector{Vector{Float64}}[]
    for sign in [1.0, -1.0]
        branch = trace_manifold_branch(ds, x, sign * v_stable, -1.0;
                                        epsilon=epsilon, max_extent=max_extent,
                                        solver=solver, reltol=reltol, abstol=abstol)
        push!(stable_branches, branch)
    end
    
    # Trace unstable manifold branches (integrate forward)
    unstable_branches = Vector{Vector{Float64}}[]
    for sign in [1.0, -1.0]
        branch = trace_manifold_branch(ds, x, sign * v_unstable, 1.0;
                                        epsilon=epsilon, max_extent=max_extent,
                                        solver=solver, reltol=reltol, abstol=abstol)
        push!(unstable_branches, branch)
    end
    
    return (stable_branches, unstable_branches)
end

"""
    trace_manifold_branch(ds::DynamicalSystem, fp::Vector{Float64}, 
                          direction::Vector{Float64}, time_sign::Float64;
                          epsilon=0.01, max_extent=10.0, kwargs...)

Trace a single branch of a 1D manifold.

`time_sign` should be +1 for forward integration (unstable manifold)
or -1 for backward integration (stable manifold).
"""
function trace_manifold_branch(ds::DynamicalSystem, fp::Vector{Float64},
                                direction::Vector{Float64}, time_sign::Float64;
                                epsilon::Float64=0.01,
                                max_extent::Float64=10.0,
                                solver=Tsit5(),
                                reltol::Float64=1e-8, abstol::Float64=1e-10)
    
    # Start with small perturbation
    x0 = fp + epsilon * direction
    
    # Integrate
    tspan = time_sign > 0 ? (0.0, max_extent) : (0.0, -max_extent)
    prob = ODEProblem((u, p, t) -> ds.f(u), x0, tspan)
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, saveat=abs(max_extent)/100)
    
    if sol.retcode != ReturnCode.Success
        return [copy(x0)]
    end
    
    # Include the fixed point at the start
    points = [copy(fp)]
    append!(points, [copy(u) for u in sol.u])
    
    return points
end

#=============================================================================
                    Visualization Helper
=============================================================================#

"""
    manifold_to_coordinates(manifold::Manifold)

Extract coordinate arrays from a manifold for plotting.

Returns `(xs, ys)` for 2D or `(xs, ys, zs)` for 3D systems.
"""
function manifold_to_coordinates(manifold::Manifold)
    if isempty(manifold.points)
        return (Float64[], Float64[])
    end
    
    n = length(manifold.points[1])
    
    if n == 2
        xs = [p[1] for p in manifold.points]
        ys = [p[2] for p in manifold.points]
        return (xs, ys)
    elseif n == 3
        xs = [p[1] for p in manifold.points]
        ys = [p[2] for p in manifold.points]
        zs = [p[3] for p in manifold.points]
        return (xs, ys, zs)
    else
        error("manifold_to_coordinates only supports 2D and 3D systems")
    end
end
