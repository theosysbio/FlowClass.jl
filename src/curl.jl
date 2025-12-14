"""
    curl.jl

Functions for computing and analysing the curl (rotational component) of vector fields.

For a dynamical system dx/dt = F(x), the curl captures the non-gradient component
of the dynamics. A gradient system has zero curl everywhere.

In n dimensions, the curl is represented by the antisymmetric part of the Jacobian.
In 3D, this corresponds to the classical curl vector ∇ × F.
"""

using LinearAlgebra

#=============================================================================
    Matrix Decomposition
=============================================================================#

"""
    compute_antisymmetric_part(J::AbstractMatrix)

Compute the antisymmetric part of a matrix: (J - Jᵀ)/2.

For the Jacobian of a vector field, this represents the "curl" or rotational
component of the dynamics. It is zero for gradient systems.

# Arguments
- `J::AbstractMatrix`: A square matrix

# Returns
- `Matrix`: The antisymmetric part (J - Jᵀ)/2

# Examples
```julia
J = [-1.0 0.5; -0.5 -1.0]
A = compute_antisymmetric_part(J)
# A = [0.0 0.5; -0.5 0.0]
```
"""
function compute_antisymmetric_part(J::AbstractMatrix)
    size(J, 1) == size(J, 2) || throw(ArgumentError("Matrix must be square"))
    return (J - transpose(J)) / 2
end


"""
    compute_symmetric_part(J::AbstractMatrix)

Compute the symmetric part of a matrix: (J + Jᵀ)/2.

For the Jacobian of a vector field, this represents the "gradient" component
of the dynamics. For a pure gradient system, J equals its symmetric part.

# Arguments
- `J::AbstractMatrix`: A square matrix

# Returns
- `Matrix`: The symmetric part (J + Jᵀ)/2

# Examples
```julia
J = [-1.0 0.5; -0.5 -1.0]
S = compute_symmetric_part(J)
# S = [-1.0 0.0; 0.0 -1.0]
```
"""
function compute_symmetric_part(J::AbstractMatrix)
    size(J, 1) == size(J, 2) || throw(ArgumentError("Matrix must be square"))
    return (J + transpose(J)) / 2
end


"""
    decompose_jacobian(J::AbstractMatrix)

Decompose a Jacobian into symmetric and antisymmetric parts.

Returns (S, A) where J = S + A, with S symmetric and A antisymmetric.

For dynamical systems interpretation:
- S represents the gradient-like component (potential-driven)
- A represents the curl component (rotational)

# Arguments
- `J::AbstractMatrix`: A square matrix (typically a Jacobian)

# Returns
- `Tuple{Matrix, Matrix}`: (symmetric_part, antisymmetric_part)

# Examples
```julia
J = [-1.0 0.5; -0.5 -1.0]
S, A = decompose_jacobian(J)
@assert J ≈ S + A
@assert S ≈ transpose(S)  # symmetric
@assert A ≈ -transpose(A)  # antisymmetric
```
"""
function decompose_jacobian(J::AbstractMatrix)
    S = compute_symmetric_part(J)
    A = compute_antisymmetric_part(J)
    return S, A
end


"""
    decompose_jacobian(ds::DynamicalSystem, x::AbstractVector)

Decompose the Jacobian of a dynamical system at point x into symmetric and 
antisymmetric parts.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `x::AbstractVector`: The point at which to evaluate

# Returns
- `Tuple{Matrix, Matrix}`: (symmetric_part, antisymmetric_part)
"""
function decompose_jacobian(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return decompose_jacobian(J)
end


#=============================================================================
    Curl Magnitude (n-dimensional)
=============================================================================#

"""
    curl_magnitude(J::AbstractMatrix)

Compute the magnitude of the curl (antisymmetric part) of a Jacobian matrix.

This returns the Frobenius norm of (J - Jᵀ)/2, which is zero for gradient
systems and positive for systems with rotational dynamics.

# Arguments
- `J::AbstractMatrix`: A square matrix (typically a Jacobian)

# Returns
- `Float64`: The Frobenius norm of the antisymmetric part

# Note
This is mathematically equivalent to `jacobian_symmetry_error(J)`.

# Examples
```julia
# Gradient system Jacobian (symmetric)
J_grad = [-2.0 0.5; 0.5 -1.0]
curl_magnitude(J_grad)  # ≈ 0.0

# System with rotation
J_rot = [-1.0 1.0; -1.0 -1.0]
curl_magnitude(J_rot)  # > 0
```
"""
function curl_magnitude(J::AbstractMatrix)
    A = compute_antisymmetric_part(J)
    return norm(A)
end


"""
    curl_magnitude(ds::DynamicalSystem, x::AbstractVector)

Compute the curl magnitude of a dynamical system at point x.
"""
function curl_magnitude(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return curl_magnitude(J)
end


"""
    curl_magnitude(f, x::AbstractVector)

Compute the curl magnitude of a vector field function at point x.
"""
function curl_magnitude(f, x::AbstractVector)
    J = compute_jacobian(f, x)
    return curl_magnitude(J)
end


"""
    relative_curl_magnitude(J::AbstractMatrix)

Compute the curl magnitude relative to the Jacobian's overall magnitude.

Returns ‖(J - Jᵀ)/2‖ / ‖J‖, a scale-independent measure between 0 and 1.
- Returns 0 for symmetric matrices (gradient systems)
- Returns 1 for purely antisymmetric matrices
- Returns 0 for zero matrices

# Arguments
- `J::AbstractMatrix`: A square matrix (typically a Jacobian)

# Returns
- `Float64`: The relative curl magnitude (between 0 and 1)

# Examples
```julia
# Purely antisymmetric (all rotation, no gradient)
J_antisym = [0.0 1.0; -1.0 0.0]
relative_curl_magnitude(J_antisym)  # ≈ 1.0

# Mixed system
J_mixed = [-1.0 1.0; -1.0 -1.0]
relative_curl_magnitude(J_mixed)  # ≈ 0.5
```
"""
function relative_curl_magnitude(J::AbstractMatrix)
    J_norm = norm(J)
    if J_norm ≈ 0
        return 0.0
    end
    return curl_magnitude(J) / J_norm
end


"""
    relative_curl_magnitude(ds::DynamicalSystem, x::AbstractVector)

Compute the relative curl magnitude of a dynamical system at point x.
"""
function relative_curl_magnitude(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return relative_curl_magnitude(J)
end


"""
    relative_curl_magnitude(f, x::AbstractVector)

Compute the relative curl magnitude of a vector field function at point x.
"""
function relative_curl_magnitude(f, x::AbstractVector)
    J = compute_jacobian(f, x)
    return relative_curl_magnitude(J)
end


#=============================================================================
    3D Curl Vector
=============================================================================#

"""
    compute_curl_3d(J::AbstractMatrix)

Compute the classical curl vector for a 3D system from its Jacobian.

For a vector field F = [F₁, F₂, F₃], the curl is:
    ∇ × F = [∂F₃/∂x₂ - ∂F₂/∂x₃, ∂F₁/∂x₃ - ∂F₃/∂x₁, ∂F₂/∂x₁ - ∂F₁/∂x₂]

This is extracted from the Jacobian matrix J where J[i,j] = ∂Fᵢ/∂xⱼ.

# Arguments
- `J::AbstractMatrix`: A 3×3 Jacobian matrix

# Returns
- `Vector{Float64}`: The curl vector [curl_x, curl_y, curl_z]

# Throws
- `ArgumentError`: If the matrix is not 3×3

# Examples
```julia
# A rotation around the z-axis: F = [-y, x, 0]
# Jacobian: J = [0 -1 0; 1 0 0; 0 0 0]
# Curl: ∇ × F = [0, 0, 2]
J = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
curl = compute_curl_3d(J)
# curl ≈ [0.0, 0.0, 2.0]
```
"""
function compute_curl_3d(J::AbstractMatrix)
    size(J) == (3, 3) || throw(ArgumentError("3D curl requires a 3×3 matrix, got $(size(J))"))
    
    # curl = [∂F₃/∂x₂ - ∂F₂/∂x₃, ∂F₁/∂x₃ - ∂F₃/∂x₁, ∂F₂/∂x₁ - ∂F₁/∂x₂]
    # J[i,j] = ∂Fᵢ/∂xⱼ
    curl_x = J[3, 2] - J[2, 3]  # ∂F₃/∂x₂ - ∂F₂/∂x₃
    curl_y = J[1, 3] - J[3, 1]  # ∂F₁/∂x₃ - ∂F₃/∂x₁
    curl_z = J[2, 1] - J[1, 2]  # ∂F₂/∂x₁ - ∂F₁/∂x₂
    
    return [curl_x, curl_y, curl_z]
end


"""
    compute_curl_3d(ds::DynamicalSystem, x::AbstractVector)

Compute the classical curl vector for a 3D dynamical system at point x.

# Arguments
- `ds::DynamicalSystem`: A 3-dimensional dynamical system
- `x::AbstractVector`: The point at which to evaluate (must have length 3)

# Returns
- `Vector{Float64}`: The curl vector [curl_x, curl_y, curl_z]
"""
function compute_curl_3d(ds::DynamicalSystem, x::AbstractVector)
    ds.dim == 3 || throw(ArgumentError("3D curl requires a 3-dimensional system, got $(ds.dim)"))
    J = compute_jacobian(ds, x)
    return compute_curl_3d(J)
end


"""
    compute_curl_3d(f, x::AbstractVector)

Compute the classical curl vector for a 3D vector field at point x.
"""
function compute_curl_3d(f, x::AbstractVector)
    length(x) == 3 || throw(ArgumentError("3D curl requires a 3-dimensional input, got $(length(x))"))
    J = compute_jacobian(f, x)
    return compute_curl_3d(J)
end


#=============================================================================
    Classification Helpers
=============================================================================#

"""
    is_curl_free(J::AbstractMatrix; rtol::Real=1e-8, atol::Real=1e-10)

Test whether a Jacobian matrix has zero curl (is symmetric) within tolerance.

A curl-free vector field is a necessary condition for a gradient system.

# Arguments
- `J::AbstractMatrix`: The Jacobian matrix to test
- `rtol::Real=1e-8`: Relative tolerance
- `atol::Real=1e-10`: Absolute tolerance

# Returns
- `Bool`: `true` if the curl is zero within tolerance

# Note
This is equivalent to `is_jacobian_symmetric(J; rtol, atol)`.

# Examples
```julia
J_grad = [-2.0 0.5; 0.5 -1.0]
is_curl_free(J_grad)  # true

J_rot = [-1.0 1.0; -1.0 -1.0]
is_curl_free(J_rot)  # false
```
"""
function is_curl_free(J::AbstractMatrix; rtol::Real=1e-8, atol::Real=1e-10)
    A = compute_antisymmetric_part(J)
    return isapprox(A, zeros(size(A)); rtol=rtol, atol=atol)
end


"""
    is_curl_free(ds::DynamicalSystem, x::AbstractVector; rtol::Real=1e-8, atol::Real=1e-10)

Test whether the curl of a dynamical system is zero at point x.
"""
function is_curl_free(ds::DynamicalSystem, x::AbstractVector; rtol::Real=1e-8, atol::Real=1e-10)
    J = compute_jacobian(ds, x)
    return is_curl_free(J; rtol=rtol, atol=atol)
end


"""
    is_approximately_curl_free(J::AbstractMatrix; threshold::Real=0.1)

Test whether a Jacobian has "small" curl relative to its magnitude.

This is useful for identifying gradient-like systems, which may have small
but non-zero curl. The test checks if:
    ‖antisymmetric part‖ / ‖J‖ < threshold

# Arguments
- `J::AbstractMatrix`: The Jacobian matrix to test
- `threshold::Real=0.1`: Maximum relative curl magnitude (default 10%)

# Returns
- `Bool`: `true` if the relative curl is below the threshold

# Examples
```julia
# Nearly gradient (small rotation)
J = [-1.0 0.05; -0.05 -1.0]
is_approximately_curl_free(J; threshold=0.1)  # true

# Significant rotation
J = [-1.0 0.5; -0.5 -1.0]
is_approximately_curl_free(J; threshold=0.1)  # false
```
"""
function is_approximately_curl_free(J::AbstractMatrix; threshold::Real=0.1)
    threshold > 0 || throw(ArgumentError("Threshold must be positive"))
    return relative_curl_magnitude(J) < threshold
end


"""
    is_approximately_curl_free(ds::DynamicalSystem, x::AbstractVector; threshold::Real=0.1)

Test whether a dynamical system has small curl at point x.
"""
function is_approximately_curl_free(ds::DynamicalSystem, x::AbstractVector; threshold::Real=0.1)
    J = compute_jacobian(ds, x)
    return is_approximately_curl_free(J; threshold=threshold)
end


#=============================================================================
    Gradient Component Magnitude
=============================================================================#

"""
    gradient_component_magnitude(J::AbstractMatrix)

Compute the magnitude of the gradient (symmetric) component of a Jacobian.

Returns the Frobenius norm of (J + Jᵀ)/2.

# Arguments
- `J::AbstractMatrix`: A square matrix (typically a Jacobian)

# Returns
- `Float64`: The Frobenius norm of the symmetric part
"""
function gradient_component_magnitude(J::AbstractMatrix)
    S = compute_symmetric_part(J)
    return norm(S)
end


"""
    gradient_component_magnitude(ds::DynamicalSystem, x::AbstractVector)

Compute the gradient component magnitude of a dynamical system at point x.
"""
function gradient_component_magnitude(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return gradient_component_magnitude(J)
end


"""
    curl_to_gradient_ratio(J::AbstractMatrix)

Compute the ratio of curl magnitude to gradient magnitude.

Returns ‖antisymmetric part‖ / ‖symmetric part‖.
- Returns 0 for purely symmetric matrices (gradient systems)
- Returns Inf for purely antisymmetric matrices
- Returns NaN for zero matrices

This ratio indicates the relative importance of rotational vs gradient dynamics.

# Arguments
- `J::AbstractMatrix`: A square matrix (typically a Jacobian)

# Returns
- `Float64`: The ratio of curl to gradient magnitude
"""
function curl_to_gradient_ratio(J::AbstractMatrix)
    curl_mag = curl_magnitude(J)
    grad_mag = gradient_component_magnitude(J)
    
    if grad_mag ≈ 0
        if curl_mag ≈ 0
            return NaN  # Zero matrix
        else
            return Inf  # Purely antisymmetric
        end
    end
    
    return curl_mag / grad_mag
end


"""
    curl_to_gradient_ratio(ds::DynamicalSystem, x::AbstractVector)

Compute the curl-to-gradient ratio of a dynamical system at point x.
"""
function curl_to_gradient_ratio(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return curl_to_gradient_ratio(J)
end
