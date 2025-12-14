"""
    jacobian.jl

Functions for computing and analysing the Jacobian matrix of dynamical systems.
"""

using ForwardDiff
using LinearAlgebra

"""
    compute_jacobian(ds::DynamicalSystem, x::AbstractVector)

Compute the Jacobian matrix of the vector field at point x.

The Jacobian J is defined as J[i,j] = ∂fᵢ/∂xⱼ, where f is the vector field.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `x::AbstractVector`: The point at which to evaluate the Jacobian

# Returns
- `Matrix{Float64}`: The n×n Jacobian matrix

# Examples
```julia
# Linear system dx/dt = Ax has constant Jacobian equal to A
A = [-1.0 0.5; -0.5 -1.0]
ds = DynamicalSystem(x -> A * x, 2)
J = compute_jacobian(ds, [0.0, 0.0])
# J ≈ A
```
"""
function compute_jacobian(ds::DynamicalSystem, x::AbstractVector)
    length(x) == ds.dim || throw(DimensionMismatch(
        "Input vector has length $(length(x)), expected $(ds.dim)"
    ))
    
    # ForwardDiff.jacobian expects a function that takes a vector
    # and returns a vector. It computes ∂f/∂x.
    return ForwardDiff.jacobian(ds.f, x)
end

"""
    compute_jacobian(f, x::AbstractVector)

Compute the Jacobian matrix of a vector field function at point x.

This is a convenience method that works directly with functions without 
wrapping them in a DynamicalSystem.

# Arguments
- `f`: A callable that takes a vector and returns a vector
- `x::AbstractVector`: The point at which to evaluate the Jacobian

# Returns
- `Matrix`: The Jacobian matrix
"""
function compute_jacobian(f, x::AbstractVector)
    return ForwardDiff.jacobian(f, x)
end


"""
    is_jacobian_symmetric(J::AbstractMatrix; rtol::Real=1e-8, atol::Real=1e-10)

Test whether a Jacobian matrix is symmetric within the specified tolerances.

A symmetric Jacobian (J = Jᵀ) is a necessary condition for a gradient system,
since for dx/dt = -∇V(x), the Jacobian is the negative Hessian of V, which
is symmetric by equality of mixed partial derivatives.

# Arguments
- `J::AbstractMatrix`: The Jacobian matrix to test
- `rtol::Real=1e-8`: Relative tolerance for comparison
- `atol::Real=1e-10`: Absolute tolerance for comparison

# Returns
- `Bool`: `true` if the matrix is symmetric within tolerance

# Details
Uses `isapprox` to compare J with its transpose, which checks whether
`norm(J - Jᵀ) ≤ max(atol, rtol * max(norm(J), norm(Jᵀ)))`.

# Examples
```julia
# Symmetric matrix (from a gradient system)
J_grad = [-2.0 0.5; 0.5 -1.0]
is_jacobian_symmetric(J_grad)  # true

# Non-symmetric matrix (from a non-gradient system)
J_nongrad = [-1.0 0.5; -0.5 -1.0]
is_jacobian_symmetric(J_nongrad)  # false
```
"""
function is_jacobian_symmetric(J::AbstractMatrix; rtol::Real=1e-8, atol::Real=1e-10)
    size(J, 1) == size(J, 2) || throw(ArgumentError("Matrix must be square"))
    return isapprox(J, transpose(J); rtol=rtol, atol=atol)
end

"""
    is_jacobian_symmetric(ds::DynamicalSystem, x::AbstractVector; 
                          rtol::Real=1e-8, atol::Real=1e-10)

Test whether the Jacobian of a dynamical system is symmetric at point x.

This is a convenience method that computes the Jacobian and tests its symmetry.

# Arguments
- `ds::DynamicalSystem`: The dynamical system
- `x::AbstractVector`: The point at which to evaluate and test the Jacobian
- `rtol::Real=1e-8`: Relative tolerance for comparison
- `atol::Real=1e-10`: Absolute tolerance for comparison

# Returns
- `Bool`: `true` if the Jacobian is symmetric within tolerance at x

# Examples
```julia
# A gradient system: dx/dt = -∇V where V(x) = x₁² + x₂²
ds_grad = DynamicalSystem(x -> [-2x[1], -2x[2]], 2)
is_jacobian_symmetric(ds_grad, [1.0, 1.0])  # true

# A non-gradient system with rotation
ds_rot = DynamicalSystem(x -> [-x[1] + x[2], -x[1] - x[2]], 2)
is_jacobian_symmetric(ds_rot, [1.0, 1.0])  # false
```
"""
function is_jacobian_symmetric(ds::DynamicalSystem, x::AbstractVector; 
                                rtol::Real=1e-8, atol::Real=1e-10)
    J = compute_jacobian(ds, x)
    return is_jacobian_symmetric(J; rtol=rtol, atol=atol)
end


"""
    jacobian_symmetry_error(J::AbstractMatrix)

Compute the Frobenius norm of the antisymmetric part of a matrix.

This quantifies how far a Jacobian is from being symmetric. For a gradient
system, this should be zero (up to numerical precision).

# Arguments
- `J::AbstractMatrix`: The Jacobian matrix

# Returns
- `Float64`: The Frobenius norm of (J - Jᵀ)/2

# Examples
```julia
J_grad = [-2.0 0.5; 0.5 -1.0]
jacobian_symmetry_error(J_grad)  # ≈ 0.0

J_nongrad = [-1.0 0.5; -0.5 -1.0]
jacobian_symmetry_error(J_nongrad)  # > 0
```
"""
function jacobian_symmetry_error(J::AbstractMatrix)
    size(J, 1) == size(J, 2) || throw(ArgumentError("Matrix must be square"))
    antisymmetric_part = (J - transpose(J)) / 2
    return norm(antisymmetric_part)
end

"""
    jacobian_symmetry_error(ds::DynamicalSystem, x::AbstractVector)

Compute the symmetry error of the Jacobian at point x.
"""
function jacobian_symmetry_error(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return jacobian_symmetry_error(J)
end


"""
    relative_jacobian_symmetry_error(J::AbstractMatrix)

Compute the symmetry error relative to the Jacobian's magnitude.

This returns `‖(J - Jᵀ)/2‖ / ‖J‖`, providing a scale-independent measure
of how non-symmetric the Jacobian is.

# Arguments
- `J::AbstractMatrix`: The Jacobian matrix

# Returns
- `Float64`: The relative symmetry error (between 0 and 1 for most cases)

# Examples
```julia
J = [-1.0 0.5; -0.5 -1.0]
relative_jacobian_symmetry_error(J)  # Measures relative asymmetry
```
"""
function relative_jacobian_symmetry_error(J::AbstractMatrix)
    size(J, 1) == size(J, 2) || throw(ArgumentError("Matrix must be square"))
    J_norm = norm(J)
    
    # Handle the case of zero Jacobian
    if J_norm ≈ 0
        return 0.0
    end
    
    antisymmetric_part = (J - transpose(J)) / 2
    return norm(antisymmetric_part) / J_norm
end

"""
    relative_jacobian_symmetry_error(ds::DynamicalSystem, x::AbstractVector)

Compute the relative symmetry error of the Jacobian at point x.
"""
function relative_jacobian_symmetry_error(ds::DynamicalSystem, x::AbstractVector)
    J = compute_jacobian(ds, x)
    return relative_jacobian_symmetry_error(J)
end
