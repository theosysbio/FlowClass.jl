"""
    types.jl

Core type definitions for FlowClass.jl
"""

"""
    DynamicalSystem{F}

Represents a continuous-time dynamical system dx/dt = f(x) where x ∈ ℝⁿ.

# Fields
- `f::F`: The vector field, a callable that takes a vector x and returns dx/dt
- `dim::Int`: The dimension n of the state space

# Constructors

    DynamicalSystem(f, dim::Int)

Create a dynamical system from a function `f(x)` that returns a vector of length `dim`.

    DynamicalSystem(f, x₀::AbstractVector)

Create a dynamical system from a function `f(x)`, inferring the dimension from a 
sample point `x₀`. This also validates that `f(x₀)` returns a vector of the correct length.

# Examples

```julia
# A simple 2D linear system: dx/dt = Ax
A = [-1.0 0.5; -0.5 -1.0]
ds = DynamicalSystem(x -> A * x, 2)

# A 3D nonlinear system (Lorenz)
function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
    return [σ * (x[2] - x[1]),
            x[1] * (ρ - x[3]) - x[2],
            x[1] * x[2] - β * x[3]]
end
ds = DynamicalSystem(lorenz, 3)

# Inferring dimension from a sample point
ds = DynamicalSystem(lorenz, [1.0, 1.0, 1.0])
```
"""
struct DynamicalSystem{F}
    f::F
    dim::Int
    
    function DynamicalSystem{F}(f::F, dim::Int) where {F}
        dim > 0 || throw(ArgumentError("Dimension must be positive, got $dim"))
        new{F}(f, dim)
    end
end

# Outer constructors
function DynamicalSystem(f::F, dim::Int) where {F}
    DynamicalSystem{F}(f, dim)
end

function DynamicalSystem(f::F, x₀::AbstractVector) where {F}
    dim = length(x₀)
    
    # Validate that f returns a vector of the correct length
    result = f(x₀)
    if length(result) != dim
        throw(DimensionMismatch(
            "Function f returned a vector of length $(length(result)), " *
            "expected length $dim based on the sample point"
        ))
    end
    
    DynamicalSystem{F}(f, dim)
end

"""
    (ds::DynamicalSystem)(x)

Evaluate the vector field at point x.

# Examples
```julia
ds = DynamicalSystem(x -> -x, 2)
ds([1.0, 2.0])  # Returns [-1.0, -2.0]
```
"""
function (ds::DynamicalSystem)(x::AbstractVector)
    length(x) == ds.dim || throw(DimensionMismatch(
        "Input vector has length $(length(x)), expected $(ds.dim)"
    ))
    return ds.f(x)
end

# Convenience method for getting dimension
"""
    dimension(ds::DynamicalSystem)

Return the dimension of the state space.
"""
dimension(ds::DynamicalSystem) = ds.dim

# Pretty printing
function Base.show(io::IO, ds::DynamicalSystem{F}) where {F}
    print(io, "DynamicalSystem{$F}(dim=$(ds.dim))")
end

function Base.show(io::IO, ::MIME"text/plain", ds::DynamicalSystem{F}) where {F}
    println(io, "DynamicalSystem")
    println(io, "  Dimension: $(ds.dim)")
    print(io, "  Vector field type: $F")
end
