# FlowClass.jl

A Julia package for classifying continuous-time dynamical systems into a hierarchy of structural classes: Gradient, Gradient-like, Morse-Smale, Structurally Stable, and General.

## Motivation

Many biological and physical systems are modelled as dynamical systems of the form:

$$\frac{d\mathbf{x}}{dt} = \mathbf{F}(\mathbf{x})$$

Understanding which structural class a system belongs to has profound implications for its qualitative behaviour. For instance:

- **Gradient systems** cannot exhibit oscillations or chaos — trajectories always descend a potential
- **Morse-Smale systems** can have limit cycles but remain structurally stable
- **General systems** may exhibit chaotic dynamics or complex attractors

This package provides computational tools to test and classify dynamical systems based on their structural properties, with applications to Waddington's epigenetic landscape and cell fate decision models.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Theosysbio/FlowClass.jl")
```

Or for local development:

```julia
cd("path/to/FlowClass.jl")
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
using FlowClass

# Define a dynamical system: dx/dt = F(x)
# Example: A gradient system with potential V(x) = x₁² + x₁x₂ + x₂²
F = x -> [-2x[1] - x[2], -x[1] - 2x[2]]
ds = DynamicalSystem(F, 2)

# Classify the system
bounds = ((-2.0, 2.0), (-2.0, 2.0))
result = classify_system(ds, bounds)
print_classification(result)
```

## The Classification Hierarchy

From most restrictive to most general:

```
Gradient Systems
└── Gradient-like Systems
    └── Morse-Smale Systems
        └── Structurally Stable Systems
            └── General Dynamical Systems
```

### Key Properties by Class

| Class | Jacobian | Curl | Periodic Orbits | Fixed Points | Lyapunov Function |
|-------|----------|------|-----------------|--------------|-------------------|
| Gradient | Symmetric | Zero | None | All hyperbolic | Global |
| Gradient-like | Nearly symmetric | Near zero | None | All hyperbolic | Global |
| Morse-Smale | No requirement | Any | Hyperbolic only | All hyperbolic | Local only |
| Structurally Stable | No requirement | Any | Hyperbolic only | All hyperbolic | Local |
| General | No requirement | Any | Any | Non-hyperbolic possible | None guaranteed |

### SystemClass Enum

```julia
@enum SystemClass begin
    GRADIENT           # Pure gradient system: F = -∇V
    GRADIENT_LIKE      # Has global Lyapunov function, no periodic orbits
    MORSE_SMALE        # Hyperbolic fixed points and orbits, transverse manifolds
    STRUCTURALLY_STABLE # Robust to small perturbations
    GENERAL            # No special structure guaranteed
end
```

## API Reference

### Types

#### `DynamicalSystem{F}`

Represents a continuous-time dynamical system dx/dt = f(x).

```julia
# From function and dimension
ds = DynamicalSystem(x -> -x, 2)

# From function and sample point (infers dimension)
ds = DynamicalSystem(x -> -x, [1.0, 2.0])

# Evaluate the vector field
ds([1.0, 2.0])  # returns [-1.0, -2.0]

# Get dimension
dimension(ds)  # returns 2
```

#### `FixedPoint`

Represents a fixed point with stability information.

```julia
struct FixedPoint
    location::Vector{Float64}    # Position in state space
    eigenvalues::Vector{ComplexF64}  # Eigenvalues of Jacobian
    type::FixedPointType         # Classification
end
```

#### `FixedPointType` Enum

```julia
@enum FixedPointType begin
    STABLE_NODE        # All eigenvalues have negative real parts (no imaginary)
    UNSTABLE_NODE      # All eigenvalues have positive real parts (no imaginary)
    SADDLE             # Mixed signs of real parts
    STABLE_FOCUS       # Negative real parts with imaginary components
    UNSTABLE_FOCUS     # Positive real parts with imaginary components
    CENTER             # Pure imaginary eigenvalues
    NON_HYPERBOLIC     # At least one eigenvalue with zero real part
end
```

#### `PeriodicOrbit`

Represents a detected periodic orbit.

```julia
struct PeriodicOrbit
    points::Vector{Vector{Float64}}  # Sample points along orbit
    period::Float64                   # Estimated period
    is_stable::Bool                   # Stability (via Floquet analysis)
end
```

#### `ClassificationResult`

Complete result from system classification.

```julia
struct ClassificationResult
    system_class::SystemClass
    fixed_points::Vector{FixedPoint}
    periodic_orbits::Vector{PeriodicOrbit}
    jacobian_symmetry::Float64       # Mean relative symmetry error
    curl_gradient_ratio::Float64     # ‖curl‖ / ‖gradient‖
    has_transverse_manifolds::Union{Bool, Nothing}
    confidence::Float64              # Classification confidence
    details::Dict{String, Any}       # Additional analysis data
end
```

### Jacobian Analysis

#### `compute_jacobian(ds, x)` / `compute_jacobian(f, x)`

Compute the Jacobian matrix J[i,j] = ∂fᵢ/∂xⱼ at point x using automatic differentiation.

```julia
ds = DynamicalSystem(x -> [x[1]^2, x[1]*x[2]], 2)
J = compute_jacobian(ds, [2.0, 3.0])
# J = [4.0 0.0; 3.0 2.0]
```

#### `is_jacobian_symmetric(J; rtol=1e-8, atol=1e-10)`

Test whether a Jacobian matrix is symmetric within tolerance.

```julia
J_sym = [-2.0 0.5; 0.5 -1.0]
is_jacobian_symmetric(J_sym)  # true

J_nonsym = [-1.0 0.5; -0.5 -1.0]
is_jacobian_symmetric(J_nonsym)  # false
```

#### `jacobian_symmetry_error(J)` / `jacobian_symmetry_error(ds, x)`

Compute the Frobenius norm of the antisymmetric part: ‖(J − Jᵀ)/2‖.

#### `relative_jacobian_symmetry_error(J)` / `relative_jacobian_symmetry_error(ds, x)`

Scale-independent symmetry error: ‖(J − Jᵀ)/2‖ / ‖J‖.

### Curl Analysis

For a vector field **F**, the curl measures the rotational component of the dynamics. In the Helmholtz decomposition **F** = −∇U + **F**_curl, the curl component **F**_curl is orthogonal to the gradient and cannot be captured by any potential landscape.

#### `curl_magnitude(ds, x)` / `curl_magnitude(f, x, n)`

Compute the magnitude of the curl at point x. For 2D systems, returns the scalar curl. For 3D, returns ‖∇ × **F**‖. For higher dimensions, returns ‖(J − Jᵀ)/2‖_F (Frobenius norm of antisymmetric part).

```julia
# Rotation system has high curl
rotation = DynamicalSystem(x -> [-x[2], x[1]], 2)
curl_magnitude(rotation, [1.0, 0.0])  # ≈ 2.0

# Gradient system has zero curl
gradient_sys = DynamicalSystem(x -> [-2x[1], -2x[2]], 2)
curl_magnitude(gradient_sys, [1.0, 1.0])  # ≈ 0.0
```

#### `is_curl_free(ds, x; atol=1e-10)` / `is_curl_free(ds, bounds; n_samples=100, atol=1e-10)`

Test if the curl is zero at a point or throughout a region.

#### `curl_to_gradient_ratio(ds, x)`

Compute the ratio ‖curl‖ / ‖F‖, indicating the relative strength of rotational dynamics.

### Fixed Point Analysis

#### `find_fixed_points(ds, bounds; n_starts=100, tol=1e-8)`

Find fixed points of the system within the specified bounds using multi-start optimisation.

```julia
# Toggle switch with two stable states
toggle = DynamicalSystem(x -> [
    1/(1 + x[2]^2) - x[1],
    1/(1 + x[1]^2) - x[2]
], 2)
bounds = ((0.0, 2.0), (0.0, 2.0))

fps = find_fixed_points(toggle, bounds)
for fp in fps
    println("Fixed point at $(fp.location): $(fp.type)")
end
```

#### `classify_fixed_point(ds, x)` / `classify_fixed_point(eigenvalues)`

Determine the type of a fixed point from its Jacobian eigenvalues.

#### `is_hyperbolic(fp::FixedPoint)` / `is_hyperbolic(eigenvalues)`

Check if a fixed point is hyperbolic (no eigenvalues with zero real part).

### Periodic Orbit Detection

#### `find_periodic_orbits(ds, bounds; n_trajectories=50, max_period=100.0)`

Search for periodic orbits by integrating trajectories and detecting recurrence.

```julia
# Van der Pol oscillator (has a limit cycle)
vdp = DynamicalSystem(x -> [x[2], (1 - x[1]^2)*x[2] - x[1]], 2)
bounds = ((-3.0, 3.0), (-3.0, 3.0))

orbits = find_periodic_orbits(vdp, bounds)
if !isempty(orbits)
    println("Found orbit with period ≈ $(orbits[1].period)")
end
```

#### `has_periodic_orbits(ds, bounds; kwargs...)`

Quick check for the existence of periodic orbits. Returns `true` if any are found.

### Manifold Analysis

#### `compute_stable_manifold(ds, fp; n_points=100, extent=1.0)`

Compute points along the stable manifold of a saddle point.

#### `compute_unstable_manifold(ds, fp; n_points=100, extent=1.0)`

Compute points along the unstable manifold of a saddle point.

#### `detect_homoclinic_orbit(ds, saddle; tol=0.1)`

Check for homoclinic connections (orbits connecting a saddle to itself).

#### `check_transversality(ds, fps; tol=0.01)`

Verify that stable and unstable manifolds intersect transversally (required for Morse-Smale).

### Classification Functions

#### `classify_system(ds, bounds; kwargs...)`

Perform full classification with detailed analysis.

```julia
ds = DynamicalSystem(x -> [-2x[1], -3x[2]], 2)
bounds = ((-2.0, 2.0), (-2.0, 2.0))

result = classify_system(ds, bounds)
print_classification(result)
```

**Keyword arguments:**
- `n_samples::Int=100` — Points sampled for Jacobian/curl analysis
- `n_starts::Int=100` — Starting points for fixed point search
- `check_manifolds::Bool=true` — Whether to analyse manifold transversality
- `orbit_timeout::Float64=10.0` — Max time for periodic orbit search

#### `quick_classify(ds, bounds)`

Fast classification with fewer samples and no manifold analysis.

#### `get_system_class(ds, bounds)`

Return only the `SystemClass` enum value.

### Classification Result Queries

```julia
result = classify_system(ds, bounds)

is_gradient(result)              # true if GRADIENT
is_gradient_like(result)         # true if GRADIENT or GRADIENT_LIKE
is_morse_smale(result)           # true if Morse-Smale or more restrictive
allows_periodic_orbits(result)   # false for gradient-like systems

# Get landscape interpretation
can_represent, landscape_type, description = has_landscape_representation(result)
```

### Utility Functions

#### `print_classification(result; io=stdout)`

Print a formatted classification report.

```julia
result = classify_system(ds, bounds)
print_classification(result)
```

Output:
```
╔══════════════════════════════════════════════════════════════╗
║                  System Classification Report                 ║
╠══════════════════════════════════════════════════════════════╣
║ System Class: GRADIENT                                       ║
║ Confidence: 0.95                                             ║
╠══════════════════════════════════════════════════════════════╣
║ Fixed Points: 1                                              ║
║   • Stable node at [0.0, 0.0]                               ║
║ Periodic Orbits: 0                                           ║
╠══════════════════════════════════════════════════════════════╣
║ Jacobian Symmetry Error: 1.2e-15                            ║
║ Curl/Gradient Ratio: 0.0                                     ║
║ Manifolds Transverse: N/A (no saddles)                      ║
╠══════════════════════════════════════════════════════════════╣
║ Landscape: Global potential V(x) exists where F = -∇V        ║
╚══════════════════════════════════════════════════════════════╝
```

## Examples

### Example 1: Testing a Gradient System

A gradient system satisfies dx/dt = −∇V(x) for some scalar potential V. Its Jacobian is the negative Hessian of V, which is always symmetric.

```julia
using FlowClass

# Potential: V(x) = x₁² + x₂² (paraboloid)
# Gradient: ∇V = [2x₁, 2x₂]
# System: dx/dt = -∇V = [-2x₁, -2x₂]

ds = DynamicalSystem(x -> -2 .* x, 2)
bounds = ((-2.0, 2.0), (-2.0, 2.0))

result = classify_system(ds, bounds)
println("Class: ", result.system_class)  # GRADIENT
println("Symmetry error: ", result.jacobian_symmetry)  # ≈ 0
println("Curl ratio: ", result.curl_gradient_ratio)  # ≈ 0
```

### Example 2: System with Rotation (Non-Gradient)

Systems with rotational dynamics have antisymmetric components in their Jacobian and non-zero curl.

```julia
using FlowClass

# Damped oscillator with rotation
# dx₁/dt = -x₁ + ωx₂
# dx₂/dt = -ωx₁ - x₂
ω = 1.0
ds = DynamicalSystem(x -> [-x[1] + ω*x[2], -ω*x[1] - x[2]], 2)

J = compute_jacobian(ds, [0.0, 0.0])
# J = [-1  1; -1  -1]

is_jacobian_symmetric(J)  # false
relative_jacobian_symmetry_error(J)  # ≈ 0.5
curl_magnitude(ds, [0.0, 0.0])  # ≈ 2.0
```

### Example 3: Lorenz System

The Lorenz system is a classic example of a chaotic, non-gradient system.

```julia
using FlowClass

function lorenz(x; σ=10.0, ρ=28.0, β=8/3)
    return [σ * (x[2] - x[1]),
            x[1] * (ρ - x[3]) - x[2],
            x[1] * x[2] - β * x[3]]
end

ds = DynamicalSystem(lorenz, 3)
bounds = ((-20.0, 20.0), (-30.0, 30.0), (0.0, 50.0))

result = classify_system(ds, bounds)
println("Class: ", result.system_class)  # GENERAL
println("Fixed points found: ", length(result.fixed_points))
```

### Example 4: Stem Cell Differentiation Model

This example implements the stem cell differentiation model from Brackston, Lakatos & Stumpf (2018), which describes the dynamics of pluripotency factors (Nanog, Oct4-Sox2, Fgf4) and differentiation marker (Gata6) under the influence of LIF signalling.

The model demonstrates non-gradient dynamics with curl components, multiple stable states (pluripotent and differentiated), and transition states — key features of Waddington's epigenetic landscape.

#### The Model Equations (Eqns. 8–16)

The developmental model consists of four molecular species: Nanog ($N$), Oct4-Sox2 complex ($O$), Fgf4 ($F$), and Gata6 ($G$), with LIF ($L$) as an external control parameter. Under a quasi-equilibrium assumption, the dynamics are governed by eight reactions: four production propensities and four degradation propensities.

**Production propensities:**

$$a_1 = \frac{k_0 O (k_1 + k_2 N^2 + k_0 O + k_3 L)}{1 + k_0 O (k_2 N^2 + k_0 O + k_3 L + k_4 F^2) + k_5 O G^2} \tag{8}$$

$$a_2 = \frac{k_6 + k_7 O}{1 + k_7 O + k_8 G^2} \tag{9}$$

$$a_3 = \frac{k_9 + k_{10} O}{1 + k_{10} O} \tag{10}$$

$$a_4 = \frac{k_{11} + k_{12} G^2 + k_{14} O}{1 + k_{12} G^2 + k_{13} N^2 + k_{14} O} \tag{11}$$

**Degradation propensities** (first-order with rate $k_d$):

$$a_5 = k_d N \tag{12}$$

$$a_6 = k_d O \tag{13}$$

$$a_7 = k_d F \tag{14}$$

$$a_8 = k_d G \tag{15}$$

**Stoichiometry matrix:**

The system evolution is described by dx/dt = S · a(x), where the stoichiometry matrix is:

$$S = \begin{bmatrix} 1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & -1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & -1 \end{bmatrix} \tag{16}$$

This yields the ODEs: $\dot{N} = a_1 - k_d N$, $\dot{O} = a_2 - k_d O$, $\dot{F} = a_3 - k_d F$, $\dot{G} = a_4 - k_d G$.

```julia
using FlowClass

# Parameters from Brackston et al. (2018) Table in Methods section
const k = (
    k0 = 0.005, k1 = 0.01, k2 = 0.4, k3 = 1.0, k4 = 0.1,
    k5 = 0.00135, k6 = 0.01, k7 = 0.01, k8 = 1.0, k9 = 1.0,
    k10 = 0.01, k11 = 5.0, k12 = 1.0, k13 = 0.005, k14 = 1.0,
    kd = 1.0
)

"""
Stem cell differentiation model (Brackston et al. 2018, Eqns. 8–16).
State vector: x = [N, O, F, G] where
  N = Nanog, O = Oct4-Sox2, F = Fgf4, G = Gata6
Parameter L controls LIF concentration (external signal).
"""
function stem_cell_model(x; L=50.0, p=k)
    N, O, F, G = x
    
    # Production propensities (Eqns. 8–11)
    a1 = p.k0 * O * (p.k1 + p.k2*N^2 + p.k0*O + p.k3*L) / 
         (1 + p.k0*O*(p.k2*N^2 + p.k0*O + p.k3*L + p.k4*F^2) + p.k5*O*G^2)
    
    a2 = (p.k6 + p.k7*O) / (1 + p.k7*O + p.k8*G^2)
    
    a3 = (p.k9 + p.k10*O) / (1 + p.k10*O)
    
    a4 = (p.k11 + p.k12*G^2 + p.k14*O) / (1 + p.k12*G^2 + p.k13*N^2 + p.k14*O)
    
    # Net rates: production − degradation (from stoichiometry, Eq. 16)
    dN = a1 - p.kd * N
    dO = a2 - p.kd * O
    dF = a3 - p.kd * F
    dG = a4 - p.kd * G
    
    return [dN, dO, dF, dG]
end

# Create system with high LIF (favours pluripotency)
ds_high_LIF = DynamicalSystem(x -> stem_cell_model(x; L=150.0), 4)

# Create system with low LIF (favours differentiation)
ds_low_LIF = DynamicalSystem(x -> stem_cell_model(x; L=10.0), 4)

# Define bounds for the four-dimensional state space
# N ∈ [0, 100], O ∈ [0, 100], F ∈ [0, 100], G ∈ [0, 120]
bounds = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 120.0))

# Classify under high LIF conditions
println("=== High LIF (L=150) — Pluripotent conditions ===")
result_high = classify_system(ds_high_LIF, bounds; n_samples=200)
print_classification(result_high)

# Classify under low LIF conditions  
println("\n=== Low LIF (L=10) — Differentiation conditions ===")
result_low = classify_system(ds_low_LIF, bounds; n_samples=200)
print_classification(result_low)

# Analyse fixed points (cell states)
println("\n=== Fixed Point Analysis ===")
for (i, fp) in enumerate(result_high.fixed_points)
    N, O, F, G = fp.location
    if N > 50 && G < 20
        state = "Pluripotent (stem cell)"
    elseif G > 50 && N < 20
        state = "Differentiated"
    else
        state = "Transition state"
    end
    println("State $i: $state")
    println("  Location: N=$(round(N, digits=1)), O=$(round(O, digits=1)), " *
            "F=$(round(F, digits=1)), G=$(round(G, digits=1))")
    println("  Type: $(fp.type)")
end

# Check for non-gradient (curl) dynamics
# The paper notes that curl dynamics are ubiquitous in gene regulatory networks
println("\n=== Curl Analysis ===")
test_point = [60.0, 50.0, 40.0, 20.0]  # Near pluripotent state
curl = curl_magnitude(ds_high_LIF, test_point)
ratio = curl_to_gradient_ratio(ds_high_LIF, test_point)
println("Curl magnitude at test point: $(round(curl, digits=4))")
println("Curl/gradient ratio: $(round(ratio, digits=4))")

if ratio > 0.1
    println("→ Significant non-gradient dynamics present")
    println("  Forward and reverse differentiation paths will differ (see paper Fig 6)")
end
```

**Expected output:**

The stem cell model exhibits:
1. **Multiple stable states**: Pluripotent (high Nanog, low Gata6) and differentiated (low Nanog, high Gata6)
2. **Non-zero curl**: The system is not a pure gradient system, meaning minimum action paths differ for differentiation vs reprogramming
3. **Transition state**: An unstable fixed point between the two stable states
4. **LIF-dependent landscape**: Changing LIF concentration reshapes the potential landscape

This connects to the paper's key insight: the presence of curl dynamics means that observing differentiation trajectories does not directly reveal reprogramming paths.

### Example 5: Analysing Landscape Structure

```julia
using FlowClass

# Simple bistable system (toggle switch)
function toggle_switch(x; a=1.0, n=2)
    return [
        a / (1 + x[2]^n) - x[1],
        a / (1 + x[1]^n) - x[2]
    ]
end

ds = DynamicalSystem(toggle_switch, 2)
bounds = ((0.0, 2.0), (0.0, 2.0))

result = classify_system(ds, bounds)

# Examine fixed points
for fp in result.fixed_points
    println("Fixed point at $(round.(fp.location, digits=3))")
    println("  Type: $(fp.type)")
    println("  Eigenvalues: $(round.(fp.eigenvalues, digits=3))")
    
    if fp.type == SADDLE
        println("  → This is a transition state between cell fates")
    end
end

# Check landscape representation
can_rep, type, desc = has_landscape_representation(result)
println("\nLandscape: $desc")
```

## Theoretical Background

### Gradient Systems

A gradient system satisfies dx/dt = −∇V(x) for some scalar potential V(x). Key properties:

- **Jacobian symmetry**: J = −H(V) where H is the Hessian, so J = Jᵀ
- **Curl-free**: ∇ × **F** = 0 (in 3D) or more generally, the Jacobian is symmetric
- **No periodic orbits**: Trajectories always descend the potential
- **Path independence**: Line integrals are path-independent

The condition ∂fᵢ/∂xⱼ = ∂fⱼ/∂xᵢ is both necessary and sufficient for the existence of a potential.

### Gradient-like Systems

Gradient-like systems possess a global Lyapunov function but may have non-symmetric Jacobians away from fixed points. They share the key property that trajectories cannot form closed loops.

### Morse-Smale Systems

Morse-Smale systems allow hyperbolic periodic orbits (limit cycles) while maintaining structural stability. They require:

1. Finitely many hyperbolic fixed points
2. Finitely many hyperbolic periodic orbits
3. Transverse intersection of stable/unstable manifolds
4. No non-wandering points other than fixed points and periodic orbits

### Non-Gradient Dynamics and Curl

As discussed by Brackston et al. (2018), most biological systems exhibit non-gradient dynamics. The vector field can be decomposed as:

$$\mathbf{F}(\mathbf{x}) = -\nabla U(\mathbf{x}) + \mathbf{F}_U(\mathbf{x})$$

where U is the potential (related to the probability landscape) and **F**_U is the curl/flux component. The curl component:

- Is indicative of non-equilibrium dynamics
- Causes forward and reverse transition paths to differ
- Cannot be inferred from static snapshot data alone
- Arises naturally in gene regulatory networks due to asymmetric interactions

### Connection to Waddington's Landscape

The classification hierarchy relates directly to interpretations of Waddington's epigenetic landscape:

| System Class | Landscape Interpretation |
|--------------|-------------------------|
| Gradient | True potential landscape; elevation = −log(probability) |
| Gradient-like | Quasi-potential exists; landscape approximation valid |
| Morse-Smale | Local potentials around attractors; limit cycles as valleys |
| General | Landscape metaphor breaks down; curl dynamics dominate |

## Dependencies

- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) — Automatic differentiation for Jacobians
- [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) — Nonlinear equation solving for fixed points
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) — ODE integration for trajectories and manifolds
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) — Standard library

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## References

- Brackston, R. D., Lakatos, E., & Stumpf, M. P. H. (2018). Transition state characteristics during cell differentiation. *PLoS Computational Biology*, 14(9), e1006405.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*. Westview Press.
- Palis, J., & de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.
- Smale, S. (1967). Differentiable dynamical systems. *Bulletin of the AMS*, 73(6), 747-817.
- Wang, J. (2015). Landscape and flux theory of non-equilibrium dynamical systems with application to biology. *Advances in Physics*, 64(1), 1-137.

## Licence

MIT License — see [LICENSE](LICENSE) for details.
