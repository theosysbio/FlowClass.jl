"""
    FlowClass.jl

A Julia package for classifying dynamical systems into hierarchical categories:
Gradient, Gradient-like, Morse-Smale, Generic, Structurally Stable, and General.

The classification is based on properties such as:
- Jacobian symmetry (gradient systems have symmetric Jacobians)
- Curl of the vector field (zero for gradient systems)
- Existence and type of periodic orbits
- Hyperbolicity of fixed points
- Transversality of stable/unstable manifolds

# Quick Start

```julia
using FlowClass

# Define a dynamical system
ds = DynamicalSystem(x -> [-2x[1], -2x[2]], 2)

# Compute the Jacobian at a point
J = compute_jacobian(ds, [1.0, 1.0])

# Test if the Jacobian is symmetric (necessary for gradient systems)
is_jacobian_symmetric(J)
```

# Dynamical System Classes (from most restrictive to most general)

1. **Gradient Systems**: dx/dt = -∇V(x) for some potential V
   - Symmetric Jacobian, zero curl, no periodic orbits

2. **Gradient-like Systems**: Have a global Lyapunov function
   - Nearly symmetric Jacobian, small curl, no periodic orbits

3. **Morse-Smale Systems**: Hyperbolic fixed points and periodic orbits
   - Transverse intersection of stable/unstable manifolds

4. **Generic Systems**: Structurally stable except at bifurcation points
   - May have non-hyperbolic points at bifurcations

5. **Structurally Stable Systems**: Qualitative behaviour unchanged under perturbation
   - All fixed points hyperbolic, transverse manifold intersections

6. **General Dynamical Systems**: No restrictions
   - May exhibit chaos, strange attractors, etc.
"""
module FlowClass

using LinearAlgebra
using ForwardDiff
using NLsolve
using OrdinaryDiffEq
using Random
using Statistics

# Include source files
include("types.jl")
include("jacobian.jl")
include("curl.jl")
include("fixed_points.jl")
include("periodic_orbits.jl")
include("manifolds.jl")
include("classification.jl")

# Export types
export DynamicalSystem
export dimension
export FixedPoint
export FixedPointType
export STABLE_NODE, UNSTABLE_NODE, SADDLE
export STABLE_FOCUS, UNSTABLE_FOCUS, SADDLE_FOCUS
export CENTER, NON_HYPERBOLIC, UNKNOWN

# Export Jacobian functions
export compute_jacobian
export is_jacobian_symmetric
export jacobian_symmetry_error
export relative_jacobian_symmetry_error

# Export curl functions
export compute_antisymmetric_part
export compute_symmetric_part
export decompose_jacobian
export curl_magnitude
export relative_curl_magnitude
export compute_curl_3d
export is_curl_free
export is_approximately_curl_free
export gradient_component_magnitude
export curl_to_gradient_ratio

# Export fixed point functions
export compute_eigenvalues
export is_hyperbolic
export is_stable
export classify_fixed_point_type
export classify_fixed_point
export latin_hypercube_sample
export cluster_points
export unique_fixed_points
export find_fixed_points_nlsolve
export find_fixed_points_ode
export find_unstable_fixed_points_ode
export find_fixed_points
export all_fixed_points_hyperbolic
export count_fixed_point_types
export stable_fixed_points
export unstable_fixed_points

# Export periodic orbit types
export PeriodicOrbit
export PeriodicOrbitType
export STABLE_LIMIT_CYCLE, UNSTABLE_LIMIT_CYCLE, SADDLE_CYCLE
export NON_HYPERBOLIC_CYCLE, UNKNOWN_CYCLE
export PoincaréSection

# Export periodic orbit functions
export classify_floquet_multipliers
export compute_monodromy_matrix
export compute_floquet_multipliers
export detect_oscillation
export find_periodic_orbit_recurrence
export refine_periodic_orbit
export find_poincare_crossings
export find_periodic_orbit_poincare
export classify_periodic_orbit
export find_periodic_orbits
export unique_periodic_orbits
export has_periodic_orbits
export all_periodic_orbits_hyperbolic
export stable_periodic_orbits
export unstable_periodic_orbits
export count_periodic_orbit_types

# Export manifold types
export ManifoldType
export STABLE_MANIFOLD, UNSTABLE_MANIFOLD
export Manifold, ManifoldIntersection

# Export manifold functions
export compute_stable_eigenspace
export compute_unstable_eigenspace
export extract_real_basis
export compute_local_manifold
export grow_manifold_1d
export compute_stable_manifold
export compute_unstable_manifold
export compute_manifolds
export find_manifold_intersections
export estimate_tangent_space
export check_transversality_at_point
export check_transversality
export has_homoclinic_orbit
export has_heteroclinic_orbit
export all_manifolds_transverse
export count_heteroclinic_connections
export compute_separatrices
export trace_manifold_branch
export manifold_to_coordinates

# Export classification types
export SystemClass
export GRADIENT, GRADIENT_LIKE, MORSE_SMALE, STRUCTURALLY_STABLE, GENERAL, UNDETERMINED
export ClassificationResult

# Export classification functions
export classify_system
export quick_classify
export get_system_class
export is_gradient_system
export is_gradient_like_system
export is_morse_smale_system
export is_structurally_stable
export is_gradient
export is_gradient_like
export is_morse_smale
export allows_periodic_orbits
export has_landscape_representation
export classification_summary
export print_classification
export fixed_point_summary
export periodic_orbit_summary
export class_hierarchy_level
export is_subclass
export is_subclass_with_dim
export compare_classifications

end # module FlowClass
