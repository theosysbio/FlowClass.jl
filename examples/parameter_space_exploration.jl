# Parameter Space Exploration for System Classification
# 
# This script explores whether the stem cell differentiation model from
# Brackston et al. (2018) can be classified as gradient or gradient-like
# under random parameter perturbations.
#
# The approach samples parameter combinations log-uniformly, classifies
# each realisation, and reports the distribution of structural classes.

using FlowClass
using Random
using Statistics
using Printf

# ============================================================================
# Stem Cell Model Definition (Eqns. 8–16 from Brackston et al. 2018)
# ============================================================================

"""
    stem_cell_model(x, p; L=50.0)

Stem cell differentiation model with explicit parameter vector.

# Arguments
- `x`: State vector [N, O, F, G] (Nanog, Oct4-Sox2, Fgf4, Gata6)
- `p`: Named tuple of kinetic parameters (k0, k1, ..., k14, kd)
- `L`: LIF concentration (external signal)

# Returns
- Vector of time derivatives [dN/dt, dO/dt, dF/dt, dG/dt]
"""
function stem_cell_model(x, p; L=50.0)
    N, O, F, G = x
    
    # Production propensities (Eqns. 8–11)
    a1 = p.k0 * O * (p.k1 + p.k2*N^2 + p.k0*O + p.k3*L) / 
         (1 + p.k0*O*(p.k2*N^2 + p.k0*O + p.k3*L + p.k4*F^2) + p.k5*O*G^2)
    
    a2 = (p.k6 + p.k7*O) / (1 + p.k7*O + p.k8*G^2)
    
    a3 = (p.k9 + p.k10*O) / (1 + p.k10*O)
    
    a4 = (p.k11 + p.k12*G^2 + p.k14*O) / (1 + p.k12*G^2 + p.k13*N^2 + p.k14*O)
    
    # Net rates: production − degradation
    dN = a1 - p.kd * N
    dO = a2 - p.kd * O
    dF = a3 - p.kd * F
    dG = a4 - p.kd * G
    
    return [dN, dO, dF, dG]
end

# ============================================================================
# Parameter Sampling
# ============================================================================

# Parameter names for the model
const PARAM_NAMES = [:k0, :k1, :k2, :k3, :k4, :k5, :k6, :k7, :k8, :k9, 
                     :k10, :k11, :k12, :k13, :k14, :kd]

# Reference parameters from Brackston et al. (2018)
const REFERENCE_PARAMS = (
    k0 = 0.005, k1 = 0.01, k2 = 0.4, k3 = 1.0, k4 = 0.1,
    k5 = 0.00135, k6 = 0.01, k7 = 0.01, k8 = 1.0, k9 = 1.0,
    k10 = 0.01, k11 = 5.0, k12 = 1.0, k13 = 0.005, k14 = 1.0,
    kd = 1.0
)

"""
    sample_log_uniform(lower, upper)

Sample a value log-uniformly from [lower, upper].

This means ln(θ) ~ Uniform(ln(lower), ln(upper)), giving equal probability
per decade of the parameter range.
"""
function sample_log_uniform(lower::Float64, upper::Float64)
    log_lower = log(lower)
    log_upper = log(upper)
    return exp(log_lower + rand() * (log_upper - log_lower))
end

"""
    sample_parameters(; lower=0.001, upper=1000.0)

Sample a complete parameter set with each parameter drawn log-uniformly.

# Arguments
- `lower`: Lower bound for all parameters (default: 0.001)
- `upper`: Upper bound for all parameters (default: 1000.0)

# Returns
- Named tuple of sampled parameters
"""
function sample_parameters(; lower::Float64=0.001, upper::Float64=1000.0)
    values = [sample_log_uniform(lower, upper) for _ in PARAM_NAMES]
    return NamedTuple{Tuple(PARAM_NAMES)}(values)
end

# ============================================================================
# Classification Wrapper
# ============================================================================

"""
    ClassificationSummary

Summary statistics from classifying a single parameter realisation.
"""
struct ClassificationSummary
    params::NamedTuple
    system_class::SystemClass
    jacobian_symmetry::Float64
    curl_ratio::Float64
    n_fixed_points::Int
    is_valid::Bool  # Whether classification succeeded without errors
end

"""
    classify_with_params(params; L=50.0, bounds=nothing, n_samples=50)

Create a dynamical system with the given parameters and classify it.

# Arguments
- `params`: Named tuple of kinetic parameters
- `L`: LIF concentration
- `bounds`: State space bounds (default: [0,100]⁴)
- `n_samples`: Number of sample points for classification

# Returns
- `ClassificationSummary` with results
"""
function classify_with_params(params; 
                              L::Float64=50.0,
                              bounds=nothing,
                              n_samples::Int=50)
    
    if bounds === nothing
        bounds = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 120.0))
    end
    
    # Create the dynamical system
    f = x -> stem_cell_model(x, params; L=L)
    ds = DynamicalSystem(f, 4)
    
    try
        result = classify_system(ds, bounds; n_samples=n_samples)
        
        return ClassificationSummary(
            params,
            result.system_class,
            result.jacobian_symmetry,
            result.curl_gradient_ratio,
            length(result.fixed_points),
            true
        )
    catch e
        # Return a failed classification
        return ClassificationSummary(
            params,
            GENERAL,
            NaN,
            NaN,
            0,
            false
        )
    end
end

# ============================================================================
# Parameter Space Exploration
# ============================================================================

"""
    ExplorationResults

Results from exploring parameter space.
"""
struct ExplorationResults
    n_samples::Int
    n_valid::Int
    class_counts::Dict{SystemClass, Int}
    class_fractions::Dict{SystemClass, Float64}
    summaries::Vector{ClassificationSummary}
    gradient_params::Vector{NamedTuple}      # Parameters yielding GRADIENT
    gradient_like_params::Vector{NamedTuple} # Parameters yielding GRADIENT_LIKE
end

"""
    explore_parameter_space(; n_samples=1000, lower=0.001, upper=1000.0,
                            L=50.0, bounds=nothing, verbose=true, seed=nothing)

Explore parameter space by sampling random parameter combinations and
classifying each resulting dynamical system.

# Arguments
- `n_samples`: Number of parameter combinations to sample (default: 1000)
- `lower`: Lower bound for log-uniform sampling (default: 0.001)
- `upper`: Upper bound for log-uniform sampling (default: 1000.0)
- `L`: LIF concentration for the model (default: 50.0)
- `bounds`: State space bounds for classification
- `verbose`: Print progress updates (default: true)
- `seed`: Random seed for reproducibility (default: nothing)

# Returns
- `ExplorationResults` struct with classification statistics
"""
function explore_parameter_space(; 
                                  n_samples::Int=1000,
                                  lower::Float64=0.001,
                                  upper::Float64=1000.0,
                                  L::Float64=50.0,
                                  bounds=nothing,
                                  verbose::Bool=true,
                                  seed=nothing)
    
    if seed !== nothing
        Random.seed!(seed)
    end
    
    if verbose
        println("=" ^ 70)
        println("Parameter Space Exploration for Stem Cell Model")
        println("=" ^ 70)
        @printf("Sampling %d parameter combinations\n", n_samples)
        @printf("Parameter range: [%.3g, %.3g] (log-uniform)\n", lower, upper)
        @printf("LIF concentration: %.1f\n", L)
        println("-" ^ 70)
    end
    
    # Initialise storage
    summaries = Vector{ClassificationSummary}(undef, n_samples)
    class_counts = Dict{SystemClass, Int}(
        GRADIENT => 0,
        GRADIENT_LIKE => 0,
        MORSE_SMALE => 0,
        STRUCTURALLY_STABLE => 0,
        GENERAL => 0
    )
    gradient_params = NamedTuple[]
    gradient_like_params = NamedTuple[]
    n_valid = 0
    
    # Progress tracking
    progress_interval = max(1, n_samples ÷ 10)
    
    for i in 1:n_samples
        # Sample parameters
        params = sample_parameters(; lower=lower, upper=upper)
        
        # Classify
        summary = classify_with_params(params; L=L, bounds=bounds)
        summaries[i] = summary
        
        if summary.is_valid
            n_valid += 1
            class_counts[summary.system_class] += 1
            
            if summary.system_class == GRADIENT
                push!(gradient_params, params)
            elseif summary.system_class == GRADIENT_LIKE
                push!(gradient_like_params, params)
            end
        end
        
        # Progress update
        if verbose && (i % progress_interval == 0 || i == n_samples)
            pct = 100.0 * i / n_samples
            @printf("  Progress: %5d / %d (%5.1f%%)\r", i, n_samples, pct)
        end
    end
    
    if verbose
        println()  # New line after progress
    end
    
    # Compute fractions
    class_fractions = Dict{SystemClass, Float64}()
    for (cls, count) in class_counts
        class_fractions[cls] = n_valid > 0 ? count / n_valid : 0.0
    end
    
    return ExplorationResults(
        n_samples,
        n_valid,
        class_counts,
        class_fractions,
        summaries,
        gradient_params,
        gradient_like_params
    )
end

"""
    print_exploration_results(results::ExplorationResults)

Print a formatted summary of the exploration results.
"""
function print_exploration_results(results::ExplorationResults)
    println("-" ^ 70)
    println("RESULTS")
    println("-" ^ 70)
    @printf("Total samples:  %d\n", results.n_samples)
    @printf("Valid samples:  %d (%.1f%%)\n", 
            results.n_valid, 100.0 * results.n_valid / results.n_samples)
    println()
    
    println("Classification Distribution:")
    println()
    
    # Define order for display
    class_order = [GRADIENT, GRADIENT_LIKE, MORSE_SMALE, STRUCTURALLY_STABLE, GENERAL]
    class_names = Dict(
        GRADIENT => "Gradient",
        GRADIENT_LIKE => "Gradient-like",
        MORSE_SMALE => "Morse-Smale",
        STRUCTURALLY_STABLE => "Structurally Stable",
        GENERAL => "General"
    )
    
    for cls in class_order
        count = results.class_counts[cls]
        frac = results.class_fractions[cls]
        bar_len = round(Int, 40 * frac)
        bar = "█" ^ bar_len * "░" ^ (40 - bar_len)
        @printf("  %-20s %5d (%6.2f%%) %s\n", 
                class_names[cls], count, 100.0 * frac, bar)
    end
    
    println()
    println("-" ^ 70)
    
    # Summary statistics for gradient/gradient-like
    n_landscape = results.class_counts[GRADIENT] + results.class_counts[GRADIENT_LIKE]
    frac_landscape = results.n_valid > 0 ? n_landscape / results.n_valid : 0.0
    
    println("Key Finding:")
    @printf("  Systems with landscape representation: %d / %d (%.2f%%)\n",
            n_landscape, results.n_valid, 100.0 * frac_landscape)
    
    if frac_landscape < 0.01
        println("  → Landscape representation is rare in this parameter space")
        println("  → The model generically exhibits non-gradient dynamics")
    elseif frac_landscape < 0.10
        println("  → Landscape representation occurs in a small fraction of parameter space")
    else
        println("  → Landscape representation is relatively common")
    end
    
    println("=" ^ 70)
end

# ============================================================================
# Analysis of Gradient-Yielding Parameters
# ============================================================================

"""
    analyse_gradient_parameters(results::ExplorationResults)

Analyse which parameter combinations yield gradient or gradient-like dynamics.
"""
function analyse_gradient_parameters(results::ExplorationResults)
    all_gradient = vcat(results.gradient_params, results.gradient_like_params)
    
    if isempty(all_gradient)
        println("\nNo gradient or gradient-like systems found in the sample.")
        return nothing
    end
    
    println("\n" * "=" ^ 70)
    println("Analysis of Gradient/Gradient-like Parameter Sets")
    println("=" ^ 70)
    @printf("Found %d gradient and %d gradient-like parameter sets\n",
            length(results.gradient_params), length(results.gradient_like_params))
    println()
    
    # Compute statistics for each parameter
    println("Parameter Statistics (for landscape-admitting systems):")
    println("-" ^ 70)
    @printf("%-8s %12s %12s %12s %12s\n", 
            "Param", "Median", "Mean", "Min", "Max")
    println("-" ^ 70)
    
    for name in PARAM_NAMES
        values = [getfield(p, name) for p in all_gradient]
        med = median(values)
        mn = mean(values)
        lo = minimum(values)
        hi = maximum(values)
        @printf("%-8s %12.4g %12.4g %12.4g %12.4g\n", 
                string(name), med, mn, lo, hi)
    end
    println("-" ^ 70)
    
    return all_gradient
end

# ============================================================================
# Comparison with Reference Parameters
# ============================================================================

"""
    classify_reference_parameters(; L=50.0, verbose=true)

Classify the system using the reference parameters from Brackston et al. (2018).
"""
function classify_reference_parameters(; L::Float64=50.0, verbose::Bool=true)
    if verbose
        println("\n" * "=" ^ 70)
        println("Reference Parameter Classification")
        println("=" ^ 70)
        println("Using parameters from Brackston et al. (2018) Table")
        @printf("LIF concentration: %.1f\n", L)
        println("-" ^ 70)
    end
    
    summary = classify_with_params(REFERENCE_PARAMS; L=L)
    
    if verbose
        println("Classification: $(summary.system_class)")
        @printf("Jacobian symmetry error: %.6f\n", summary.jacobian_symmetry)
        @printf("Curl/gradient ratio: %.6f\n", summary.curl_ratio)
        @printf("Fixed points found: %d\n", summary.n_fixed_points)
        println("=" ^ 70)
    end
    
    return summary
end

# ============================================================================
# Main Execution
# ============================================================================

"""
    run_exploration(; n_samples=1000, seed=42)

Run the complete parameter space exploration analysis.
"""
function run_exploration(; n_samples::Int=1000, seed::Int=42)
    # First, classify with reference parameters
    ref_summary = classify_reference_parameters(; L=50.0)
    
    # Explore parameter space
    results = explore_parameter_space(; 
                                       n_samples=n_samples, 
                                       seed=seed,
                                       verbose=true)
    
    # Print results
    print_exploration_results(results)
    
    # Analyse gradient-yielding parameters
    gradient_params = analyse_gradient_parameters(results)
    
    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_exploration(; n_samples=1000, seed=42)
end
