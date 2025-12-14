---
title: 'FlowClass.jl: Classifying Dynamical Systems by Structural Properties in Julia'
tags:
  - Julia
  - dynamical systems
  - gradient systems
  - Morse-Smale
  - Waddington landscape
  - systems biology
  - cell differentiation
authors:
  - name: [Michael P.H. Stumpf]
  orcid: 0000-0002-3577-1222
  affiliation: 1,2,3
affiliations:
  - name: [School of BioScience, University of Melbourne, Melbourne Australia]
  index: 1
-  name: [School of Mathematics and Statistics, University of Melbourne, Melbourne Australia]
  index: 2
-  name: [Cell Bauhaus PTY LTD, Melbourne Australia]
  index: 3
date: 13 December 2025
bibliography: paper.bib
---

# Summary

`FlowClass.jl` is a Julia package [@Bezanson:2017,@Roesch2023] for classifying continuous-time dynamical systems into a hierarchy of structural classes: Gradient, Gradient-like, Morse-Smale, Structurally Stable, and General. Given a vector field $\mathbf{F}(\mathbf{x})$ defining the system $\mathrm{d}\mathbf{x}/\mathrm{d}t = \mathbf{F}(\mathbf{x})$, the package performs a battery of computational tests—Jacobian symmetry analysis, curl magnitude estimation, fixed point detection and stability classification, periodic orbit detection, and stable/unstable manifold computation—to determine where the system sits within the classification hierarchy. This classification has direct implications for qualitative behaviour: gradient systems cannot oscillate, Morse-Smale systems are structurally stable in less than 3 dimensions, and general systems may exhibit chaos. Much of classical developmental theory going back to Waddington's epigenetic landscape  [@Waddington1957] rests on an implicit assumption of gradient dynamics. 

The package is designed with applications in systems and developmental biology in mind, particularly the analysis of gene regulatory networks and cell fate decision models in the context of Waddington's epigenetic landscape. It provides tools to assess whether a landscape metaphor is appropriate for a given dynamical model, and to quantify the magnitude of non-gradient (curl) dynamics.

# Statement of Need

Dynamical systems models are ubiquitous in biology, physics, and engineering [@Strogatz2015]. A central question when analysing such models is whether the system can be understood in terms of a potential landscape—that is, whether trajectories follow the gradient of a potential function toward attracting states. This question is especially pertinent in developmental biology [@Huang2007,@Moris2016], where Waddington's epigenetic landscape provides a powerful metaphor for cell differentiation [@Waddington1957]. However, as recent work has emphasised, most biological systems are *not* gradient systems: they exhibit curl dynamics arising from non-reciprocal interactions in gene regulatory networks [@Brackston2018; @Wang2015].

Despite the theoretical importance of this distinction [@Smale1967,@PalisDeMelo1982], there has been no comprehensive software package for computationally classifying dynamical systems by their structural properties. Existing tools focus on specific aspects—bifurcation analysis, Lyapunov exponent computation, or trajectory simulation—but do not provide an integrated framework for structural classification. `FlowClass.jl` fills this gap by implementing a systematic classification pipeline that moves from the most restrictive class (gradient systems) to the most general, providing quantitative measures at each stage.
![](JOSS-Fig1.pdf)
*Figure 1. Waddington's epigenetic landscape showing potential wells corresponding to cell fates. On the left hand side the flow is purely gradient. On the right hand side the flow has curl components and the dynamics are no longer determined by the landscape.*

The classification has practical consequences. For gradient systems, the potential function fully characterises the dynamics, and forward and reverse transition paths coincide. For systems with even moderate curl fluxes, minimum action paths for transitions in opposite directions differ, which has implications for understanding differentiation versus reprogramming in stem cell biology [@Brackston2018,@Guillemin2020]. By quantifying the curl-to-gradient ratio, researchers can assess the validity of landscape-based analyses for their specific models. This for example, makes it possible to infer aspects of stem cell dynamics from single cell data [@Liu20024].

# Classification of Dynamical Systems

**Table 1.** Properties of dynamical system classes in the classification hierarchy.

| Property          | Gradient                 | Gradient-like           | Morse-Smale       | Generic                                 | General        |
| ----------------- | ------------------------ | ----------------------- | ----------------- | --------------------------------------- | -------------- |
| Curl              | Zero everywhere          | Near zero/small         | Non-zero possible | Any value                               | Any value      |
| Jacobian          | Symmetric (J = Jᵀ)       | Nearly symmetric        | No requirement    | No requirement                          | No requirement |
| Path integrals    | Path-independent         | Nearly path-independent | Path-dependent    | Path-dependent                          | Path-dependent |
| Periodic orbits   | None                     | None                    | Hyperbolic only   | Hyperbolic & non-hyperbolic             | Any            |
| Critical points   | All hyperbolic           | All hyperbolic          | All hyperbolic    | Non-hyperbolic possible at bifurcations | Any            |
| Lyapunov function | Global (potential V)     | Global                  | Local only        | Local (away from bifurcations)          | May not exist  |
| Transversality    | N/A (no periodic orbits) | N/A                     | Required          | May have tangencies                     | Not required   |

Our ability to make qualitative statements about a dynamical system depends crucially on the nature of the dynamics [@Smale1967,@PalisDeMelo1982,@ BrackstonPhysRevE2018]. The qualitative aspects can deliver profound biological systems. If stem cell differentiation were to follow gradient dynamics, for example, then the forward and backward paths through gene expression space would be identical [@Guillemin2020,Vittadello2025]. For gradient systems, and gradient-like systems we have access to Lyapunov functions, and concepts from catastrophe theory can be applied and yield powerful insights [@Rand2021]. 

For more general dynamical systems we cannot fall back on such elegant theory. A main focus in the design of`FlowClass.jl` was to classify dynamical systems into the relevant categories that determine whether or not a given system (at least in the specific parameterisation considered). The routines provided as part of the package make it possible to determine the characterising features of different classes of dynamical systems, (c.f. Table 1). 


# Key Features

**Jacobian analysis.** The package computes Jacobian matrices using automatic differentiation via `ForwardDiff.jl` and tests for symmetry. For a gradient system $\mathbf{F} = -\nabla V$, the Jacobian equals the negative Hessian and is therefore symmetric. The relative symmetry error $\|(J - J^\top)/2\|_F / \|J\|_F$ provides a scale-independent measure of deviation from gradient structure.

**Curl quantification.** For 2D and 3D systems, `FlowClass.jl` computes the curl directly. For higher-dimensional systems, it uses the Frobenius norm of the antisymmetric part of the Jacobian as a generalised curl measure. The curl-to-gradient ratio indicates the relative strength of rotational versus potential-driven dynamics.

**Fixed point analysis.** Multi-start optimisation via `NLsolve.jl` locates fixed points within user-specified bounds. Each fixed point is classified by eigenvalue analysis into stable nodes, unstable nodes, saddles, foci, centres, or non-hyperbolic points. The presence of non-hyperbolic fixed points excludes the system from the Morse-Smale class.

**Periodic orbit detection.** Poincaré section methods detect limit cycles, and Floquet multiplier analysis determines their stability. Gradient and gradient-like systems cannot possess periodic orbits; their presence indicates at least Morse-Smale classification.

**Manifold computation.** For saddle points, the package computes stable and unstable manifolds by integrating trajectories from initial conditions perturbed along eigenvector directions. Transversality of manifold intersections is assessed numerically as this is a requirement for Morse-Smale structure.

**Integrated classification.** The `classify_system` function orchestrates all analyses and returns a `ClassificationResult` containing the system class, all detected invariant sets, quantitative measures (Jacobian symmetry, curl ratio), and a confidence score.

# Example: Stem Cell Differentiation

As a demonstration, `FlowClass.jl` includes an implementation of the stem cell differentiation model from @Brackston2018, who built on earlier work by @Chickarmane2012, which describes the dynamics of pluripotency factors (Nanog, Oct4-Sox2, Fgf4) and the differentiation marker Gata6. The model exhibits multiple stable states corresponding to pluripotent and differentiated cell fates, with a saddle point representing the transition state. Classification reveals significant curl dynamics, confirming that the system is not gradient and that differentiation and reprogramming paths will differ—a key biological insight.

```julia
using FlowClass

ds = DynamicalSystem(stem_cell_model, 4)
bounds = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 120.0))
result = classify_system(ds, bounds)
print_classification(result)
```

# Related Software

Several Julia packages provide complementary functionality. `DifferentialEquations.jl` [@Rackauckas2017] offers comprehensive ODE/SDE solvers but does not perform structural classification. `BifurcationKit.jl` focuses on continuation and bifurcation analysis. `DynamicalSystems.jl` [@Datseris2018] provides tools for chaos detection and attractor characterisation. `GAIO.jl` [@GAIO2025] provides numerical routines for the  global analysis of dynamical systems.  `FlowClass.jl` complements these by addressing a distinct question: not *how* a system behaves, but *what kind* of system it is structurally.

In Python, `PyDSTool` and `PySCeS` offer dynamical systems modelling for biology, but neither provides systematic structural classification. The landscape and flux decomposition methods of @Wang2015 are related theoretically; `FlowClass.jl` provides practical tools for the classification aspect of this framework.

# Conclusion
Our ability to make qualitative statements about a dynamical system depends crucially on the nature of the dynamics. The qualitative aspects can deliver profound biological systems. If stem cell differentiation were to follow gradient dynamics, for example, then the forward and backward paths through gene expression space would be identical [@Guillemin2020,Vittadello2025]. For gradient systems, and gradient-like systems we have access to Lyapunov functions, and concepts from catastrophe theory can be applied and yield powerful insights [@Rand2021]. 

For more general dynamical systems we cannot fall back on such elegant theory. A main focus in the design of`FlowClass.jl` was to classify dynamical systems into the relevant categories that determine whether or not a given system (at least in the specific parameterisation considered). The routines provided as part of the package make 
  
# Acknowledgements

The author thanks the Australian Research Council for financial support through an ARC Laureate Fellowship (FL220100005)
# References
