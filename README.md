# TITLE

## Introduction

Network disruption problems involve the removal of key nodes through which different components flow--for example, money, information, energy, etc. This problem has been formalized as the _Key Player Problem_ introduced by [Borgatti (2006)](https://link.springer.com/article/10.1007/s10588-006-7084-x). These problems are often practically constrained, meaning that actors seeking to disrupt networks are constrained to the removal of, at maximum, a limited number of nodes.

To characterize network disruption, Borgatti discusses a concept called *Distance-Based Fragmentation*, or fragmentation. Fragmentation captures the average of the inverse distance between nodes in the graph; higher measures indicate a greater average distance between all vertices on the graph, making it more difficult for elements to flow across the network. We use a generalized version that allows for the calculation of fragmentation on directed graphs, i.e.,

$$\,^DF(G) = 1 - \frac{1}{n(n - 1)}\sum_{i \not= j}d_{ij}^{-1}.$$

In smaller graphs, optimal fragmentation can usually be calculated using a brute force combinatorial approach. However, as the number of vertices and edges grows, the problem can become computationally intractable. This package combines the power of Julia with a suite of parallelizable discrete graph optimization algorithms to make many real-world networks tractable.


### `DiscreteGraphAlgorithms.jl`

The `DiscreteGraphAlgorithms.jl` package includes a suite of several algorithms for discrete subgraph optimization. These algorithms identify subsets of vertices that identify local extrema of an objective function calculated on the selection of vertices included in some subgraph of size _k_. `DiscreteGraphAlgorithms.jl` allows for distributed executation of distance algorithms using data parallelization (e.g., `@distributed` instead of `@thread`) and efficient memory management, allowing for faster solution times and less memory pressure.

At this time, the only objective function that has been implemented is `fragmentation`, based on the `GraphFragments.jl` package. However, `DiscreteGraphAlgorithms.jl` is designed to be updated to integrate additional metrics that might be of interest for vertex subsets.


#### Algorithms

The `DiscreteGraphAlgorithms.jl` includes implementations of several algorithms. Each algorithm is associated with an `Iterand` object, or a set of operations conducted within the general `DiscreteGraphAlgorithms.iterate` framework`. 

`DiscreteGraphAlgorithms.jl` includes the following algorithms:

1. *Ant Colony Optimization*
    - Ant Colony Optimization (ACO) is a version of particle-swarm optimization that reflects the ability of ant colonies to identify food sources and lay pheremone trails to efficiently exploit these sources. The `DiscreteGraphAlgorithms` uses ACO to identify locally-optimal subsets of vertices using a modified set covering problem approach; ACO for the set covering problem approach was defined by Leguizamón\& Michalewicz (2000) and Hadji et al. (2000) and reviewed by Dorigo and Stützle (2004).
    - Iterand: `DiscreteGraphAlgorithms.aco_iterand`

1. *Genetic*
    - Genetic algorithms are a class of algorithms that are derived from the biological process of evolution, including crossover between parents and mutations of genetic codes. 
    - Iterand: `DiscreteGraphAlgorithms.genetic_iterand`

1. *Gradient Descent*
    - The gradient descent algorthm included herein is a discrete analog of the gradient descent optimization heuristic, which follows the "steepest path" down in differentiable functions. In this discrete implementation, the algorithm searches for the best swap for a single element in the current set of vertices for an element outside of the vertices. This can be very efficient on smaller gra532826
    phs while time-consuming on larger graphs (each iteration requires $k*(n - k)$ calculations of the objective).
    - Iterand: `graddesc_iterand`

1. *Greedy Stochastic*
    - Greedy algorithms are a class of algorithms that only make decisions about where to go based on the best local decision. This algorithm randomly swaps out elements from the current best set of vertices for elements outside of the set, iterating until stopping conditions are met. 
    - Iterand: `DiscreteGraphAlgorithms.gs_iterand`

1. *Simulated Annealing*
    - Simulated annealing (SA) is a stochastic approach to optimization that mimics the process of annealing in metallurgy, or the heating and cooling of a material. In general, simulated annealing sets probabilities to facilitate for a broad search space early in the iterative process, reprsenting the heating of a material. As iteration proceeds--and the "material cools"--the probability of including previously successful components increases, and the algorithm converges toward local optima. 
    - Iterand: `DiscreteGraphAlgorithms.sann_iterand`

The `DiscreteGraphAlgorithms.jl` package uses a generic iterator structure defined in GraphOptimizationIterators to manage iterands defined for each of the algorithms illuminated above.


## Use

To use the algorithm to identify optimal removal vertices based on the Key Player Problem, users perform four steps:

1. *Load `DiscreteGraphAlgorithms`*
2. *Get the graph*: Specify `DiscreteGraphAlgorithms.GraphWrapper`. This can be defined from a `Graphs.jl AbstractGraph` or by reading in a sparse edge list using `read_egl` (see `?read_egl` for more information)
3. *Define the Optimization Parameters*: Using an `OptimizationParameters` object, define the number of vertices _k_ to remove from the graph, add a graph wrapper object, and specify other parameters, such as maximum number of iterations, the maximum numnber of iterations with no improvement, or algorihthm-specific parameters, which are passed through a special dictionary using the `opt` keyword argument. For more information, see `?OptimizationParameters`.
    - Note: users can specify the distance algorithm they want to use for distance-based metrics. However, by default, the `OptimizationParameters` will perform some comparative benchmarking to empirically select a distance algortihm to use for the graph being analyzed. 

4. *Run the algorithm*: Using `DiscreteGraphAlgorithms.iterate` function, pass the `OptimizationParameters` and the `Iterand` object you want to use. The `Iterand` object (see the list of algorithms above for the specification of the associated `Iterand`) represents the algorithm chosen by the user. `DiscreteGraphAlgorithms.iterate`returns a tuple of the following form:

    (
        obj_best,
        S_best,
        grpah_best
    )

    where:

    - `obj_best` is the best value of the objective found by the algorithm
    - `S_best` is the set of vertices to remove to achieve `obj_best`
    - `graph_best` is the modified graph ($G(V\S, E\S)$)

    **NOTE**: Algorithm parameters can significantly impact performance. For example, in the genetic and ant-colony algorithms, the number of orgnisms or ants can significantly affect performance. Furthermore, both use elitism to ensure monotonic behavior of the objective function. Both of these parameters--population and elite fraction--can be controled by passing information to `OptimizationParameters.opts` using the algorithm's appropriate options prefix (see `?OptimizationParameters` for more)


### Example

```julia

using DiscreteGraphAlgorithms

# read a sparse adjacency matrix to a GraphWrapper, which pairs a Graphs.jl AbstractGraph with some additional properties
graph_wrapper_fragment = read_egl(
    file_path_to_sparse_graph_csv;
)

# define optimization parameters
op_fragment = OptimizationParameters(
    5, # number of vertices to identify in optimal subgraph
    graph_wrapper_fragment; # graph to run on  
    max_iter = 200, # maximum number of iterations
    max_iter_no_improvement = 20, # maximum number of iterations with no improvement
)


# call the iterator to retrieve information
out_fragment = DiscreteGraphAlgorithms.iterate(
    genetic_iterand,
    op_fragment; 
    log_interval = 10, 
)
```



## Data

The code only needs a Graph to work off of. This can be loaded in a Julia session and converted to a `GraphWrapper`, or a `GraphWrapper` object can be created directly using `read_egl`. See `?read_egl` for information about arguments and keyword arguments.

Example data--including the Krebs terrorist network (Krebs, CITE)--are included in this package 


## Project information

The authors are grateful to RAND Center for Global Risk and Security Advisory Board members Michael Munemann and Paul Cronson for funding this project. All code was developed between April 2023 and October 2024.



## References/Bibliography

Borgatti, S.P. Identifying sets of key players in a social network. Comput Math Organiz Theor 12, 21–34 (2006). [doi](https://doi.org/10.1007/s10588-006-7084-x)

Dorigo, Marco and Stützle, T.. Ant Colony Optimization. 2004. The MIT Press, Cambridge, Massachusetts. ISBN 0-262-04219-3. [MIT Press](https://mitpress.mit.edu/9780262042192/ant-colony-optimization/)

Hadji, R., Rahoual, M., Talbi, E. G., & Bachelet, V. (2000). Ant colonies for the set covering problem. In Abstract proceedings of ANTS (pp. 63-66). [Research Gate](https://www.researchgate.net/publication/245585705_Ant_Colonies_for_the_Set_Covering_Problem)

Katoch, S., Chauhan, S.S. & Kumar, V. A review on genetic algorithm: past, present, and future. Multimed Tools Appl 80, 8091–8126 (2021). [doi](https://doi.org/10.1007/s11042-020-10139-6)

Krebs, Valdis. (2002). Mapping Networks of Terrorist Cells. 24. [Research Gate](https://www.researchgate.net/publication/2490397_Mapping_Networks_of_Terrorist_Cells)

Leguizamón, Guillermo, Michalewicz, Z. and Schutz, M. An ant system for the maximum independent set problem. VII Congreso Argentino de Ciencias de la Computación (2001). [Article](https://sedici.unlp.edu.ar/handle/10915/23384)  

Peixoto, T. terrorists_911 — 9-11 terrorist network. Accessed Feb 2024. Netzschleuder network catalogue, repository and centrifuge. https://networks.skewed.de/net/terrorists_911 [Krebs Terrorist Network Data](https://networks.skewed.de/net/terrorists_911)


 

## Copyright and License

Copyright (C) <2024> RAND Corporation. This code is made available under the MIT license.

 

## Authors and Reference

James Syme

@misc{GDA2024,
  author       = {Syme, James},
  title        = {DiscreteGraphAlgorithms.jl: Distributed implementation of some graph shortest distance algorithms.},
  year         = 2024,
  url = {URLHERE}
}
