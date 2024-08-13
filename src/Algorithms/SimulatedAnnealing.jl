"""
Implement a simulated annealing algorithm for KPP-Negative (maximize 
    fragmentation).

    TEMPORARY: ADDED A COMMENT


## Key Functions

`sann_continuation`
`sann_iterand!`
`sann_log_iteration!`
`sann_log_result!`


##  Key Structures

`sann_iterand`: the GraphOptimizationIterators.Iterand object used inside of 
    heuristic optimization to implement simulated annealing. This object 
    defines steps to take inside each iteration, ideal state swaps, stopping 
    conditions, acceptance thresholds, and more.
"""


###########################
#    BEGIN DEFINITIONS    #
###########################

"""
Implement the acceptance probability and check if the new iteration is accepted

# Constructs

```
sann_acceptance_test_threshold(
    energy_current::Float64,
    energy_new::Float64;
    alpha::Float64 = 0.001,
    algorithm_direction::Symbol = :increasing,
)
```

##  Function Arguments

- `energy_current`: current energy
- `energy_new`: new energey
- `T`: temperature


##  Keyword Arguments

- `rand`: optional random number for checking probability of acceptance
- `return_rand`: return the random trial (or `nothing` if not applicable). If
    `true`, returns a tuple of the form

    (bool, rand)

    where `bool` is the test conclusion and `rand` is the random value used to
    detremine acceptance

"""
function sann_acceptance_test_threshold(
    energy_current::Float64,
    energy_new::Float64,
    T::Float64;
    optimization_direction::Symbol = :maximize,
    rand::Union{Float64, Nothing} = nothing,
    return_rand::Bool = false,
)
    # accept if improving
    scal = (optimization_direction == :maximize) ? 1 : -1
    (energy_new*scal > energy_current*scal) && (return true)


    # check whether the energy values meet the acceptance criteria
    p_acceptance = sann_probability(
        energy_current,
        energy_new,
        T;
        optimization_direction = optimization_direction,
    )
    
    out = true

    # if it's non-improving
    if p_acceptance < 1.0
        rand = isa(rand, Nothing) ? Random.rand() : max(min(1.0, rand), 0.0)
        out = (rand <= p_acceptance)
    end

    out = return_rand ? (out, rand) : out
    
    return out
end



"""
Implement the acceptance probability and check if the new iteration is accepted

# Constructs

```
sann_accept_iteration(
    energy_current::Float64,
    energy_new::Float64,
    T::Float64;
    acceptance_function::Symbol = :threshold_acceptance
)
```


##  Function Arguments

- `energy_current`: current energy
- `energy_new`: new energey
- `T`: temperature


##  Keyword Arguments

- `acceptance_function`: Symbol specificying the acceptance function to use. Options are:
    * `:threshold_acceptance`: use the threshold of acceptance (fixed value) approach 
        cite: [P. Moscato and J.F. Fontanari, Phys. Lett. A 146, 204 (1990)](https://d1wqtxts1xzle7.cloudfront.net/80373576/0375-9601_2890_2990166-l20220206-18150-1wjjlbn-libre.pdf?1644195509=&response-content-disposition=inline%3B+filename%3DStochastic_versus_deterministic_update_i.pdf&Expires=1695151578&Signature=Lp9ZxDF5EwNpFqxWvHHHmVzczQe4xjQv-SJLVn5USZR0vEaTeXAldmp2XtEhfKWLurq7xfw3hSEi2tZYxoJT1oRZHgYv8~eE0PWj1JeyIu57nVn-0lbiWhD85777C-Tlx01H~OjUpNrfRKUkyeIxqLOp-~6gcEpVjdJhZLgFqaxGJQYOjGRXhdhclE1NPQVch9lm2GaF6BnWntl3zuCC2lYPMh3~aCQlP05cQtEdvqEVhaYKv1wl4Z07x5Rigd3ZuNOS~ibPun1OubZ2SwmAN8z61qrE5XgWc64UvUn0H4FFySsG1d7Tdw7lJllQ9Eqa7JGW--Bq0uqB92HtYXqtNw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- `..kwargs`: keyword arguments that are passed to the call function 
    `sann_aceeptance_test_####`
"""
function sann_accept_iteration(
    energy_current::Float64,
    energy_new::Float64,
    T::Float64;
    acceptance_function::Symbol = :threshold,
    kwargs...
)::Union{Bool, Nothing}
    
    out = nothing
    
    if acceptance_function == :threshold
        out = sann_acceptance_test_threshold(
            energy_current, 
            energy_new,
            T;
            kwargs...
        )
    end
    
    return out
end



"""
Check if the error is too high and if, based on the iterand i & max_iter, the 
    current swap needs to reset to the best known. 
    
Returns a bool, where `true` indicates that the state S should be set back to 
    S_best and `false` indicates otherwise.


# Constructs

```
sann_check_if_reset_to_best(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    acceptable_error::Real,
    iter_start::Real,
)
```


##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run


##  Keyword Arguments

- `acceptable_error`: error acceptability in late cooling phase; if an error 
    relative to the known best exceeds this value, then will reset to best
    parameters and restart.
- `iter_start`: iteration where error comparison starts

"""
function sann_check_if_reset_to_best(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    obj_try::Real,
    acceptable_error::Real,
    iter_start::Real,
)
    out = false
    
    if params_iterator.i > iter_start

        error = (obj_try - params_iterator.obj_best)
        error *= (-1 + 2*Int64(params_optimization.objective_direction == :minimize))

        out = error > acceptable_error
    end

    return out
end



"""
Generate the probability of acceptance as a function of temperature and 
    objective function delta. Based on outline from Allison Z Liu:

    https://allisonznliu.com/simulated-annealing#:~:text=Simulated%20Annealing%20(SA),problem%20is%20NP%2DHard)


# Constucts

```
sann_probability(
    e_current::Float64,
    e_proposed::Float64,
    temp::Float64;
    min_temp::Float64 = 10.0^(-9.0),
    optimization_direction::Symbol = :maximize,
)
```

##  Function Arguments

- `e_current`: objective function value in current state
- `e_proposed`: objective function value in proposed state
- `temp`: current temperature in system (> 0)


##  Keyword Arguments

- `optimization_direction`: Symbol, either :maximize or :minimize. Determines
    the direction of dele
"""
function sann_probability(
    e_current::Float64,
    e_proposed::Float64,
    temp::Float64;
    optimization_direction::Symbol = :maximize,
)
    # set a floor for the minimum temperature
    #min_temp = 10.0^(-9.0) # max(min_temp, 10.0^(-12.0))
    temp = max(temp, 10.0^(-9.0))

    #=
    If delta_e > 0, then the proposed state is smaller than the current state
        - If we are maximizing, then we want the probability to be < 1, and 
            delta_e shoule be < 0, so multiply by -1
        - If we are minimizing, then we want the probability to be == 1, and 
            delta_e shoule be > 0, so multiply by 1
    =#
    delta_e = e_current - e_proposed 
    delta_e *= (optimization_direction == :maximize) ? -1.0 : 1.0
    p = min(exp(delta_e/temp), 1.0)
    
    return p
end



"""
Set the temperature function based on iteration. In general, the temperature
    should decrease (cooling) over time, minimizing the probability of accepting
    a new, worse state. Early on, higher temperatures allow for a wider range of
    space exploration.

# Constructs

```
sann_temperature(
    energy_current::Float64,
    energy_new::Float64,
    T::Float64;
    acceptance_function::Symbol = :threshold_acceptance
)
```


##  Function Arguments

- `iter`: current iteration
- `alpha`: scalar to apply
- `q`: exponent in denominator; higher values of q signal faster cooling rates


##  Keyword Arguments

"""
function sann_temperature(
    iter::Int64;
    alpha::Real = 1.0,
    q::Real = 0.5,
)
    temp = alpha/((iter + 1)^q)
    return temp
end



"""
Define conditions under when to break iteration

# Constructs

```
sann_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool
```

##  Function Arguments

- `ip`: parameters used in iteration
- `op`: parameters used to set up the algorithm run


# Returns

Returns a `Bool` defining whether or not to continue iterating
"""
function sann_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool

    out = (ip.i < op.max_iter) 
    #out &= (ip.eps >= op.epsilon) 
    #out &= (ip.i_no_improvement < op.max_iter_no_improvement)
    out &= ip.cont

    return out
end



"""
Iterand function for greedy optimization of graph fragmentation

# Constructs

```
sann_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```


##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run


##  Keyword Arguments

- `alpha`: scalar applied in temperature function (see ?sann_temperature)
- `max_exploration_error`: error acceptability in late cooling phase; if an 
    error relative to the known best exceeds this value, then will reset to best
    parameters and restart.
- `q`: exponential cooling rate in temperature function (see ?sann_temperature)
- `restart_from_best_fraction`: fraction of time where the algorithm checks the 
    error and reverts to the best known outcome if attempt errors are too high
    (i.e., if they exceeed `max_exploration_error`). This process is only 
    applied in the latter part of the algorithm--e.g., starting at time 
    
        params_optimization.max_iter*(1 - restart_from_best_fraction)
    
    to reduce the chances of getting caught in a dominated local extrema.
"""
function sann_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters;
    alpha::Float64 = 1.0,
    max_exploration_error::Float64 = 0.25,
    q::Float64 = 0.5,
    restart_from_best_fraction::Float64 = 0.33,
)
    
    # set the "temperature"
    temperature = sann_temperature(
        params_iterator.i;
        alpha = alpha,
        q = q,    
    )
    
    #randomize_swap_count = params_optimization.randomize_swap_count
    swap_size = 1#randomize_swap_count ? sample(1:n_nodes, 1; replace = false)[1] : 1
    S_try, S_out, S_in = sample_for_swap(params_optimization, swap_size)

    """
    could try it with state memory
    obj_try = get(params_iterator.dict_optional_use, S_eval_out_tup, nothing)
    
    if !isa(obj_try, Nothing)
        if obj_try > max_f_inner_iter
            S_f_inner_iter = S_eval_out
            max_f_inner_iter = obj_try
        end

        continue
    end

    graph_try = copy(params_optimization.graph)
    rem_vertex!.((graph_try, ), S_try)
    obj_try = fragmentation(graph_try)

    """
    # recalculate distances
    obj_try = get_objective_from_removed_vertices(
        params_optimization, 
        S_try,
    )


    ##  DETERMINE WHETHER TO ACCEPT THE ITERATION

    # check the error (use params_iterator.obj_best since it is never nothing)
    reset_to_best = sann_check_if_reset_to_best(
        params_iterator,
        params_optimization,
        obj_try,
        max_exploration_error,
        params_optimization.max_iter*(1 - restart_from_best_fraction)
    )

    if reset_to_best 
        accept_swap = true
        obj_try = params_iterator.obj_best
        S_try = params_optimization.S_best

    else
        accept_swap = sann_accept_iteration(
            params_iterator.obj,
            obj_try,
            temperature;
            acceptance_function = :threshold,
            optimization_direction = params_optimization.objective_direction,
        )

    end
    
    
    if accept_swap

        msg = "\n\n** keeping swap at iteration $(params_iterator.i) - F = $(params_iterator.obj), F_0 = $(params_iterator.obj_0), F_try = $(obj_try)\n"

        update_accepted_swap_params!(
            params_iterator,
            params_optimization,
            obj_try,
            S_try;
            msg = msg,
        )
    end

end



"""
Iteration logging function (within iteration)

# Constructs

```
sann_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function sann_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("$(params_iterator.i) iterations complete with value $(params_iterator.obj)")
end



"""
Iteration logging function (after iteration)

# Constructs

```
sann_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function sann_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("Simulated annealing complete in $(params_iterator.i) iterations.") 
end



"""
Iterand for the greedy optimization approach
"""
sann_iterand = Iterand(
    sann_continuation,
    sann_iterand!;
    log_iteration! = sann_log_iteration!,
    log_result! = sann_log_result!,
    opts_prefix = :sann,
)
