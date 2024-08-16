
"""
Structure containing information for fragmentation


- `A`: adjacency matrix used to initialize 
"""
struct GraphWrapper
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}}
    force_undirected::Bool
    vertex_names::Vector
    w::Vector{Float64}
    D::Matrix{Float64}
    D_inv::Matrix{Float64}
    dims::Tuple{Int64, Int64}
    graph::Union{SimpleGraph, SimpleDiGraph}
    
    
    function GraphWrapper(
        A::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}};
        force_undirected::Bool = false,
        vertex_names::Union{Vector, Nothing} = nothing,
        w::Union{Vector{Float64}, Nothing} = nothing,
    )
        
        # intialize some graph components
        A = isa(A, Matrix{Float64}) ? sparse(A) : A
        dims = (size(A)[1], sum(A))
        
        vertex_inds = collect(1:size(A)[1])
        vertex_names = isa(vertex_names, Nothing) ? vertex_inds : ((length(vertex_names) == dims[1]) ? vertex_names : vertex_inds)
        w = isa(w, Nothing) ? A.nzval : ((size(w) == size(A.nzval)) ? w : A.nzval)
        graph = LinearAlgebra.issymmetric(A) ? SimpleGraph(A) : SimpleDiGraph(A)

        D, D_inv = get_distance_matrices(A);
  
        return new(
            A,
            force_undirected,
            vertex_names,
            w,
            D,
            D_inv,
            dims,
            graph,
        )
    end
    
end



"""
# Information

Return a distance mtarix and a lower-triangular distance matrix (without 
    diagonal). Returns a tuple of the form

    `D`, `D_invs`

    where `D` is the distance matrix (based on the shorted path) and `D_invs` is 
    a matrix of distance inverses (excluding diagonal)


# Constructs

```
get_distance_matrices(
    adj::Union{SparseMatrixCSC, Matrix},
    D::SharedArray{Float64};
    digits_round::Int64 = 10,
    fp_shared_array::Union{String, Nothing} = nothing,
    graph::Union{SimpleDiGraph, SimpleGraph, Nothing} = nothing
)::Tuple{Matrix{Float64}, Matrix{Float64}}
```

```
get_distance_matrices(
    graph::Union{SimpleDiGraph, SimpleGraph, Nothing};
    algorithm::Symbol = :auto,
    digits_round::Int64 = 10,
)::Tuple{Matrix{Float64}, Matrix{Float64}}
```


## Function Arguments

- `adj`: sparse adjacency matrix used to set graph. Can be directed and/or 
    weighted
- `D`: SharedArray used to initialize the distance matrix; passing an already
    instantiated SharedArray allows repeat-running without running into issues
    with instantiating within the function (SharedArrays are not destroyed after
    a function completes)


## Keyword Arguments

- `algorithm`: optional algorithm to set. 
    * :auto
        If number of vertices is less than 100, calls 
        Graphs.dijkstra_shortest_paths(); otherwise calls
        Graphs.bellman_ford_shortest_paths()
    * :bellman_ford
    * :dijkstra
- `digits_round`: number of digits to round distance matrix to. Used to chop 
    Infs to 0.
- `graph`: optional graph associated with adj to pass
"""
function get_distance_matrices(
    adj::Union{SparseMatrixCSC, Matrix};
    algorithm::Symbol = :auto,
    digits_round::Int64 = 10,
    graph::Union{SimpleDiGraph, SimpleGraph, Nothing} = nothing
)::Tuple{Matrix{Float64}, Matrix{Float64}}
    
    # initialization
    m, n = size(adj)
    
    # initialize zeros
    D_invs = Float64.(zeros(size(adj)))
    D = Float64.(zeros(size(adj)))

    graph = isa(graph, Nothing) ? (
        LinearAlgebra.issymmetric(adj) ? SimpleGraph(adj) : SimpleDiGraph(adj)
    ) : graph
    

    # get the algorithm to use for the distance matrics
    alg_name = select_algorithm(algorithm; n_vertices = m)
    dict_algs = Dict(
        :bellman_ford => bellman_ford,
        :dijkstra_kary => dijkstra_kary,
        :dijkstra_quickheaps => dijkstra_quickheaps,
    )

    alg_func = get(dict_algs, alg_name, nothing)
    isa(alg_func, Nothing) && error("Function $(alg_nm) undefined.")


    for i in 1:m
        # TEMP: FIGURE OUT ISSUE WITH SharedArrays and shm_open() TO USE
        # vec_dists = Graphs.Parallel.dijkstra_shortest_paths(graph, i).dists
        vec_dists = alg_func(graph, i)
        
        # see https://docs.julialang.org/en/v1/devdocs/boundscheck/ for @inbounds macro info
        D[i, :] = vec_dists
        for j in 1:n
            @inbounds D_invs[i, j] = ifelse(D[i, j] > 0, 1.0/D[i, j], 0.0)
        end
    end

    return D, D_invs
end



function get_distance_matrices(
    graph::Union{SimpleDiGraph, SimpleGraph, Nothing};
    algorithm::Symbol = :auto,
    digits_round::Int64 = 10,
)::Tuple{Matrix{Float64}, Matrix{Float64}}
    
    # initialization
    m, n = size(graph)
    
    # initialize zeros
    D_invs = Float64.(zeros(size(graph)))
    D = Float64.(zeros(size(graph)))

    alg_func = select_algorithm(algorithm; n_vertices = m)

    for i in 1:m
        # TEMP: FIGURE OUT ISSUE WITH SharedArrays and shm_open() TO USE
        #vec_dists = Graphs.Parallel.dijkstra_shortest_paths(graph, i).dists
        vec_dists = alg_func(graph, i)
        
        # see https://docs.julialang.org/en/v1/devdocs/boundscheck/ for @inbounds macro info
        D[i, :] = vec_dists
        for j in 1:n
            @inbounds D_invs[i, j] = ifelse(D[i, j] > 0, 1.0/D[i, j], 0.0)
        end
    end

    # set D_shared to nothing and garbage collect
    #D = Float64.(Matrix(D_shared))
    #@everywhere D_shared = nothing
    #@everywhere GC.gc()
    
    return D, D_invs
end



"""
Read an edgelist from a .egl file. Returns a 

##  Constructs

```
read_egl(
    fp::String;
    delim::String = " ",
    edge_weight_default::Float64 = 1.0,
    infer_weights::Bool = true,
    skip_rows::Int64 = 0,
)::Union{Nothing, test1_FastGraph}
```

##  Function Arguments

- `fp`: file path to edgelist file. Can be .egl, .csv, or other.


##  Keyword Arguments

- `delim`: delimiter in edgelist to use to split rows into columns. Infers if
    nothing:
    * if the extension in `fp` is .csv, infers delim as ","
    * if the extension in `fp` is .egl, infers delim as " "
- `edge_weight_default`: default edge weight value to specify if an
    invalid edge weight is found
- `infer_weights`: bool denoting whether or not to infer weighting. 
    - If `true`, looks for inputs with 3 columns, assumed to be 

        i, j, w

        where `i` is the row, `j` is the columns, and `w` is the 
        specified weight. If only vectors of length 2 are specified, 
        assumes the graph is unweighted. 

    - If `false`, only looks for edges, ignoring potential weight 
        specification.
- `skip_lines`: number of lines to skip in input edge weights

"""
function read_egl(
    fp::String;
    delim::Union{String, Nothing} = nothing,
    edge_weight_default::Float64 = 1.0,
    infer_weights::Bool = true,
    skip_rows::Int64 = 0,
)::Union{Nothing, GraphWrapper}
    
    # check if file is undefined
    !ispath(fp) && (return nothing)

    # check delimiter
    dict_valid_exts = Dict(
        ".csv" => ",", 
        ".egl" => " ", 
        ".tsv" => "\t",
    )

    if isa(delim, Nothing)

        !any([endswith(fp, x) for x in keys(dict_valid_exts)]) && error("Unable to infer delimiter in read_egl: invalid extension.")

        # get delimiter
        delim = nothing
        for (key, val) in dict_valid_exts
            endswith(fp, key) && (delim = val)
        end
    end
    
    # read in edges
    mat_edges = split.(readlines(fp), delim)
    (skip_rows >= length(mat_edges)) && (return nothing)
    mat_edges = mat_edges[(skip_rows + 1):end]


    # filter out invalid edges and specify weights
    filter!(x -> (length(x) >= 2), mat_edges)
    (length(mat_edges) == 0) && (return nothing)
    
    weights = Float64.(ones(length(mat_edges)))
    weights *= edge_weight_default
    
    # separate edge weights and adjacency  
    for i in 1:length(mat_edges)

        if (length(mat_edges[i]) > 2) & infer_weights
            try_weight = tryparse(Float64, mat_edges[i][3])
            weights[i] = isa(try_weight, Nothing) ? weights[i] : try_weight
        end
        
        mat_edges[i] = mat_edges[i][1:2]
    end

    
    # get all vertices as a dictionary
    all_vertex_names = sort(string.(unique(vcat(mat_edges...))))
    all_vertices = collect(1:length(all_vertex_names))
    n_vertices = length(all_vertices)
    
    dict_vertex_to_index = Dict(zip(all_vertex_names, all_vertices))
    dict_index_to_vertex = Dict(zip(all_vertices, all_vertex_names))
    vertex_names = get.((dict_index_to_vertex, ), all_vertices, nothing)

    
    # get edges and sparse adjacency
    mat_edges_string = string.(permutedims(hcat(mat_edges...)))
    mat_edges_index = get.((dict_vertex_to_index, ), mat_edges_string, 0)
    
    # 
    adj = sparse(
        mat_edges_index[:, 1], 
        mat_edges_index[:, 2], 
        weights,
        n_vertices,
        n_vertices
    )
    
    
    graph_wrapper = GraphWrapper(adj; vertex_names = vertex_names)
    
    return graph_wrapper
end


function df_to_graph_wrapper(
    data_frame::DataFrame,
    field_vertex_i::Symbol,
    field_vertex_j::Symbol;
    edge_weight_default::Float64 = 1.0,
    field_weight::Union{Symbol, Nothing} = nothing,
    skip_rows::Int64 = 0,
)::Union{Nothing, GraphWrapper}

    ##  GET WEIGHTS IF SPECIFIED AND SPECIFY IN DATA FRAME

    # 
    field_weight_def = :weight
    weights = nothing

    if isa(field_weight, Symbol)
        if (String(field_weight) in names(data_frame))

            try_weight = try_parse_float.(data_frame[:, field_weight])
            
            # if successful, set weights and update weight field
            if !(nothing in try_weight) 
                weights = try_weight
                field_weight_def = field_weight
            end
        end
    end

    # if no weights are found, 
    if isa(weights, Nothing)
        weights = Float64.(ones(nrow(data_frame)))
        weights *= edge_weight_default
    end

    # build the data frame to work with and drop missing rows
    df = select(data_frame, [field_vertex_i, field_vertex_j])
    df[!, field_weight_def] = weights
    filter!(
        x -> (
            !ismissing(x[field_vertex_i])
            & !ismissing(x[field_vertex_j])
        ), 
        df
    )


    ##  NEXT, BUILD SPACE OF VERTICES, FILTER

    all_vertex_names = sort(string.(unique(vcat(
        df[:, field_vertex_i],
        df[:, field_vertex_j],    
    ))))
    all_vertices = collect(1:length(all_vertex_names))
    n_vertices = length(all_vertices)
    
    dict_vertex_to_index = Dict(zip(all_vertex_names, all_vertices))
    dict_index_to_vertex = Dict(zip(all_vertices, all_vertex_names))
    vertex_names = get.((dict_index_to_vertex, ), all_vertices, nothing)

    
    # get edges and sparse adjacency
    mat_edges_string = string.(Matrix(df[:, [field_vertex_i, field_vertex_j]]))
    mat_edges_index = get.((dict_vertex_to_index, ), mat_edges_string, 0)
    
    # 
    adj = sparse(
        mat_edges_index[:, 1],
        mat_edges_index[:, 2],
        df[:, field_weight_def],
        n_vertices,
        n_vertices,
    )

    graph_wrapper = GraphWrapper(adj; vertex_names = vertex_names)
    
    return graph_wrapper
end


"""
# GraphWrapper implementation wraps for graph operation

```
select_algorithm_from_benchmark_for_iteration(
    graph::GraphWrapper;
    kwargs...
)
```
"""
function select_algorithm_from_benchmark_for_iteration(
    graph::GraphWrapper;
    kwargs...
)
    out = select_algorithm_from_benchmark_for_iteration(
        graph.graph;
        kwargs...
    )

    return out
end