begin
    using CSV
    using DataFrames
    using Statistics
    using Pkg; Pkg.add("Optim"); Pkg.add("Tables")
    using Optim
    using Tables
    using LinearAlgebra
    using Plots
end

begin
    function methylator(state::Vector, variables::Vector)
        possibilities = []
        for i in eachindex(state)
            if state[i] == 0
                new_state = copy(state)
                new_state[i] = variables[i]
                push!(possibilities, new_state)
            end
        end
        return possibilities
    end
    function demethylator(state::Vector)
        possibilities = []
        for i in eachindex(state)
            if state[i] != 0
                new_state = copy(state)
                new_state[i] = 0
                push!(possibilities, new_state)
            end
        end

        return possibilities
    end
    function to_ix(state::Vector)::Int
        u = BitVector(state .!= 0)
        return parse(Int, join(Int.(u)), base = 2) + 1
    end
    
    """
        transition_matrix(variables)

    Compute the transition matrix for a given set of variables.

    # Arguments
    - `variables`: An array of variables representing the transition rates.

    # Returns
    - `M`: The transition matrix.

    # Examples
    ```julia
    transition_matrix([1,1])
    ```
    """
    function transition_matrix(variables)
        n = length(variables)
        M_track = Array{Any}(nothing, 2^n, 2^n)
        function assign_out!(state, variables)
            next = methylator(state, variables)
            if length(next) != 0
                for s in next
                    i = to_ix(state)
                    v = -variables[argmax(s - state)]
                    if typeof(M_track[i,i]) == Nothing
                        M_track[i,i] = Set([(state, s, v)])
                    else
                        M_track[i,i] = M_track[i,i] ∪ Set([(state, s, v)])
                    end
                    if sum(s - state) != 0
                        assign_out!(s, variables)
                    end
                end
            end
        end
        function assign_in!(state, variables)
            next = demethylator(state)
            if length(next) != 0
                for s in next
                    i = to_ix(state)
                    j = to_ix(s)
                    v = variables[argmax(state - s)]
                    if typeof(M_track[i,j]) == Nothing
                        M_track[i,j] = Set([(state, s, v)])
                    else
                        M_track[i,j] = M_track[i,j] ∪ Set([(state, s, v)])
                    end
                    if sum(state - s) != 0
                        assign_in!(s, variables)
                    end
                end
            end
        end
        state = zeros(n)
        assign_out!(state, variables)
        state = copy(variables)
        assign_in!(state, variables)
        
        #Construct matrix
        M = zeros(2^n, 2^n)
        for i in 1:2^n
            for j in 1:2^n
                if typeof(M_track[i,j]) == Nothing
                    M[i,j] = 0
                else
                    M[i,j] = sum([x[3] for x in M_track[i,j]])
                end
            end
        end
        return M
    end
end

begin
    # Load the data
    ix_to_meth = Dict(
        1 => "sm",
        2 => "sm295",
        3  =>"sm302",
        5 => "sm309",
        4 => "sm295&302",
        6 => "sm295&309",
        7 => "sm302&309",
        8 => "sm295&302&309"
    )
    types = Dict(s => Float64 for s in values(ix_to_meth))
    columns = collect(values(ix_to_meth))
    g_columns = copy(columns)
    push!(g_columns, "Exp")
    data = CSV.read("data.csv", DataFrame, delim = "\t", types = types)
    data = data[:, g_columns]
    data = combine(groupby(data, :Exp), names(data[:, columns]) .=> mean, renamecols = false)
    totals = sum(Matrix(data[:,columns]), dims=2)
    data[:, columns] = data[:, columns] ./ totals
    A_values = data[data.Exp .== "A", columns]
    A_values = Dict(pairs(eachcol(A_values)))
    A = [A_values[Symbol(ix_to_meth[i])][1] for i ∈ 1:length(A_values)]
    A
end

begin
    k = rand(3)
    @info "Ansatz: $k"
    function objective(variables)
        variables = abs.(variables) #Must be positive
        n = length(variables)
        B = zeros(2^n)
        B[1] = 1
        M = transition_matrix(variables)
        P = exp(M)*B
        residue = sum((P - A).^2)
        return residue
    end
    result = optimize(objective, k)
    k_optim = abs.(result.minimizer)
    @info "Found: $k_optim"
end

@time begin
    n = length(k_optim)
    M_optim = transition_matrix(k_optim)
    λ, Q = eigen(M_optim) # M = QΛQ^(-1)
    Λ = Diagonal(λ)
    expΛ = exp(Λ)
    iQ = inv(Q)
    B = zeros(2^n)
    B[1] = 1
    t = LinRange(0, 15, 100)
    P = [Q*((expΛ)^t_i)*iQ*B for t_i in t]
    P = mapreduce(permutedims, vcat, P)
end

begin
    m = size(P)[2]
    labels = permutedims([bitstring(UInt8(i))[end-2:end] for i ∈ 0:m-1])
    colors = collect(palette(:tab10))[1:m]
    plot(t, P,
        label = labels,
        xlabel = "Time (min)",
        ylabel = "Occupation Probability",
        legend = :outerright,
        size = (900, 600),
        linewidth = 2,
        palette = colors
    )
    scatter!([1], transpose(A), labels = nothing, marker = :x, palette = colors, markersize = 5, markerstrokewidth = 2)
    savefig("transition_rate_find.png")
end