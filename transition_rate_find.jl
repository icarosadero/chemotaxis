begin
    using Pkg; 
    Pkg.add(["Optim", "Tables", "CSV", "DataFrames", "Statistics", "Plots", "LinearAlgebra", "PrettyTables"])
end

begin
    using CSV
    using DataFrames
    using Statistics
    using Optim
    using Tables
    using LinearAlgebra
    using Plots
    using PrettyTables
end

begin
    nMet = 3
    nStates = 1 << nMet
end

begin
    """
        transition_matrix(kseq)

    Constructs the transition matrix for a given sequence of transition rates.

    # Arguments
    - `kseq`: A sequence of transition rates. The length of `kseq` must be even.

    # Returns
    - `T`: The transition matrix.

    """
    function transition_matrix(kseq)
        T = zeros(nStates, nStates)
        #Scan the initial states
        for s0 ∈ 0:nStates-1
            #Define if each site is metilable or not. For example, for s0 = 0, ~s0 = 7 - 0 = 7 = 111
            metilable = digits((nStates - 1) - s0, base = 2, pad = nMet)
            @debug "s0: $s0, metilable: $metilable"
            for i ∈ 1:nMet
                #Bit in the state of the metilable site
                site = metilable[i] << (i - 1)
                if site != 0
                    s1 = s0 | site
                    @debug "s0: $s0, s1: $s1, site: $site"
                    k = (site & (s0 << 1 + 1)) > 0 ? kseq[i] : kseq[i+nMet-1]
                    T[s0+1, s0+1] -= k
                    T[s1+1, s0+1] += k
                end
            end
        end
        return T
    end
end

begin
    # The columns must be in the correct order (0 to 111)
    columns = [
        "sm",
        "sm309",
        "sm302",
        "sm302&309",
        "sm295",
        "sm295&309",
        "sm295&302",
        "sm295&302&309"
    ]
    
    types = Dict(s => Float64 for s ∈ columns)
    g_columns = copy(columns)
    push!(g_columns, "Exp")
    data = CSV.read("data.csv", DataFrame, delim = "\t", types = types, comment = "#")
    data = data[:, g_columns]
    data = combine(groupby(data, :Exp), names(data[:, columns]) .=> mean, renamecols = false)
    totals = sum(Matrix(data[:,columns]), dims=2)
    data[:, columns] = data[:, columns] ./ totals
    # Iterate over each group and convert to dictionary
    data_groups = Dict()
    for exp in unique(data.Exp)
        row = data[data.Exp .== exp, :]
        X = [data[data.Exp .== exp, col][1] for col ∈ columns]
        data_groups[exp] = X
    end
    ordered_column_names = sort(collect(keys(data_groups)))
end

@time begin
    k = [1.0,2.0,4.0,0.1,0.2,1.0,1.0] #k1, k2, k3, k2', k3', tA, tB
    @info "Ansatz: $k"
    function objective(K, R, t = 1)
        B = zeros(2^nMet)
        B[1] = 1
        M = transition_matrix(K)
        P = exp(t*M)*B
        residue = sum((P - R).^2)
        return residue
    end
    function ΣObjective(variables, column_order)
        variables = variables.^2 #Force positives
        t_shifts = variables[end-1:end]
        push!(t_shifts, 1) #Last one is kept constant
        K = variables[1:end-2]
        return sum([objective(K, data_groups[col], t_shifts[i]) for (i, col) ∈ enumerate(column_order)])
    end
    function optimize_given_permutation(column_order)
        optim = optimize(x -> ΣObjective(x, column_order), k)
        Chisq = ΣObjective(optim.minimizer, column_order)
        return Chisq, optim
    end

    N = size(ordered_column_names, 1)
    optimizations = [optimize_given_permutation(circshift(ordered_column_names, k)) for k ∈ 0:N-1]    
    best = argmin([optim[1] for optim ∈ optimizations])
    column_order = circshift(ordered_column_names, best)

    optim = optimizations[best][2]
    Chisq = optimizations[best][1]
    variables_best = optim.minimizer.^2
    t_best = optim.minimizer[end-1:end]
    push!(t_best, 1)
    t_best = Dict(zip(column_order, t_best))
    k_best = variables_best[1:end-2]

    #Pretty printing
    v_pretty = copy(variables_best)
    push!(v_pretty, 1)
    push!(v_pretty, Chisq)
    labels = [["k1", "k2", "k3", "k2'", "k3'"]; column_order; ["Chisq"]]
    pretty_table(v_pretty, row_labels = labels, max_num_of_rows = -1)
end

@time begin
    t = LinRange(0, 2, 100)
    function evol(k, t)
        M_optim = transition_matrix(k)
        λ, Q = eigen(M_optim) # M = QΛQ^(-1)
        Λ = Diagonal(λ)
        expΛ = exp(Λ)
        iQ = inv(Q)
        B = zeros(2^nMet)
        B[1] = 1
        P = [Q*((expΛ)^(t_i))*iQ*B for t_i in t]
        P = mapreduce(permutedims, vcat, P)
        return P
    end
    P = evol(k_best, t);
end

begin
    for col ∈ ordered_column_names
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
            palette = colors,
            title = "Measurements $col | Χ²: $(round(Chisq, digits = 3))"
        )
        scatter!([t_best[col]], transpose(data_groups[col]), labels = nothing, marker = :x, palette = colors, markersize = 5, markerstrokewidth = 2)
        savefig("transition_rate_find_$col.png")
    end
end
