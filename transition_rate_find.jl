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
end

@time begin
    k = [1.0,2.0,4.0,0.1,0.2,1] #k1,k2,k3,k2',k3',α
    @info "Ansatz: $k"
    function objective(variables, R, α = 1)
        variables = variables.^2 #Must be positive
        B = zeros(2^nMet)
        B[1] = 1
        M = transition_matrix(variables)
        P = exp(α*α*M)*B
        residue = sum((P - R).^2)
        return residue
    end
    k_results = Dict{String1, Vector{Float64}}()
    Chisq = Dict{String, Float64}() 
    for col ∈ keys(data_groups)
        if col != "A"
            k_results[col] = optimize(x -> objective(x[1:end-1], data_groups[col], x[end]), k).minimizer.^2
            Chisq[col] = objective(k_results[col], data_groups[col], k_results[col][end])
        else
            k_results[col] = optimize(x -> objective(x, data_groups[col]), k[1:end-1]).minimizer.^2
            Chisq[col] = objective(k_results[col], data_groups[col])
            push!(k_results[col], 1) #α of B is kept as reference
        end
    end
end

@time begin
    t = LinRange(0, 2, 100)
    function evol(k, t, α)
        M_optim = transition_matrix(k)
        λ, Q = eigen(M_optim) # M = QΛQ^(-1)
        Λ = Diagonal(λ)
        expΛ = exp(Λ)
        iQ = inv(Q)
        B = zeros(2^nMet)
        B[1] = 1
        P = [Q*((expΛ)^(t_i*α))*iQ*B for t_i in t]
        P = mapreduce(permutedims, vcat, P)
        return P
    end
    P = Dict(col => evol(k_results[col][1:end-1], t, k_results[col][end]) for col ∈ keys(k_results))
end

begin
    for col ∈ keys(P)
        G = P[col]
        m = size(G)[2]
        labels = permutedims([bitstring(UInt8(i))[end-2:end] for i ∈ 0:m-1])
        colors = collect(palette(:tab10))[1:m]
        plot(t, G,
            label = labels,
            xlabel = "Time (min)",
            ylabel = "Occupation Probability",
            legend = :outerright,
            size = (900, 600),
            linewidth = 2,
            palette = colors,
            title = "Χ² $(round(Chisq[col], digits = 3))"
        )
        scatter!([1], transpose(data_groups[col]), labels = nothing, marker = :x, palette = colors, markersize = 5, markerstrokewidth = 2)
        savefig("transition_rate_find_$col.png")
    end
end

begin
    k_pretty = copy(k_results)
    labels = ["k1", "k2", "k3", "k2'", "k3'", "α"]
    pretty_table(k_pretty, row_labels = labels, max_num_of_rows = -1)
end