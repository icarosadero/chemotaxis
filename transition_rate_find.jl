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
    """
        transition_matrix(kseq, η=1.0)

    Constructs the transition matrix for a given set of transition rates.

    # Arguments
    - `kseq`: An array of transition rates for each state.
    - `η`: A scaling factor for the random transition rates. Default is 1.0.

    # Returns
    - `T`: The transition matrix.
    """
    function transition_matrix(kseq, η=1.0)
        nMet = length(kseq)
        nStates = 1 << nMet
        T = zeros(nStates, nStates)
        for s0 ∈ 0:nStates-1
            #Union of all accesbile states. For example, for s0 = 0, ~s0 = 7 - 0 = 7 = 111
            accessible = digits((nStates - 1) - s0, base = 2, pad = nMet)
            @debug "s0: $s0, accessible: $accessible"
            for i ∈ 1:nMet
                site = accessible[i] << (i - 1)
                if site != 0
                    s1 = s0 | site
                    @debug "s0: $s0, s1: $s1, site: $site"
                    k = (s1 == s0 << 1 + 1) ? kseq[i] : η*kseq[i]
                    T[s0+1, s0+1] -= k
                    T[s1+1, s0+1] += k
                end
            end
        end
        return T
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

@time begin
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