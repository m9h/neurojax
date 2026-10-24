#!/usr/bin/env julia
# SOTA surrogate-data oracle (TimeseriesSurrogates.jl) for the WAND irreversibility
# null test.  Reads a multichannel embedding (T, k) from a .npy file, generates an
# ensemble of per-channel surrogates, and writes (n_surr, T, k) to a .npy file.
#
# Per-channel IAAFT (the default) preserves each channel's power spectrum AND
# amplitude distribution exactly and is time-reversible — the correct reversible,
# spectrum/marginal-matched null for an irreversibility test (unlike a time-shuffle,
# which whitens the spectrum, or MVPR, which preserves the directed cross-spectrum).
#
#   julia --project=. gen_surrogates.jl IN.npy OUT.npy N_SURR METHOD SEED
#   METHOD ∈ {iaaft, ft, wls}   (wls = wavelet/Keylock, for non-stationary data)

using TimeseriesSurrogates
using NPZ
using Random

function method_for(name::AbstractString)
    name == "iaaft" && return IAAFT()
    name == "ft"    && return RandomFourier(true)      # phase randomization (FT)
    name == "wls"   && return WLS(IAAFT(); rescale=true)  # wavelet (Keylock), nonstationary
    error("unknown surrogate method: $name")
end

function main()
    infile  = ARGS[1]
    outfile = ARGS[2]
    nsurr   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
    method  = length(ARGS) >= 4 ? ARGS[4] : "iaaft"
    seed    = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0

    X = npzread(infile)                       # (T, k) Float
    T, k = size(X)
    rng = Xoshiro(seed)
    m = method_for(method)

    out = Array{Float64}(undef, nsurr, T, k)
    # one surrogate generator per channel (reuses plan/precomputation across draws)
    gens = [surrogenerator(Float64.(@view X[:, j]), m, rng) for j in 1:k]
    for s in 1:nsurr
        for j in 1:k
            out[s, :, j] = gens[j]()
        end
    end
    npzwrite(outfile, out)
    println("wrote $(size(out)) surrogates [$method] -> $outfile")
end

main()
