Base.@kwdef struct Bigram
    config::Config
    n::Int64
    logits::Matrix{Float64}
end

# Declare logits to be trainable for Flux
# This macro lets Flux track "what to differentiate against"
Flux.@functor Bigram (logits,)

function Bigram(config)
    n = config.vocabsize
    return Bigram(config, n, zeros(n, n))
end

# Forward pass, basically just a lookup
(m::Bigram)(index) = m.logits[:, index]

function generate(model::Bigram, indices, maxnewtokens; temperature=1.0)
    # @TODO support temperature sampling
    if length(indices) < model.blocksize
        indices = vcat(fill(Int(starttoken), model.blocksize - length(indices) + 1), indices)
    end
    for _ in 1:maxnewtokens
        logits = model(indices[end-model.blocksize:end])[:, end]
        # only last is needed for bigram
        if ndims(logits) == 3
            logits = logits[:, :, 1]
        end
        probs = Flux.softmax(logits, dims=1)
        nextletter = sample(Weights(probs))

        indices = vcat(indices, nextletter)
    end

    return indices
end
