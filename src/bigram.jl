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

# Forward pass, basically a lookup
(m::Bigram)(index) = m.logits[:, index]

function loss(m::Bigram, real, predicted)
    return Flux.Losses.logitcrossentropy(predicted, real)
end

function generate(model, indices, maxnewtokens; tempertaure=1.0)
    for _ in 1:maxnewtokens
        logits = model(indices[end]) # only last is needed for bigram
        probs = Flux.softmax(logits, dims=1)
        nextletter = sample(Weights(probs))

        indices = vcat(indices, nextletter)
    end

    return indices
end


function getsamples(model, dataset, num=10)
    samples = []
    for _ in 1:num
        start = [Int(starttoken), ]
        indices = generate(model, start, model.config.vocabsize)

        push!(samples, decode(dataset, indices))
    end
    return samples
end
