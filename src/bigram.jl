Base.@kwdef struct Bigram
    config::Config
    n::Int64
    logits::Matrix{Float64}
end

function Bigram(config)
    n = config.vocabsize
    return Bigram(config, n, zeros(n, n))
end

# Forward pass
(m::Bigram)(index) = m.logits[index]

function loss(m::Bigram, real, predicted)
    return Flux.Losses.logitcrossentropy(predicted, real)
end
