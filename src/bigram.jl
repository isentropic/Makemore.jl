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
