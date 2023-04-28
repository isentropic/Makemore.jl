Base.@kwdef struct MLP
    config::Config
    blocksize::Int64
    vocabsize::Int64
    embedding::Embedding
    mlp::Chain
end

Flux.@functor MLP (embedding, mlp)

function MLP(config)
    emb = Embedding(config.vocabsize, config.nembedding)
    mlp = Chain(
        Dense(config.blocksize * config.nembedding, config.nembedding2, tanh),
        Dense(config.nembedding2, config.vocabsize)
    )
    return MLP(config, config.blocksize, config.vocabsize, emb, mlp)
end

function prefixpadshift(prefix, y, units)
    if ndims(y) == 3
        zeropadding = repeat(prefix, 1, units)
        zeropadding = repeat(zeropadding, 1, 1, size(y)[end])
    else
        zeropadding = repeat(prefix, 1, units)
    end
    shifted = circshift(y, (0, units))[:, (units+1):end, :]
    return hcat(zeropadding, shifted)
end

function (m::MLP)(index)
    emb = m.embedding(index)
    prefix = m.embedding(Int(starttoken))
    x = vcat([prefixpadshift(prefix, emb, i) for i in 0:m.blocksize-1]...)
    logits = m.mlp(x)
    return logits
end
