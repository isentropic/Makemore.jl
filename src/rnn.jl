Base.@kwdef struct RNNCell
    config::Config
    xh2h::Dense
end


function RNNCell(config)
    xh2h = Dense(config.nembedding + config.nembedding2, config.nembedding2)
    return RNNCell(config, xh2h)
end

function (m::RNNCell)(x, h)
    xh = hcat([x, h])
    ht = tanh(m.xh2h(xh))
    return ht
end

Base.@kwdef struct GRUCell
    config::Config
    xh2h::Dense
end

Base.@kwdef struct RNN{T<:Integer}
    blocksize::T
    vocabsize::T
    start::Any
    wte::Embedding
    cell::Union{RNNCell,GRUCell}
    lmhead::Dense
end

function RNN(config)
    emb = Embedding(config.vocabsize, config.nembedding)
    return RNN(
        config.blocksize,
        config.vocabsize,
        zeros(1, config.nembedding2)
        emb,
        RNNCell(config),
        Dense(config.nembedding2, config.vocabsize)
    )
end

Flux.@functor RNNCell (xh2h,)
Flux.@functor RNN (cell,)

function (m::RNN)(index)
    emb = m.embedding(index)
    t, b = size(index)
end
