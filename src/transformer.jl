Base.@kwdef mutable struct CausalSelfAttention{T<:Integer}
    cattn::Dense
    cproj::Dense
    bias
    nhead::T
    nembd::T
end
Flux.@functor CausalSelfAttention (cattn, cproj)

function CausalSelfAttention(config::Config)
    @assert config.nembedding % config.nhead == 0

    cattn = Dense(config.nembedding, 3 * config.nembedding)
    cproj = Dense(config.nembedding, config.nembedding)

    bias = reshape(tril(ones(Float32, config.blocksize, config.blocksize)), (1, config.blocksize, config.blocksize, 1))
    return CausalSelfAttention(cattn, cproj, bias, config.nhead, config.nembedding)
end

function (m::CausalSelfAttention)(x)
    @assert ndims(x) == 3
    t, c, b = size(x)
    q, k, v = m.cattn(x)
end

Base.@kwdef struct Block{T<:Integer}
    ln1::LayerNorm
    attn::CausalSelfAttention
    ln2::LayerNorm
    mlpf::Chain
    config::Config
end
Flux.@functor Block (attn, mlpf)

function Block(config::Config)
    ln1 = LayerNorm(config.nembedding)
    attn = CausalSelfAttention(config)
    ln2 = LayerNorm(config.nembedding)
    # Let's implement mlp as a simple Flux chain with builtin activation
    mlpf = Chain(
        Dense(config.nembedding, 4 * config.nembedding, gelu),
        Dense(4 * config.nembedding, config.nembedding)
    )
    return Block(ln1, attn, ln2, mlpf, config)
end

function (m::Block)(x)
    x = x + m.attn(m.ln1(x))
    x = x + m.mlpf(m.ln2(x))
    return x
end

Base.@kwdef struct Transformer{T<:Integer}
    blocksize::T
    wte::Embedding
    wpe::Embedding
    h::Vector{Block}
    lnf::LayerNorm
    lmhead::Dense
    config::Config
end
Flux.@functor Transformer (wte, wpe, h, lmhead,)

function Transformer(config::Config)
    wte = Embedding(config.vocabsize, config.nembedding)
    wpe = Embedding(config.blocksize, config.nembedding)
    h = [Block(config) for _ in 1:config.nlayer]
    lnf = LayerNorm(config.nembedding)
    lmhead = Dense(config.nembedding, config.vocabsize, bias=false)
    return Transformer(config.blocksize, wte, wpe, h, lnf, lmhead, config)
end

function (m::Transformer)(index)
    t, b = size(index)
    @assert t <= m.blocksize

    positions = Flux.unsqueeze(1:t, 2)

    tokemb = m.wte(index)
    posemb = m.wpe(positions)

    x = tokemb .+ posemb

    for block in m.h
        x = block(x)
    end
    x = m.lnf(x)

    return m.lmhead(x)
end
# Flux.@functor RNN (wte, cell, lmhead, start) including start causes Flux
# error, help needed.
# Luckily including start as a trainable param is not necessary
