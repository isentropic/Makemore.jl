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
    return CausalSelfAttention(cattn,
        cproj,
        bias,
        config.nhead,
        config.nembedding)
end

function (m::CausalSelfAttention)(x)
    @assert ndims(x) == 3
    c, t, b = size(x) # emb, time, batch

    unsplit = m.cattn(x)
    q, k, v = Flux.chunk(unsplit, 3, dims=1)

    headsize = c ÷ m.nhead
    q = reshape(q, (m.nhead, headsize, t, b))
    k = reshape(k, (m.nhead, headsize, t, b))
    v = reshape(v, (m.nhead, headsize, t, b))
    # nhead, headsize, t, b
    #
    q = permutedims(q, [3, 2, 1, 4])
    k = permutedims(k, [3, 2, 1, 4])
    v = permutedims(v, [3, 2, 1, 4])
    # t, headsize, nhead, b

    att = q ⊠ permutedims(k, [2, 1, 3, 4])
    att = att ./ sqrt(size(k)[2])
    # t, t, nhead, b

    att = att .+ triu(fill(-Inf32, t, t), 1)
    att = Flux.softmax(att, dims=2)

    # (t, t, nhead, b) x (t, headsize, nhead, b) -> (t, headsize, nhead, b)
    y = att ⊠ v

    # (nhead, headsize, t, b)
    y = permutedims(y, [3, 2, 1, 4])
    y = reshape(y, (c, t, b))

    return m.cproj(y)
end

Base.@kwdef struct Block
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

    @show mlpf
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

    positions = Flux.unsqueeze(1:t, dims=2)

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

function generate(model::Transformer, indices, maxnewtokens; temperature=1.0)
    maxnewtokens = model.blocksize
    # In BoW maxnewtokens does not exceed blocksize for simplicity of the of
    # implementation. BoW scales cheaply with increasing blocksize as it just
    # averages out all the previously encountered tokens
    for _ in 1:maxnewtokens
        indices = Flux.unsqueeze(indices, dims=2)
        logits = model(indices)
        probs = Flux.softmax(logits[:, end, 1])
        nextletter = sample(Weights(probs))

        indices = vcat(indices[:, 1], nextletter)
    end

    return indices
end
