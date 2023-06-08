# Bag of words language models
import LinearAlgebra: LowerTriangular, UpperTriangular, triu, tril
Base.@kwdef mutable struct CausalBoW
    blocksize
    bias::Matrix{Float32}  # ? cannot understand why is this needed
end

function CausalBoW(config::Config)
    bias = ones(Float32, config.blocksize, config.blocksize)
    bias = LowerTriangular(bias)
    return CausalBoW(config.blocksize, bias)
end

function (m::CausalBoW)(x)
    @assert ndims(x) == 3
    t, c, b = size(x)
    # ? cannot understand why bias even needed
    # Probably bias needs to accumlate some value,
    # but I could not figure out what exactly
    att = zeros(Float32, t, t, b)
    att = att .+ triu(fill(-Inf32, t, t), 1)
    att = Flux.softmax(att, dims=2)
    y = Flux.batched_mul(att, x)

    return y
end

Base.@kwdef struct MLPF
    cfc::Dense
    cproj::Dense
end
Flux.@functor MLPF

function (m::MLPF)(x)
    return m.cproj(tanh.(m.cfc(x)))
end

Base.@kwdef struct BoWBlock
    cbow::CausalBoW
    mlpf::MLPF
end
Flux.@functor BoWBlock (mlpf,)

function BoWBlock(config::Config)
    mlpf = MLPF(
        Dense(config.nembedding, config.nembedding2),
        Dense(config.nembedding2, config.nembedding)
    )
    causal = CausalBoW(config)
    return BoWBlock(causal, mlpf)
end

function (m::BoWBlock)(x)
    x = x + m.cbow(x)
    x = x + m.mlpf(x)
    return x
end

Base.@kwdef struct BoW{T<:Integer}
    blocksize::T
    vocabsize::T
    wte::Embedding
    wpe::Embedding
    contextblock::BoWBlock
    lmhead::Dense
    config::Config
end
Flux.@functor BoW (wte, wpe, contextblock, lmhead)

function BoW(config::Config)
    return BoW(
        config.blocksize,
        config.vocabsize,
        Embedding(config.vocabsize, config.nembedding),
        Embedding(config.blocksize, config.nembedding),
        BoWBlock(config),
        Dense(config.nembedding, config.vocabsize),
        config)

end

function (m::BoW)(index)
    # 1. Encode previous blocksize tokens with embedding
    # 2. Add positional embedding
    # 3. Average all and feed into BoWBlock
    t, b = size(index)
    @assert t <= m.blocksize
    positions = Flux.unsqueeze(1:t, 2)

    tokemb = m.wte(index)
    posemb = m.wpe(positions)

    x = tokemb .+ posemb
    x = m.contextblock(x)
    logits = m.lmhead(x)

    return logits
end

function generate(model::BoW, indices, maxnewtokens; temperature=1.0)
    maxnewtokens = model.blocksize
    # In BoW maxnewtokens does not exceed blocksize for simplicity of the of
    # implementation. BoW scales cheaply with increasing blocksize as it just
    # averages out all the previously encountered tokens
    for _ in 1:maxnewtokens
        indices = Flux.unsqueeze(indices, 2)
        logits = model(indices)
        probs = Flux.softmax(logits[:, end, 1])
        nextletter = sample(Weights(probs))

        indices = vcat(indices[:, 1], nextletter)
    end

    return indices
end
