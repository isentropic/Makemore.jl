module Makemore
using Flux
using Random
using StatsBase

# Special tokens need to start with 1 as it is used for array indexing
@enum SpecialToken starttoken = 1 endtoken = 2
const nreservedtokens = length(instances(SpecialToken))

include("data.jl")
include("bigram.jl")
include("mlp.jl")

Base.@kwdef struct Config
    blocksize::Integer # length of the input to predict next char (longer -> more information)
    vocabsize::Integer # number of unique letters + specialtokens
    nlayer::Integer = 4
    nembedding::Integer = 64
    nembedding2::Integer = 64
    nhead::Integer = 4
end

function loss(model, x, y)
    real = Flux.onehotbatch(y, 1:model.config.vocabsize, 1)
    Flux.Losses.logitbinarycrossentropy(model(x), real)
end

function loss(pred, y)
    real = Flux.onehotbatch(y, 1:28, 1)
    Flux.Losses.logitbinarycrossentropy(pred, real)
end


function generate(model, indices, maxnewtokens; temperature=1.0)
    if length(indices) < model.blocksize
        indices = vcat(fill(Int(starttoken), model.blocksize - length(indices) + 1), indices)
    end
    for _ in 1:maxnewtokens
        logits = model(indices[end-model.blocksize:end])[:, end] # only last is needed for bigram
        if ndims(logits) == 3
            logits = logits[:, :, 1]
        end
        probs = Flux.softmax(logits, dims=1)
        nextletter = sample(Weights(probs))

        indices = vcat(indices, nextletter)
    end

    return indices
end


function getsamples(model, dataset, num=10)
    samples = []
    for _ in 1:num
        start = [Int(starttoken),]
        indices = generate(model, start, model.config.vocabsize * 2)

        push!(samples, decode(dataset, indices))
    end
    return samples
end

end
