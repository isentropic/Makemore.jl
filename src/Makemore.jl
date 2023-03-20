module Makemore
using Flux
using Random

struct Dataset
    words::Vector{String}
    chars::Vector{Char}
    max_word_length::Integer
    stoi::Dict{Char,Integer}
    itos::Dict{Integer,Char}
end
Base.length(dataset::Dataset) = length(dataset.words)

function Dataset(words, chars, max_word_length)
    stoi = Dict{Char,Integer}()
    for (i, c) in enumerate(chars)
        stoi[c] = i + 1
    end
    itos = Dict{Integer,Char}()
    for (k, v) in stoi
        itos[v] = k
    end

    return Dataset(words, chars, max_word_length, stoi, itos)
end

encode(dataset::Dataset, word) = [dataset.stoi[c] for c in word]
decode(dataset::Dataset, indices) = String([dataset.itos[ix] for ix in indices])

function Base.getindex(dataset::Dataset, index)
    word = dataset.words[index]
    encodedword = encode(dataset, word)

    x = ones(Int, dataset.max_word_length + 1)
    y = ones(Int, dataset.max_word_length + 1)

    # Each word is encoded into same length arrays
    # with zeros for start and end locations
    x[2:1+length(encodedword)] .= encodedword

    y[1:length(encodedword)] .= encodedword
    y[length(encodedword)+1:end] .= -1

    # Prediction (y) is 1 index shifted
    return x, y
end

function loaddatasets(filename, testsplit=0.1, toshuffle=true)
    lines = map(strip, filename |> open |> readlines)
    uniquechars = Set{Char}()
    for line in lines
        for c in line
            push!(uniquechars, c)
        end
    end

    maxwordlength = 0
    for word in lines
        maxwordlength = max(maxwordlength, length(word))
    end

    toshuffle && shuffle!(lines)

    testcut = floor(Int, length(lines) * testsplit)
    trainlines = lines[begin:end-testcut-1]
    testlines = lines[end-testcut:end]

    sortedchars = sort(collect(uniquechars))
    train_dataset = Dataset(trainlines, sortedchars, maxwordlength)
    test_dataset = Dataset(testlines, sortedchars, maxwordlength)

    return train_dataset, test_dataset
end

Base.@kwdef struct Config
    blocksize::Integer # length of the input to predict next char (longer -> more information)
    vocabsize::Integer # number of unique letters + 1
    nlayer::Integer = 4
    nembedding::Integer = 64
    nembedding2::Integer = 64
    nhead::Integer = 4
end

include("bigram.jl")
end
