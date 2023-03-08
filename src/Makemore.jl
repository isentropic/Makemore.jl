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
        stoi[c] = i
    end
    itos = Dict{Integer,Char}()
    for (k, v) in stoi
        itos[v] = k
    end

    return Dataset(words, chars, max_word_length, stoi, itos)
end

encode(dataset::Dataset, word) = [dataset.stoi[c] for c in word]
decode(dataset::Dataset, indices) = String([dataset.itos[ix] for ix in indices])

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
    uniquechars_count = length(uniquechars)

    toshuffle && shuffle!(lines)

    testcut = floor(Int, length(lines) * testsplit)
    trainlines = lines[begin:end-testcut-1]
    testlines = lines[end-testcut:end]

    sortedchars = sort(collect(uniquechars))
    train_dataset = Dataset(trainlines, sortedchars, maxwordlength)
    test_dataset = Dataset(testlines, sortedchars, maxwordlength)

    return train_dataset, test_dataset
end

end
