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
        stoi[c] = i + nreservedtokens
    end
    itos = Dict{Integer,Char}()
    for (k, v) in stoi
        itos[v] = k
    end

    return Dataset(words, chars, max_word_length, stoi, itos)
end

encode(dataset::Dataset, word) = [dataset.stoi[c] for c in word]

function decode(dataset::Dataset, indices)
    # Decode while ignoring SpecialToken
    output = Char[]
    for i in indices
        if i in keys(dataset.itos)
            push!(output, dataset.itos[i])
        end
    end
    return String(output)
end

function Base.getindex(dataset::Dataset, index)
    word = dataset.words[index]
    encodedword = encode(dataset, word)

    x = fill(Int(endtoken), dataset.max_word_length + nreservedtokens)
    y = fill(Int(endtoken), dataset.max_word_length + nreservedtokens)

    # Each word is encoded into same length arrays
    # with zeros for start and end locations
    x[1] = Int(starttoken)
    x[nreservedtokens:1+length(encodedword)] .= encodedword

    y[1:length(encodedword)] .= encodedword
    y[length(encodedword)+1:end] .= Int(endtoken)

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

function get_dataloader(dataset)
    X = []
    Y = []

    for i in eachindex(dataset.words)
        x, y = dataset[i]
        push!(X, x)
        push!(Y, y)
    end

    X = hcat(X...)
    Y = hcat(Y...)
    train_loader = Flux.DataLoader((X, Y), batchsize=32)
    return train_loader
end
