module Makemore

using Flux
using Random
using StatsBase

# Special tokens need to start with 1 as it is used for array indexing
@enum SpecialToken starttoken = 1 endtoken = 2
const nreservedtokens = length(instances(SpecialToken))

include("data.jl")

# Models
Base.@kwdef struct Config
    blocksize::Integer # length of the input to predict next char (longer -> more information)
    vocabsize::Integer # number of unique letters + specialtokens
    nlayer::Integer = 4
    nembedding::Integer = 64
    nembedding2::Integer = 64
    nhead::Integer = 4
end

include("bigram.jl")
include("mlp.jl")
include("rnn.jl")

function loss(model, x, y)
    real = Flux.onehotbatch(y, 1:model.config.vocabsize, 1)
    return Flux.Losses.logitbinarycrossentropy(model(x), real)
end

function loss(pred, y)
    real = Flux.onehotbatch(y, 1:size(pred)[1], 1)
    return Flux.Losses.logitbinarycrossentropy(pred, real)
end

function evaluate(model, dataset, maxbatches=nothing)
    dataloader = get_dataloader(dataset)
    losses = Float32[]
    i = 0
    for (x, y) in dataloader
        ŷ = model(x)
        ℓ = loss(ŷ, y)
        push!(losses, ℓ)
        if maxbatches !== nothing && i > maxbatches
            break
        end
        i += 1
    end
    return mean(losses)
end

function train_model!(model, train_dataset, test_dataset, maxepocs=100)
    log = []
    opt_state = Flux.setup(Flux.Adam(), model)
    train_loader = get_dataloader(train_dataset)

    for epoch in 1:maxepocs
        losses = Float32[]
        for (x, y) in train_loader
            val, grads = Flux.withgradient(model) do m
                # Any code inside here is differentiated.
                # Evaluation of the model and loss must be inside!
                result = m(x)
                loss(result, y)
            end

            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(losses, val)
            Flux.update!(opt_state, model, grads[1])
        end

        trainloss = mean(losses)
        testloss = evaluate(model, test_dataset,)
        println("epoch: $epoch, trainloss: $trainloss, testloss: $testloss")
        # # Compute some accuracy, and save details as a NamedTuple
        # acc = my_accuracy(model, train_set)
        push!(log, (; trainloss, testloss))

        if length(log) >= 3 && issorted(l.testloss for l in log[end-2:end])
            println("Early stopping: testloss is increasing")
            break
        end

    end


    return log
end

function generate(model, indices, maxnewtokens; temperature=1.0)
    # @TODO support temperature sampling
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

function getsamples(model, traindataset, testdataset, num=10)
    samples = []
    existingwords = Set{String}()
    for data in (traindataset, testdataset)
        for word in data.words
            push!(existingwords, word)
        end
    end

    generated = 0
    maxiters = 1e6
    iter = 0
    while generated < num
        start = [Int(starttoken),]
        indices = generate(model, start, model.config.vocabsize * 2)
        newword = decode(traindataset, indices)
        iter += 1
        if iter > maxiters
            @error "Max iters at generation reached"
            break
        end

        if newword ∉ existingwords
            push!(samples, newword)
            generated += 1
        else
            continue
        end
    end
    return samples
end

end
