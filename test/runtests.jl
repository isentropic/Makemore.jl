using Makemore
using Test

import Makemore as M

@testset "Makemore.jl" begin
    train, test = M.loaddatasets("../datasets/english_names.txt")
    config = M.Config(blocksize=vocabsize = length(train.chars) + 2, vocabsize=length(train.chars) + 2)

    for modeltype in (M.Transformer, M.BoW, M.RNN, x -> M.RNN(x, "gru"),)
        println("Testing model: ", modeltype)
        model = modeltype(config)
        train_loader = M.get_dataloader(train)

        x, y = first(train_loader)
        model(x)

        M.generate(model, [1], config.vocabsize * 2)

        println("model inference passed")
    end
end
