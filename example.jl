import Pkg
using Revise
using StatsBase
import Makemore as M
const Flux = M.Flux

train, test = M.loaddatasets("names.txt")

config = M.Config(blocksize=20, vocabsize=length(train.chars) + 2)
model = M.Transformer(config)
# OR other variants like
# M.RNN(config, "gru"); M.RNN(config, "gru")
# M.BoW(config, "gru")
train_loader = M.get_dataloader(train)

x, y = first(train_loader)

model(x)
mylog = M.train_model!(model, train, test, 10)

M.generate(model, [1], config.vocabsize * 2)
M.getsamples(model, train, test, 10)
