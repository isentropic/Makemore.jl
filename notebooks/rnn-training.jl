import Pkg
using Revise
using StatsBase
import Makemore as M
const Flux = M.Flux

train, test = M.loaddatasets("names.txt")

config = M.Config(blocksize=3, vocabsize=length(train.chars) + 2)
model = M.RNN(config)
train_loader = M.get_dataloader(train)

config = M.Config(blocksize=3, vocabsize=length(train.chars) + 2)
model = M.RNN(config, "rnn")

x, y = first(train_loader)

mylog = M.train_model!(model, train, test, 10)


M.getsamples(model, train, test, 10)

model(x)

mlp = M.MLP(config)
M.train_model!(mlp, train, test, 10)
M.getsamples(mlp, train, test, 1)