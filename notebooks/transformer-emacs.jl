import Pkg
using Revise
using StatsBase
import Makemore as M
const Flux = M.Flux

train, test = M.loaddatasets("../names.txt")

config = M.Config(blocksize=3, vocabsize=length(train.chars) + 2)
model = M.BoW
train_loader = M.get_dataloader(train)

x, y = first(train_loader)

x[:, 1]

x
y

m = model
emb = m.wte(x) # (n_emb, t, b) opposite to torch
t, b = size(x)
hprev = hcat([m.start for i = 1:b]...)
