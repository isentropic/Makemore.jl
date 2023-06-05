import Pkg
using Revise
using StatsBase
import Makemore as M
const Flux = M.Flux

train, test = M.loaddatasets("names.txt")

config = M.Config(blocksize=3, vocabsize=length(train.chars) + 2)
model = M.RNN(config)
train_loader = M.get_dataloader(train)

x, y = first(train_loader)

x[:, 1]

x
y

m = model
emb = m.wte(x) # (n_emb, t, b) opposite to torch
t, b = size(x)
hprev = hcat([m.start for i = 1:b]...)


hiddens = Flux.Zygote.Buffer(emb, m.cell.config.nembedding2, t, b)
for i in 1:t
    xt = emb[:, i, :]
    ht = m.cell(xt, hprev)
    hprev = ht
    hiddens[:, i, :] = hprev
end
hidden = copy(hiddens) # t, n_emb, b 
hidden
logits = m.lmhead(hidden)


M.loss(logits, y)