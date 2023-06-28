Base.@kwdef struct RNNCell
    config::Config
    xh2h::Dense
end
Flux.@functor RNNCell (xh2h,)

function RNNCell(config)
    xh2h = Dense(config.nembedding + config.nembedding2, config.nembedding2)
    return RNNCell(config, xh2h)
end

function (m::RNNCell)(x, h)
    xh = vcat(x, h)
    ht = Flux.tanh.(m.xh2h(xh))
    return ht
end

Base.@kwdef struct GRUCell
    config::Config
    xh2z::Dense
    xh2r::Dense
    xh2hbar::Dense
end
Flux.@functor GRUCell (xh2z, xh2r, xh2har)

function GRUCell(config)
    xh2z = Dense(config.nembedding + config.nembedding2, config.nembedding2)
    xh2r = Dense(config.nembedding + config.nembedding2, config.nembedding2)
    xh2hbar = Dense(config.nembedding + config.nembedding2, config.nembedding2)
    return GRUCell(config, xh2z, xh2r, xh2hbar)
end

function (m::GRUCell)(x, hprev)
    xh = vcat(x, hprev)
    r = Flux.sigmoid.(m.xh2r(xh))
    hprev_reset = r .* hprev

    xhr = vcat(x, hprev_reset)
    hbar = Flux.tanh.(m.xh2hbar(xhr))

    z = Flux.sigmoid.(m.xh2z(xh))
    ht = @. (1 - z) * hprev + z * hbar
    return ht
end

Base.@kwdef struct RNN{T<:Integer}
    blocksize::T
    vocabsize::T
    start::Matrix{Float32}
    wte::Embedding
    cell::Union{RNNCell,GRUCell}
    lmhead::Dense
    config::Config
end
Flux.@functor RNN (wte, cell, lmhead,)

function RNN(config, celltype="rnn")
    emb = Embedding(config.vocabsize, config.nembedding)
    cell = if celltype == "rnn"
        RNNCell(config)
    elseif celltype == "gru"
        GRUCell(config)
    else
        @error("unsupported cell")
    end
    return RNN(
        config.blocksize,
        config.vocabsize,
        zeros(Float32, config.nembedding2, 1),
        emb,
        RNNCell(config),
        Dense(config.nembedding2, config.vocabsize),
        config
    )
end
# Flux.@functor RNN (wte, cell, lmhead, start) including start causes Flux
# error, help needed.
# Luckily including start as a trainable param is not necessary


function (m::RNN)(index)
    emb = m.wte(index) # (n_emb, t, b) opposite to torch
    t, b = size(index)
    # Somebody will make this better (better to avoid copying)
    hprev = hcat([m.start for i = 1:b]...)
    hiddens = Flux.Zygote.Buffer(emb, m.cell.config.nembedding2, t, b)
    for i in 1:t
        xt = emb[:, i, :]
        ht = m.cell(xt, hprev)
        hprev = ht
        hiddens[:, i, :] = ht
    end
    hidden = copy(hiddens) # t, n_emb, b
    logits = m.lmhead(hidden)
    # @show size(hidden), size(logits)
    return logits
end
