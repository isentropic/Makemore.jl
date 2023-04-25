Base.@kwdef struct Bigram
    config::Config
    n::Int64
    logits::Matrix{Float64}
end

# Declare logits to be trainable for Flux
# This macro lets Flux track "what to differentiate against"
Flux.@functor Bigram (logits,)

function Bigram(config)
    n = config.vocabsize
    return Bigram(config, n, zeros(n, n))
end

# Forward pass, basically just a lookup
(m::Bigram)(index) = m.logits[:, index]


Base.@kwdef struct MLP
    config::Config
    blocksize::Int64
    vocabsize::Int64
    embedding::Embedding
    mlp::Chain
end

Flux.@functor MLP (embedding, mlp)

function MLP(config)
    emb = Embedding(config.vocabsize, config.nembedding)
    mlp = Chain(
        Dense(config.blocksize * config.nembedding, config.nembedding2, tanh),
        Dense(config.nembedding2, config.vocabsize)
    )
    return MLP(config, config.blocksize, config.vocabsize, emb, mlp)
end

function zeropadshift(emb, units)
	if ndims(emb) == 3
		zeropadding = Flux.fill_like(emb, Int(starttoken), (size(emb)[1], units, size(emb)[end]))
	else
		zeropadding = Flux.fill_like(emb, Int(starttoken), (size(emb)[1], units))
	end
	shifted = circshift(emb, (0, units))[:, (units+1):end, :]
	return hcat(zeropadding, shifted)
end

function (m::MLP)(index)
    emb = m.embedding(index)
    x = vcat([zeropadshift(emb, i) for i in 0:m.blocksize-1]...)
    logits = m.mlp(x)
    return logits
end


function loss(model, x, y)
	real = Flux.onehotbatch(y, 1:model.config.vocabsize, 1)
	Flux.Losses.logitbinarycrossentropy(model(x), real)
end

function loss(pred, y)
	real = M.Flux.onehotbatch(y, 1:28, 1)
	M.Flux.Losses.logitbinarycrossentropy(pred, real)
end


function generate(model, indices, maxnewtokens; tempertaure=1.0)
    for _ in 1:maxnewtokens
        logits = model(indices[end]) # only last is needed for bigram
        probs = Flux.softmax(logits, dims=1)
        nextletter = sample(Weights(probs))

        indices = vcat(indices, nextletter)
    end

    return indices
end


function getsamples(model, dataset, num=10)
    samples = []
    for _ in 1:num
        start = [Int(starttoken), ]
        indices = generate(model, start, model.config.vocabsize)

        push!(samples, decode(dataset, indices))
    end
    return samples
end
