### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 80771a80-c6bf-11ed-1213-9d3bbdac53ee
begin
	import Pkg
	using Revise
	Pkg.activate()
end

# ╔═╡ c7f23ced-2639-43ef-b547-c800cba00f79
using Statistics

# ╔═╡ 03f4172a-07de-4647-a380-b5eba634d36b
using Plots

# ╔═╡ cb851261-f75f-4ff1-826e-958674dbae02


# ╔═╡ 71c17c1d-9494-433e-b818-66b40065fbc3
import Makemore as M

# ╔═╡ 8d1fd7d6-7acf-4312-a4ee-c1c342618196
test, train = M.loaddatasets("../names.txt")

# ╔═╡ 9931ef02-d44f-41e2-a4f4-0d12a213e4e1
begin
	config = M.Config(blocksize = 1, vocabsize = length(train.chars) + 1)
	model = M.Bigram(config)
end

# ╔═╡ b1e9663b-6fcc-4935-9a09-53709dc2605e
test[1]

# ╔═╡ 0fea5f13-6f0a-4446-b607-ecbb0931effd
config

# ╔═╡ fea9213d-b5d5-4204-b05f-6134e40da484
M.softmax(model(test[1][1]))

# ╔═╡ c03c6f76-da90-47a4-b32a-911c510e1d09
begin
	pred = model(test[1][1])
	real = M.Flux.onehotbatch(test[1][2], 1:27, 1)
	M.Flux.Losses.logitbinarycrossentropy(pred, real)
end

# ╔═╡ 1e360aca-ba2b-425d-b9b0-c3f2f149b484
train[1]

# ╔═╡ 19d62908-ad4e-4174-8276-71f5bef5444d
begin
	X = []
	Y = []
	
	for i in eachindex(train.words)
		x, y = train[i]
		push!(X, x)
		push!(Y, y)
	end

	X = hcat(X...)
	Y = hcat(Y...)
end

# ╔═╡ 00633b76-3f40-47ba-8c65-1c025c00386a
train_loader = M.Flux.DataLoader((X, Y), batchsize = 32)

# ╔═╡ ab38c72c-d64b-4a30-b874-3cabd835390d


# ╔═╡ 528b2e14-efef-4a0d-a841-9007cc00c448
const Flux = M.Flux

# ╔═╡ 1f19773f-5352-4960-a1fe-cf03916867c5
function loss(pred, y)
	real = M.Flux.onehotbatch(y, 1:27, 1)
	M.Flux.Losses.logitbinarycrossentropy(pred, real)
end

# ╔═╡ ab8176e0-60ef-486d-af64-648160697a6b
function loss(model, x, y)
	pred = model(x)
	loss(pred, y)
end

# ╔═╡ 86f555d7-81b0-466e-96ab-905629bde454
Y[:,10]

# ╔═╡ 1ee16e06-3e06-48d9-8c79-d36fd5c75d53
X[:, 10]

# ╔═╡ d43b31a4-c4ee-4f96-8145-b761b3722409
loss(model, X[:, 11], Y[:, 11])

# ╔═╡ 8970687b-a6c4-4ebe-838b-6829c60dbc35
opt_state = Flux.setup(Flux.Adam(), model)

# ╔═╡ d8de3426-db43-4757-a2f8-53b416630119
Flux.@functor M.Bigram (logits, )

# ╔═╡ 4d9bb267-832c-46fb-a9e3-6a0d186dc30a
begin
	my_log = []
	for epoch in 1:100
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
	
	    # Detect loss of Inf or NaN. Print a warning, and then skip update!
	    if !isfinite(val)
	      @warn "loss is $val on item $i" epoch
	      continue
	    end
	
	    Flux.update!(opt_state, model, grads[1])
	  end
	
	  # # Compute some accuracy, and save details as a NamedTuple
	  # acc = my_accuracy(model, train_set)
	  push!(my_log, (;losses))
	
	  # # Stop training when some criterion is reached
	  # if  acc > 0.95
	  #   println("stopping after $epoch epochs")
	  #   break
	  # end
	
	end
end

# ╔═╡ d696249f-9f36-41cb-b680-cb549bc5e4b3
my_log

# ╔═╡ a28f4631-5e89-4f7d-8a1b-6530d904972d
plot([mean(l.losses) for l in my_log], yscale=:log10)

# ╔═╡ f565059f-be94-48f1-84ed-1427e1139e21
Flux.softmax(model.logits * M.Flux.onehot(1, 1:27))

# ╔═╡ 4be5df73-326b-4201-b4b6-9fef40c5a96c
X[:, 1]

# ╔═╡ 18da3e30-4c5d-4bc2-9daa-faa47be8864a
model(X[:, 1])

# ╔═╡ 2c4ca89e-1cdf-45f4-9cc5-ec25fa77e427
model

# ╔═╡ Cell order:
# ╠═80771a80-c6bf-11ed-1213-9d3bbdac53ee
# ╠═cb851261-f75f-4ff1-826e-958674dbae02
# ╠═71c17c1d-9494-433e-b818-66b40065fbc3
# ╠═8d1fd7d6-7acf-4312-a4ee-c1c342618196
# ╠═9931ef02-d44f-41e2-a4f4-0d12a213e4e1
# ╠═b1e9663b-6fcc-4935-9a09-53709dc2605e
# ╠═0fea5f13-6f0a-4446-b607-ecbb0931effd
# ╠═fea9213d-b5d5-4204-b05f-6134e40da484
# ╠═c03c6f76-da90-47a4-b32a-911c510e1d09
# ╠═1e360aca-ba2b-425d-b9b0-c3f2f149b484
# ╠═19d62908-ad4e-4174-8276-71f5bef5444d
# ╠═00633b76-3f40-47ba-8c65-1c025c00386a
# ╠═ab38c72c-d64b-4a30-b874-3cabd835390d
# ╠═528b2e14-efef-4a0d-a841-9007cc00c448
# ╠═1f19773f-5352-4960-a1fe-cf03916867c5
# ╠═ab8176e0-60ef-486d-af64-648160697a6b
# ╠═86f555d7-81b0-466e-96ab-905629bde454
# ╠═1ee16e06-3e06-48d9-8c79-d36fd5c75d53
# ╠═d43b31a4-c4ee-4f96-8145-b761b3722409
# ╠═8970687b-a6c4-4ebe-838b-6829c60dbc35
# ╠═d8de3426-db43-4757-a2f8-53b416630119
# ╟─a86dae93-7577-4568-a14e-c0f9827125bb
# ╠═4d9bb267-832c-46fb-a9e3-6a0d186dc30a
# ╠═c7f23ced-2639-43ef-b547-c800cba00f79
# ╠═03f4172a-07de-4647-a380-b5eba634d36b
# ╠═d696249f-9f36-41cb-b680-cb549bc5e4b3
# ╠═a28f4631-5e89-4f7d-8a1b-6530d904972d
# ╠═f565059f-be94-48f1-84ed-1427e1139e21
# ╠═4be5df73-326b-4201-b4b6-9fef40c5a96c
# ╠═18da3e30-4c5d-4bc2-9daa-faa47be8864a
# ╠═2c4ca89e-1cdf-45f4-9cc5-ec25fa77e427
