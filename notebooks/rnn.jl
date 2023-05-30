### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 8491c6de-fed3-11ed-18c3-17edd727c9be
begin
	import Pkg
	using Revise
	Pkg.activate()
	using StatsBase
	import Makemore as M
	const Flux = M.Flux
end

# ╔═╡ 098171fd-2346-4941-874d-a9ce6d728979
train, test = M.loaddatasets("../names.txt")

# ╔═╡ 8e1d564c-0571-4347-8e60-8a3ee4e375fe
length(train.chars)

# ╔═╡ 09975fc0-7492-4d59-b058-df3ee1491127
begin
	config = M.Config(blocksize = 3, vocabsize = length(train.chars) + 2)
	model = M.RNN(config)
end

# ╔═╡ b7c51b80-8791-480c-a436-6f2904e08d6a
train_loader = M.get_dataloader(train)

# ╔═╡ 233f6cbe-7ab7-4083-b7a7-5e3912e19ded
x, y = first(train_loader)

# ╔═╡ 40acf1ee-f311-4a5c-847d-7f04b571c48a
x[:, 1]

# ╔═╡ 694dcd66-bbdd-4c28-a1ce-aa4320cb34c6
size(x)

# ╔═╡ ac446c8c-a933-44e7-a4cb-e8d996dade02
model.wte(x)

# ╔═╡ Cell order:
# ╠═8491c6de-fed3-11ed-18c3-17edd727c9be
# ╠═098171fd-2346-4941-874d-a9ce6d728979
# ╠═8e1d564c-0571-4347-8e60-8a3ee4e375fe
# ╠═09975fc0-7492-4d59-b058-df3ee1491127
# ╠═b7c51b80-8791-480c-a436-6f2904e08d6a
# ╠═233f6cbe-7ab7-4083-b7a7-5e3912e19ded
# ╠═40acf1ee-f311-4a5c-847d-7f04b571c48a
# ╠═694dcd66-bbdd-4c28-a1ce-aa4320cb34c6
# ╠═ac446c8c-a933-44e7-a4cb-e8d996dade02
