### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 576e4d5e-ed48-11ed-045a-c1c22b01796b
begin
	import Pkg
	using Revise
	Pkg.activate()
	using StatsBase
	import Makemore as M
	const Flux = M.Flux
end

# ╔═╡ c0f8e7e5-2595-450f-9ce1-1e3af56fc7f8
begin
	using Plots
	using Statistics
end

# ╔═╡ 33f873a0-11ea-435e-9385-e5664f75e9d9
train, test = M.loaddatasets("../russian_names.txt")

# ╔═╡ 10a0834c-990a-4b05-8b9c-7653c96c3bd2
begin
	config = M.Config(blocksize = 3, vocabsize = length(train.chars) + 2)
	model = M.MLP(config)
	mylog = M.train_model!(model, train, test, 200)
end

# ╔═╡ 2ff357a7-cd5c-4521-8e6c-3a09d2d3dc15


# ╔═╡ 431f2842-ee6c-4516-bd13-15351b807d2d
mylog

# ╔═╡ a834d2ac-7f0e-4e4a-9b06-fdcf28a7b8c1
plot([mean(l.testloss) for l in mylog], )

# ╔═╡ a3909d30-303d-4cc0-8160-e2fb372481c8


# ╔═╡ 2638d2e9-9669-40c6-ad93-1a74c5e795e1
M.getsamples(model, train, test, 20)

# ╔═╡ 30026299-dc2f-4091-8bc1-d57872fb4021
train

# ╔═╡ a035e58c-c4ba-4cdf-a766-feba774697b9

M.evaluate(model, train)
	

# ╔═╡ Cell order:
# ╠═576e4d5e-ed48-11ed-045a-c1c22b01796b
# ╠═33f873a0-11ea-435e-9385-e5664f75e9d9
# ╠═10a0834c-990a-4b05-8b9c-7653c96c3bd2
# ╠═2ff357a7-cd5c-4521-8e6c-3a09d2d3dc15
# ╠═c0f8e7e5-2595-450f-9ce1-1e3af56fc7f8
# ╠═431f2842-ee6c-4516-bd13-15351b807d2d
# ╠═a834d2ac-7f0e-4e4a-9b06-fdcf28a7b8c1
# ╠═a3909d30-303d-4cc0-8160-e2fb372481c8
# ╠═2638d2e9-9669-40c6-ad93-1a74c5e795e1
# ╠═30026299-dc2f-4091-8bc1-d57872fb4021
# ╠═a035e58c-c4ba-4cdf-a766-feba774697b9
