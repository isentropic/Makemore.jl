### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ecaa4b38-e2bd-11ed-3531-87982cbbf375
begin
	import Pkg
	using Revise
	Pkg.activate()
end

# ╔═╡ 503840b6-dc7e-4290-9b81-7abcec665d04
using StatsBase

# ╔═╡ 699830dd-70ad-453b-954b-aa43dd37d4a9
begin
	using Plots
	using Statistics
end

# ╔═╡ 5dd3a74b-41a1-4a11-b2af-ced34bd102f4
import Makemore as M

# ╔═╡ e76beb40-c24d-4ffd-a0ea-19e8e8b512a6
const Flux = M.Flux

# ╔═╡ bfd18072-6105-4bbf-b11f-b729581cc084
train, test = M.loaddatasets("../names.txt")

# ╔═╡ d4950e3e-599b-4e86-9ea3-f4f0a774003e
begin
	config = M.Config(blocksize = 3, vocabsize = length(train.chars) + 2)
	model = M.MLP(config)
end

# ╔═╡ d18cf387-dc03-4c77-9e6d-ae5db56f93de
train_loader = M.get_dataloader(train)

# ╔═╡ 3fcc15b2-de39-48de-a202-a52637596cb6
repeat(model.embedding([Int(M.starttoken)]), 1, 0)

# ╔═╡ 3ea0763b-2595-425a-973a-a3d5563f1303
train[1][1]

# ╔═╡ 067ba152-d290-4e3a-a8fa-32292c94a966
begin
	opt_state = Flux.setup(Flux.Adam(), model)
	my_log = []
	for epoch in 1:20
	  losses = Float32[]
	  for (x, y) in train_loader
	
	    val, grads = Flux.withgradient(model) do m
	      # Any code inside here is differentiated.
	      # Evaluation of the model and loss must be inside!
	      result = m(x)
	      M.loss(result, y)
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

# ╔═╡ d36a1df3-baea-41b9-b773-255ba2b7b5d6
plot([mean(l.losses) for l in my_log], )

# ╔═╡ 88e8c11c-afaa-40ac-8b61-273ac4b0542e
my_log[end].losses[end]

# ╔═╡ 3668800c-77ce-46bb-a050-fe855e547aba


# ╔═╡ b611c730-2dd4-4f02-9c2e-d8c163f6ce96


# ╔═╡ 90953fc0-edd8-4e73-973b-09e3f823637d
M.getsamples(model, train, 20)

# ╔═╡ Cell order:
# ╠═ecaa4b38-e2bd-11ed-3531-87982cbbf375
# ╠═503840b6-dc7e-4290-9b81-7abcec665d04
# ╠═5dd3a74b-41a1-4a11-b2af-ced34bd102f4
# ╠═e76beb40-c24d-4ffd-a0ea-19e8e8b512a6
# ╠═bfd18072-6105-4bbf-b11f-b729581cc084
# ╠═d4950e3e-599b-4e86-9ea3-f4f0a774003e
# ╠═d18cf387-dc03-4c77-9e6d-ae5db56f93de
# ╠═3fcc15b2-de39-48de-a202-a52637596cb6
# ╠═3ea0763b-2595-425a-973a-a3d5563f1303
# ╠═067ba152-d290-4e3a-a8fa-32292c94a966
# ╠═699830dd-70ad-453b-954b-aa43dd37d4a9
# ╠═d36a1df3-baea-41b9-b773-255ba2b7b5d6
# ╠═88e8c11c-afaa-40ac-8b61-273ac4b0542e
# ╠═3668800c-77ce-46bb-a050-fe855e547aba
# ╠═b611c730-2dd4-4f02-9c2e-d8c163f6ce96
# ╠═90953fc0-edd8-4e73-973b-09e3f823637d
