# Makemore
This repo is a Julia rewrite of [original
makemore](https://github.com/karpathy/makemore) primarily meant for educational
purposes. This repo is meant for educational purposes, modern
optimizations to transformers or any Julia (or Flux)-specific optimizations are
not employed.
## Quickstart
1. Clone this repo, via `Pkg.add("https://github.com/isentropic/Makemore.jl")`
2. Run the `example.jl` script:
   ```julia
	# Train the model
	mylog = M.train_model!(model, train, test, 10)

	# Now sample the results
	M.generate(model, [1], config.vocabsize * 2)
	M.getsamples(model, train, test, 10)
	# 10-element Vector{Any}:
	#  "aimberli"
	#  "jaquin"
	#  "carmelly"
	#  "eurion"
	#  "robes"
	#  "davius"
	#  "drania"
	#  "phaneam"
	#  "alaniya"
	#  "manaya"
   ```

## Available models
Although Flux.jl offers a variety of models like RNNs and Transformers (through
Transformers.jl) this repo is a rudimentary write-up of these popular models.
Browsking through files like `src/transformer.jl` and `src/rnn.jl` could be
helpful for people getting into ML and Flux.  