[![Build Status](https://github.com/isentropic/Makemore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/isentropic/Makemore.jl/actions/workflows/CI.yml?query=branch%3Amain)
# Makemore
This repo is a Julia rewrite of [original
makemore](https://github.com/karpathy/makemore) primarily meant for educational
purposes. This repo is meant for educational purposes, modern
optimizations to transformers or any Julia (or Flux)-specific optimizations are
not employed.
## What does this do?
Auto-regressive line-by-line text generation based on the input datafile. Perfect for coming up with new baby names based on the exisiting corpus of names. Currently the tokenization is character based, hence this might not work very well with composite characters like emojis or korean/chinese/japanese and alike characters.
## Quickstart
1. Clone this repo
2. Activate the provided julia environment
3. Include the `example.jl` script - `include("example.jl")`
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