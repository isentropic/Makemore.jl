import Makemore as M
# Load predefined datasets
train, test = M.loaddatasets("datasets/english_names.txt")

# Create confit for the language model
config = M.Config(blocksize=20, vocabsize=length(train.chars) + 2)
model = M.Transformer(config)
# OR other variants like
# M.RNN(config, "gru"); M.RNN(config, "gru")
# M.BoW(config, "gru")

# Next, prepare the dataloader
train_loader = M.get_dataloader(train)

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