require 'torch'
require 'nn'
npy4th = require 'npy4th'


local cmd = torch.CmdLine()
cmd:option('-input_units', 378)
cmd:option('-hidden_units', 1024)
cmd:option('-output_units', 44)
cmd:option('-size', 94306)
cmd:option('-output', 'loss/loss.npy')
cmd:option('-learning_rate', 0.01)
local opt = cmd:parse(arg)


-- read a .npy file into a torch tensor
X = npy4th.loadnpy('X.npy')
Y = npy4th.loadnpy('Y.npy')


dataset={};
function dataset:size() return opt.size end
for i=1,dataset:size() do 
  local input = X[i]
  local output = Y[i]
  dataset[i] = {input, output}
end


mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = opt.input_units; outputs = opt.output_units; HUs = opt.hidden_units; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))


criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = 5
trainer.learningRate = opt.learning_rate
trainer:train(dataset)


print(criterion:forward(mlp:forward(X), Y))

local loss = torch.sum(torch.pow(mlp:forward(X) - Y, 2), 2)

npy4th.savenpy(opt.output, loss)




