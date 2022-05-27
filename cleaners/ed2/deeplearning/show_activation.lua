require 'torch'
require 'nn'

require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-file', '')
cmd:option('-output', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
if opt.verbose == 1 then print(msg) end

model:evaluate()


print(model.model_type)
print(model.wordvec_dim)
print(model.rnn_size)
print(model.num_layers)
print(model.dropout)
print(model.batchnorm)

-- print(model.view2)
-- print("start module")
-- print("1")
-- print(model.net:get(1))
-- print("2")
-- print(model.net:get(2))
-- print("3")
-- print(model.net:get(3))
-- print("4")
-- print(model.net:get(4))
-- print("5")
-- print(model.net:get(5))
-- print("stop module")


-- http://lua-users.org/wiki/FileInputOutput

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

-- tests the functions above
local file = opt.file
local lines = lines_from(file)


local max_string_length = -1
for k,v in pairs(lines) do
	local max = string.len(v)
	if max_string_length < max then
		max_string_length = max
	end
end

print("max: " .. max_string_length)

function formatString (str_value, char_fill, length)
	for i = #str_value + 1, length do
    	str_value = str_value .. char_fill
	end
	return str_value
end


function forward_all (model, all, part_id, output_path, time, num_layers)
	print("start forward: " .. time.hour .. ":" .. time.min .. ":" .. time.sec)

	local scores = model:forward(all)

	model:resetStates()

	local activations = model.net:get(num_layers + 1).output

	npy4th = require 'npy4th'
	npy4th.savenpy(output_path .. 'features' .. part_id .. '.npy', activations)

	-- print(activations)

	print("end: " .. time.hour .. ":" .. time.min .. ":" .. time.sec)
end

all = torch.Tensor()


time = os.date("*t")

local i = 0
local part = 0

local max_items = 1000

-- print all line numbers and their contents
for k,v in pairs(lines) do

	-- do string padding
	local padded_string = formatString (v, '\n', max_string_length)

    local encoded_string = model:encode_string(padded_string)

	model:resetStates()

	local x = encoded_string:view(1, -1)
	
	if i % max_items == 0 then
		all = x
	else
		all = all:cat(x, 1)
	end
	i = i + 1

	if i % 10000 == 0 then
		time = os.date("*t")
		print(i .. " at: " .. time.hour .. ":" .. time.min .. ":" .. time.sec)
	end

	if i % max_items == 0 then
		forward_all (model, all, part, opt.output, time, model.num_layers)
		part = part + 1
	end
end

if i % max_items ~= 0 then 
	forward_all (model, all, part, opt.output, time, model.num_layers)
end


