require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'
require 'math'
ffi = require 'ffi'
torch.setdefaulttensortype('torch.FloatTensor')


-- Process command line arguments, load helper functions
paths.dofile('opts.lua')
paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')

-- Random number seed
if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
else torch.seed() end                           

-- Initialize dataset
if not dataset then
    local Dataset = paths.dofile(projectDir .. '/src/util/dataset/' .. opt.dataset .. '.lua')
    dataset = Dataset()
end

-- Global reference (may be updated in the task file below)
if not ref then
    ref = {}
    ref.nOutChannels = #dataset.skeletonRef * 3
    ref.inputDim = {3, opt.inputRes, opt.inputRes}
    ref.outputDim = {ref.nOutChannels, opt.outputRes, opt.outputRes}
    ref.nJoints = 17
end

-- Load up task specific variables / functions
paths.dofile('util/' .. 'pose' .. '.lua')


-- Print out input / output tensor sizes
if not ref.alreadyChecked then
    print("# of validation images:", opt.nValidImgs)
    ref.alreadyChecked = true
end
