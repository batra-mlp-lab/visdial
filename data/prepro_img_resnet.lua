require 'torch'
require 'nn'
require 'prepro_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text('Extract Image Features from Pretrained ResNet Models (t7 models)')
cmd:text()
cmd:text('Options')
cmd:option('-inputJson', 'visdial_params.json', 'Path to JSON file')
cmd:option('-imageRoot', '/path/to/images/', 'Path to COCO image root')
cmd:option('-cnnModel', '/path/to/t7/model', 'Path to Pretrained T7 Model')
cmd:option('-batchSize', 50, 'Batch size')

cmd:option('-outName', 'data_img.h5', 'Output name')
cmd:option('-gpuid', 0, 'Which gpu to use. -1 = use CPU')

cmd:option('-imgSize', 224)

opt = cmd:parse(arg)
print(opt)

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpuid + 1)
end

-------------------------------------------------------------------------------
-- Loading model and removing extra layers
-------------------------------------------------------------------------------
model = torch.load(opt.cnnModel);
-- Remove the last fully connected + softmax layer of the model
model:remove()
model:evaluate()

-------------------------------------------------------------------------------
-- Infer output dim
-------------------------------------------------------------------------------
local dummy_img = torch.DoubleTensor(1, 3, opt.imgSize, opt.imgSize)
model:forward(dummy_img)
local ndims = model.output:squeeze():size():totable()

if opt.gpuid >= 0 then
    model = model:cuda()
end

-------------------------------------------------------------------------------
-- Extract features and save to HDF
-------------------------------------------------------------------------------
extractFeatures(model, opt, ndims)
