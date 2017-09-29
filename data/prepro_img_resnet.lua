require 'torch'
require 'prepro_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text('Extract Image Features from Pretrained ResNet Models')
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
model:remove(#model.modules)
model:evaluate()

if opt.gpuid >= 0 then
    model = model:cuda()
end

-------------------------------------------------------------------------------
-- Extract features and save to HDF
-------------------------------------------------------------------------------
local ndims = 2048
extractFeatures(model, opt, ndims)
