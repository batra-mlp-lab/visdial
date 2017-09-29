require 'nn'
require 'loadcaffe'
require 'prepro_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text('Extract Image Features from Pretrained VGG16 Models (prototxt + caffemodel)')
cmd:text()
cmd:text('Options')
cmd:option('-inputJson', 'visdial_params.json', 'Path to JSON file')
cmd:option('-imageRoot', '/path/to/images/', 'Path to COCO image root')
cmd:option('-cnnProto', 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt', 'Path to the CNN prototxt')
cmd:option('-cnnModel', 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel', 'Path to the CNN model')
cmd:option('-batchSize', 50, 'Batch size')

cmd:option('-outName', 'data_img.h5', 'Output name')
cmd:option('-gpuid', 0, 'Which gpu to use. -1 = use CPU')
cmd:option('-backend', 'nn', 'nn|cudnn')

cmd:option('-imgSize', 224)
cmd:option('-layerName', 'relu7')

opt = cmd:parse(arg)
print(opt)

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

-------------------------------------------------------------------------------
-- Loading model and removing extra layers
-------------------------------------------------------------------------------
model = loadcaffe.load(opt.cnnProto, opt.cnnModel, opt.backend);

for i = #model.modules, 1, -1 do
    local layer = model:get(i)
    if layer.name == opt.layerName then break end
    model:remove()
end
model:evaluate()

if opt.gpuid >= 0 then
    model = model:cuda()
end

-------------------------------------------------------------------------------
-- Extract features and save to HDF
-------------------------------------------------------------------------------
local ndims = 4096
extractFeatures(model, opt, ndims)
