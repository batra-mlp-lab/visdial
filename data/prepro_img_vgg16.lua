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
cmd:option('-trainSplit', 'train', 'Which split to use: train | trainval')
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

-------------------------------------------------------------------------------
-- Infering output dim
-------------------------------------------------------------------------------
local dummy_img = torch.DoubleTensor(1, 3, opt.imgSize, opt.imgSize)

if opt.gpuid >= 0 then
    dummy_img = dummy_img:cuda()
    model = model:cuda()
end

model:forward(dummy_img)
local ndims = model.output:squeeze():size():totable()

-------------------------------------------------------------------------------
-- Defining function for image preprocessing, like mean subtraction
-------------------------------------------------------------------------------
function preprocessFn(im)
    -- mean pixel for caffemodels trained on imagenet
    local meanPixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    im = im:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
    meanPixel = meanPixel:view(3, 1, 1):expandAs(im)
    im:add(-1, meanPixel)
    return im
end

-------------------------------------------------------------------------------
-- Extract features and save to HDF
-------------------------------------------------------------------------------
extractFeatures(model, opt, ndims, preprocessFn)
