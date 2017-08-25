require 'nn'
require 'xlua'
require 'math'
require 'hdf5'
require 'image'
require 'loadcaffe'
cjson = require('cjson')

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-inputJson', 'visdial_params.json', 'Path to JSON file')
cmd:option('-imageRoot', '/path/to/images/', 'Path to COCO image root')
cmd:option('-cnnProto', 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt', 'Path to the CNN prototxt')
cmd:option('-cnnModel', 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel', 'Path to the CNN model')
cmd:option('-batchSize', 20, 'Batch size')

cmd:option('-outName', 'data_img_pool5.h5', 'Output name')
cmd:option('-gpuid', 0, 'Which gpu to use. -1 = use CPU')
cmd:option('-backend', 'nn', 'nn|cudnn')

cmd:option('-imgSize', 448)
cmd:option('-layerName', 'pool5')

opt = cmd:parse(arg)
print(opt)

model = loadcaffe.load(opt.cnnProto, opt.cnnModel, opt.backend);
for i = #model.modules, 1, -1 do
    local layer = model:get(i)
    if layer.name == opt.layerName then break end
    model:remove()
end
model:evaluate()

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid+1)
    model = model:cuda()
end

function loadImage(imageName)
    im = image.load(imageName)

    if im:size(1) == 1 then
        im = im:repeatTensor(3, 1, 1)
    elseif im:size(1) == 4 then
        im = im[{{1,3}, {}, {}}]
    end

    im = image.scale(im, opt.imgSize, opt.imgSize)
    local meanPixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    im = im:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
    meanPixel = meanPixel:view(3, 1, 1):expandAs(im)
    im:add(-1, meanPixel)
    return im
end

local file = io.open(opt.inputJson, 'r')
local text = file:read()
file:close()
jsonFile = cjson.decode(text)

local trainList={}
for i, imName in pairs(jsonFile.unique_img_train) do
    table.insert(trainList, string.format('%s/train2014/COCO_train2014_%012d.jpg', opt.imageRoot, imName))
end

local valList={}
for i, imName in pairs(jsonFile.unique_img_val) do
    table.insert(valList, string.format('%s/val2014/COCO_val2014_%012d.jpg', opt.imageRoot, imName))
end

local batchSize = opt.batchSize

local sz = #trainList
local trainFeats = torch.FloatTensor(sz, 14, 14, 512)
print(string.format('Processing %d images...', sz))
for i = 1, sz, batchSize do
    xlua.progress(i, sz)
    r = math.min(sz, i + batchSize - 1)
    ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
    for j = 1, r - i + 1 do
        ims[j] = loadImage(trainList[i+j-1])
    end
    if opt.gpuid >= 0 then
        ims = ims:cuda()
    end
    model:forward(ims):permute(1, 3, 4, 2):contiguous():float()
    trainFeats[{{i, r}, {}}] = model.output:float()
    collectgarbage()
end

local sz = #valList
local valFeats = torch.FloatTensor(sz, 14, 14, 512)
print(string.format('Processing %d images...', sz))
for i = 1, sz, batchSize do
    xlua.progress(i, sz)
    r = math.min(sz, i + batchSize - 1)
    ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
    for j = 1, r - i + 1 do
        ims[j] = loadImage(valList[i+j-1])
    end
    if opt.gpuid >= 0 then
        ims = ims:cuda()
    end
    model:forward(ims):permute(1, 3, 4, 2):contiguous():float()
    valFeats[{{i, r}, {}}] = model.output:float()
    collectgarbage()
end

local h5File = hdf5.open(opt.outName, 'w')
h5File:write('/images_train', trainFeats)
h5File:write('/images_val', valFeats)
h5File:close()

