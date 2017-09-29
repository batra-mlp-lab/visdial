require 'torch'
require 'math'
require 'nn'
require 'image'
require 'xlua'
require 'hdf5'
cjson = require('cjson')


function loadImage(imageName, imgSize)
    im = image.load(imageName)

    if im:size(1) == 1 then
        im = im:repeatTensor(3, 1, 1)
    elseif im:size(1) == 4 then
        im = im[{{1,3}, {}, {}}]
    end

    im = image.scale(im, imgSize, imgSize)
    local meanPixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    im = im:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
    meanPixel = meanPixel:view(3, 1, 1):expandAs(im)
    im:add(-1, meanPixel)
    return im
end


function extractFeatures(model, opt, ndims)
    local file = io.open(opt.inputJson, 'r')
    local text = file:read()
    file:close()
    jsonFile = cjson.decode(text)

    local trainList = {}
    for i, imName in pairs(jsonFile.unique_img_train) do
        table.insert(trainList, string.format('%s/train2014/COCO_train2014_%012d.jpg', opt.imageRoot, imName))
    end

    local valList = {}
    for i, imName in pairs(jsonFile.unique_img_val) do
        table.insert(valList, string.format('%s/val2014/COCO_val2014_%012d.jpg', opt.imageRoot, imName))
    end

    local sz = #trainList
    local trainFeats = torch.FloatTensor(sz, unpack(ndims))
    -- feature_dims shall be either 2 (NW format), else 4 (having NCHW format)
    local feature_dims = #trainFeats:size()  
    print(string.format('Processing %d train images...', sz))
    for i = 1, sz, opt.batchSize do
        xlua.progress(i, sz)
        r = math.min(sz, i + opt.batchSize - 1)
        ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
        for j = 1, r - i + 1 do
            ims[j] = loadImage(trainList[i + j - 1], opt.imgSize)
        end
        if opt.gpuid >= 0 then
            ims = ims:cuda()
        end

        if feature_dims == 4 then
            -- forward pass and permute to get NHWC format
            model:forward(ims):permute(1, 3, 4, 2):contiguous():float()
        else
            model:forward(ims)
        end
        trainFeats[{{i, r}, {}}] = model.output:float()
        collectgarbage()
    end

    print('\n')

    local sz = #valList
    local valFeats = torch.FloatTensor(sz, unpack(ndims))
    print(string.format('Processing %d val images...', sz))
    for i = 1, sz, opt.batchSize do
        xlua.progress(i, sz)
        r = math.min(sz, i + opt.batchSize - 1)
        ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
        for j = 1, r - i + 1 do
            ims[j] = loadImage(valList[i + j - 1], opt.imgSize)
        end
        if opt.gpuid >= 0 then
            ims = ims:cuda()
        end

        if feature_dims == 4 then
            -- forward pass and permute to get NHWC format
            model:forward(ims):permute(1, 3, 4, 2):contiguous():float()
        else
            model:forward(ims)
        end
        valFeats[{{i, r}, {}}] = model.output:float()
        collectgarbage()
    end

    local h5File = hdf5.open(opt.outName, 'w')
    h5File:write('/images_train', trainFeats)
    h5File:write('/images_val', valFeats)
    h5File:close()
end
