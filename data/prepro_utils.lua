require 'torch'
require 'math'
require 'nn'
require 'image'
require 'xlua'
require 'hdf5'
cjson = require('cjson')


function loadImage(imageName, imgSize, preprocessType)
    im = image.load(imageName)

    if im:size(1) == 1 then
        im = im:repeatTensor(3, 1, 1)
    elseif im:size(1) == 4 then
        im = im[{{1,3}, {}, {}}]
    end

    im = image.scale(im, imgSize, imgSize)
    return im
end

function extractFeaturesSplit(model, opt, ndims, preprocessFn, split)
    local file = io.open(opt.inputJson, 'r')
    local text = file:read()
    file:close()
    jsonFile = cjson.decode(text)

    local imList = {}
    if split == 'train' do
        for i, imName in pairs(jsonFile.unique_img_train) do
            table.insert(trainList, string.format('%s/train2014/COCO_train2014_%012d.jpg', opt.imageRoot, imName))
        end
    elseif split == 'val' do
        for i, imName in pairs(jsonFile.unique_img_val) do
            table.insert(valList, string.format('%s/val2014/COCO_val2014_%012d.jpg', opt.imageRoot, imName))
        end
    else
        for i, imName in pairs(jsonFile.unique_img_test) do
            table.insert(valList, string.format('%s/test2015/COCO_test2015_%012d.jpg', opt.imageRoot, imName))
        end
    end

    local sz = #imList
    local imFeats = torch.FloatTensor(sz, unpack(ndims))
    -- feature_dims shall be either 2 (NW format), else 4 (having NCHW format)
    local feature_dims = #imFeats:size()  
    print(string.format('Processing %d %s images...', sz, split))
    for i = 1, sz, opt.batchSize do
        xlua.progress(i, sz)
        r = math.min(sz, i + opt.batchSize - 1)
        ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
        for j = 1, r - i + 1 do
            ims[j] = loadImage(imList[i + j - 1], opt.imgSize)
            ims[j] = preprocessFn(ims[j])
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
        imFeats[{{i, r}, {}}] = model.output:float()
        collectgarbage()
    end
    print('\n')

    local h5File = hdf5.open(opt.outName, 'w')
    h5File:write(string.format('/images_%s', split), imFeats)
    h5File:close()
end

function extractFeatures(model, opt, ndims, preprocessFn)
    extractFeaturesSplit(model, opt, ndims, preprocessFn, 'train')
    extractFeaturesSplit(model, opt, ndims, preprocessFn, 'val')
    extractFeaturesSplit(model, opt, ndims, preprocessFn, 'test')
end
