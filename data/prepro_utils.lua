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

function extractFeaturesSplit(model, opt, ndims, preprocessFn, dtype)
    local file = io.open(opt.inputJson, 'r')
    local text = file:read()
    file:close()
    jsonFile = cjson.decode(text)

    local imList = {}
    for i, imName in pairs(jsonFile['unique_img_'..dtype]) do
        table.insert(imList, string.format('%s/%s', opt.imageRoot, imName))
    end

    local sz = #imList
    local imFeats = torch.FloatTensor(sz, unpack(ndims))

    -- feature_dims shall be either 2 (NW format), else 4 (having NCHW format)
    local feature_dims = #imFeats:size()

    print(string.format('Processing %d %s images...', sz, dtype))
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

    return imFeats
end

function extractFeatures(model, opt, ndims, preprocessFn)
    local h5File = hdf5.open(opt.outName, 'w')
    imFeats = extractFeaturesSplit(model, opt, ndims, preprocessFn, 'train')
    h5File:write('/images_train', imFeats)
    if opt.trainSplit == 'train' then
        imFeats = extractFeaturesSplit(model, opt, ndims, preprocessFn, 'val')
        h5File:write('/images_val', imFeats)
        imFeats = extractFeaturesSplit(model, opt, ndims, preprocessFn, 'test')
        h5File:write('/images_test', imFeats)
    elseif opt.trainSplit == 'trainval' then
        imFeats = extractFeaturesSplit(model, opt, ndims, preprocessFn, 'test')
        h5File:write('/images_test', imFeats)
    end
    h5File:close()
end
