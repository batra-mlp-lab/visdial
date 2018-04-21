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

function extractFeatures(model, opt, ndims, preprocessFn)
    local file = io.open(opt.inputJson, 'r')
    local text = file:read()
    file:close()
    jsonFile = cjson.decode(text)

    local trainList = {}
    for i, imName in pairs(jsonFile.unique_img_train) do
        table.insert(trainList, string.format('%s/%s', opt.imageRoot, imName))
    end

    if opt.train_split == 'train' then
        local valList = {}
        for i, imName in pairs(jsonFile.unique_img_val) do
            table.insert(valList, string.format('%s/%s', opt.imageRoot, imName))
        end
    elseif opt.train_split == 'trainval' then
        local testList = {}
        for i, imName in pairs(jsonFile.unique_img_test) do
            table.insert(testList, string.format('%s/%s', opt.imageRoot, imName))
        end
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
        trainFeats[{{i, r}, {}}] = model.output:float()
        collectgarbage()
    end

    print('\n')

    if opt.train_split == 'train' then

        local sz = #valList
        local valFeats = torch.FloatTensor(sz, unpack(ndims))
        print(string.format('Processing %d val images...', sz))
        for i = 1, sz, opt.batchSize do
            xlua.progress(i, sz)
            r = math.min(sz, i + opt.batchSize - 1)
            ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
            for j = 1, r - i + 1 do
                ims[j] = loadImage(valList[i + j - 1], opt.imgSize)
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
            valFeats[{{i, r}, {}}] = model.output:float()
            collectgarbage()
        end

        print('\n')

    elseif opt.train_split == 'trainval' then

        local sz = #testList
        local testFeats = torch.FloatTensor(sz, unpack(ndims))
        print(string.format('Processing %d test images...', sz))
        for i = 1, sz, opt.batchSize do
            xlua.progress(i, sz)
            r = math.min(sz, i + opt.batchSize - 1)
            ims = torch.DoubleTensor(r - i + 1, 3, opt.imgSize, opt.imgSize)
            for j = 1, r - i + 1 do
                ims[j] = loadImage(valList[i + j - 1], opt.imgSize)
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
            testFeats[{{i, r}, {}}] = model.output:float()
            collectgarbage()
        end
    end

    local h5File = hdf5.open(opt.outName, 'w')
    h5File:write('/images_train', trainFeats)
    if opt.train_split == 'train' then
        h5File:write('/images_val', valFeats)
    elseif opt.train_split == 'trainval' then
        h5File:write('/images_test', testFeats)
    end
    h5File:close()
end
